"""Model-specific HF NLA inference, loaded only inside the HPC runtime.

No model text is printed. Base, AV and AR are loaded one at a time with NF4
weights and FP32 compute; this quantization is recorded in validation metadata.
"""
import gc
import json
from pathlib import Path
import re
from stepper import metrics


def injection_position(ids, metadata):
    tokens = metadata['tokens']
    matches = [i for i in range(1, len(ids)-1)
               if ids[i] == tokens['injection_token_id']
               and ids[i-1] == tokens['injection_left_neighbor_id']
               and ids[i+1] == tokens['injection_right_neighbor_id']]
    if len(matches) != 1:
        raise ValueError('AV injection marker/neighbors must match exactly once')
    return matches[0]


def validate_contract(profile, metadata, role):
    layer = metadata.get('extraction_layer_index', metadata.get('critic', {}).get('extraction_layer_index'))
    if metadata['d_model'] != profile['dimension'] or layer != profile['layer']:
        raise ValueError('Model and NLA dimensions/extraction layer differ')
    if metadata.get('role') != role:
        raise ValueError('Incorrect NLA role metadata')
    if role == 'av' and not metadata['extraction']['injection_scale'] > 0:
        raise ValueError('Invalid AV injection scale')


class Backend:
    def __init__(self, profile, directory):
        import torch
        import yaml
        from transformers import AutoTokenizer
        self.torch = torch
        self.profile = profile
        self.directory = Path(directory)
        self.model = self.role = self.head = None
        self.tokenizers = {role: AutoTokenizer.from_pretrained(self.directory/role, local_files_only=True,
                           trust_remote_code=False) for role in ('base', 'av', 'ar')}
        self.metadata = {}
        for role in ('av', 'ar'):
            metadata = yaml.safe_load((self.directory/role/'nla_meta.yaml').read_text())
            validate_contract(profile, metadata, role)
            self.metadata[role] = metadata
        self.precision = 'bitsandbytes-NF4-double-quant; FP32 nonquantized tensors and compute'

    def load(self, role):
        if self.role == role:
            return self.model
        torch = self.torch
        self.model = self.head = None
        self.role = None
        gc.collect()
        torch.cuda.empty_cache()
        from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, Gemma3ForConditionalGeneration
        from safetensors.torch import load_file
        directory = self.directory/role
        config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        text_config = getattr(config, 'text_config', config)
        if text_config.hidden_size != self.profile['dimension']:
            raise ValueError('Checkpoint hidden size mismatch')
        if role == 'ar' and text_config.num_hidden_layers != self.profile['layer']+1:
            raise ValueError('AR must contain exactly extraction layer plus one blocks')
        maximum = {index: int(torch.cuda.mem_get_info(index)[0]*0.88)
                   for index in range(torch.cuda.device_count())}
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)
        cls = Gemma3ForConditionalGeneration if config.model_type == 'gemma3' else AutoModelForCausalLM
        model = cls.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                    torch_dtype=torch.float32, quantization_config=quantization,
                                    device_map='auto', max_memory=maximum,
                                    attn_implementation='eager', low_cpu_mem_usage=True)
        if any(device in ('cpu','disk') for device in getattr(model,'hf_device_map',{}).values()):
            raise MemoryError('Insufficient GPU memory for this checkpoint')
        if hasattr(model, 'language_model'):
            nested = model.language_model
            if hasattr(nested, 'lm_head'):
                model = nested
            else:
                with torch.device('meta'):
                    wrapper = AutoModelForCausalLM.from_config(text_config)
                wrapper.model = nested
                if not text_config.tie_word_embeddings:
                    raise ValueError('Wrapped Gemma text head must tie embeddings')
                wrapper.tie_weights()
                model = wrapper
        if role == 'ar':
            model.lm_head = torch.nn.Identity()
            model.model.norm = torch.nn.Identity()
            head = load_file(str(directory/'value_head.safetensors'))
            if set(head) != {'weight'} or tuple(head['weight'].shape) != (self.profile['dimension'],)*2:
                raise ValueError('Invalid AR value head')
            self.head = head['weight'].float().cpu()
        self.model = model.eval()
        self.role = role
        return self.model

    def input_tensor(self, model, ids):
        torch = self.torch
        return torch.tensor([ids], dtype=torch.long, device=model.get_input_embeddings().weight.device)

    def distribution(self, logits):
        torch = self.torch
        probabilities = torch.softmax(logits.float(), dim=-1)
        values, indices = torch.topk(probabilities, 30)
        if not torch.isfinite(probabilities).all():
            raise ValueError('Nonfinite model probability')
        return {'completion_probabilities': [{'id': int(indices[0]), 'top_logprobs': [
                    {'id': int(index), 'prob': float(value)} for index, value in zip(indices, values)]}],
                'top_probability_mass': float(values.sum())}

    def predict(self, ids, target=None):
        torch = self.torch
        model = self.load('base')
        selected = []
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            selected.append(tensor[0, -1].detach().float().cpu())
            if target is not None:
                tensor = tensor.clone()
                tensor[0, -1] = torch.as_tensor(target, device=tensor.device, dtype=tensor.dtype)
                return (tensor, *output[1:]) if isinstance(output, tuple) else [tensor, *output[1:]] if isinstance(output, list) else tensor
        handle = model.model.layers[self.profile['layer']].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                result = model(input_ids=self.input_tensor(model, ids), use_cache=False, logits_to_keep=1)
                prediction = self.distribution(result.logits[0, -1])
        finally:
            handle.remove()
        if len(selected) != 1 or not torch.isfinite(selected[0]).all():
            raise ValueError('Invalid extracted activation')
        return selected[0], prediction

    def reconstruct(self, explanation):
        torch = self.torch
        model = self.load('ar')
        metadata = self.metadata['ar']
        text = metadata['prompt_templates']['ar'].format(explanation=explanation)
        ids = self.tokenizers['ar'].encode(text, add_special_tokens=True)
        suffix = metadata['tokens'].get('critic_suffix_ids')
        if suffix and ids[-len(suffix):] != suffix:
            raise ValueError('AR suffix/tokenizer mismatch')
        with torch.inference_mode():
            out = model.model(input_ids=self.input_tensor(model, ids), use_cache=False)
            vector = torch.nn.functional.linear(out.last_hidden_state[0, -1].float().cpu(), self.head)
        if not torch.isfinite(vector).all():
            raise ValueError('Nonfinite reconstruction')
        return vector

    def explain(self, activation):
        torch = self.torch
        model = self.load('av')
        metadata = self.metadata['av']
        content = metadata['prompt_templates']['av'].format(injection_char=metadata['tokens']['injection_char'])
        tokenizer = self.tokenizers['av']
        ids = tokenizer.apply_chat_template([{'role':'user','content':content}], tokenize=True, add_generation_prompt=True)
        position = injection_position(ids, metadata)
        vector = torch.as_tensor(activation, dtype=torch.float32)
        if vector.numel() != self.profile['dimension'] or not torch.isfinite(vector).all() or not vector.norm() > 0:
            raise ValueError('Invalid AV activation')
        vector = vector*(metadata['extraction']['injection_scale']/vector.norm())
        # Hook AFTER the embedding module's own Gemma sqrt(d) scaling. The AV
        # sees precisely the sidecar's requested norm, without a second scaling.
        def hook(module, inputs, output):
            if output.shape[1] == len(ids):
                output = output.clone()
                output[0, position] = vector.to(device=output.device, dtype=output.dtype)
            return output
        handle = model.get_input_embeddings().register_forward_hook(hook)
        try:
            with torch.inference_mode():
                generated = model.generate(input_ids=self.input_tensor(model, ids), do_sample=False,
                                           max_new_tokens=240, use_cache=True)
        finally:
            handle.remove()
        text = tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True)
        parsed = re.search(r'<explanation>\s*(.*?)\s*</explanation>', text, re.DOTALL)
        if parsed is None:
            raise ValueError('AV explanation missing closing tags')
        return parsed.group(1).strip()

    def run(self, action, **body):
        torch = self.torch
        if action == 'start':
            tokenizer = self.tokenizers['base']
            ids = tokenizer.apply_chat_template([{'role':'user','content':body['prompt']}], tokenize=True, add_generation_prompt=True)
            model = self.load('base')
            eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
            return {'tokens': ids, 'eos_token_ids': eos if isinstance(eos, list) else [eos]}
        ids = body['tokens']
        if not ids or len(ids) > 1900 or any(type(value) is not int for value in ids):
            raise ValueError('Invalid token context')
        if action == 'inspect':
            activation, prediction = self.predict(ids)
            return {'h': activation.tolist(), 'baseline': prediction}
        if action == 'decode':
            return {'content': self.tokenizers['base'].decode(ids, skip_special_tokens=True)}
        if action == 'explain':
            text = self.explain(body['h'])
            reconstruction = self.reconstruct(text)
            original = torch.tensor(body['h'], dtype=torch.float32)
            error = original-reconstruction
            if not torch.isfinite(error).all():
                raise ValueError('Nonfinite reconstruction error')
            return {'text':text, 'reconstruction':reconstruction.tolist(), 'error':error.tolist(),
                    'metrics':metrics(original.tolist(),reconstruction.tolist())}
        if action == 'compare':
            original_reconstruction = torch.tensor(body['reconstruction'], dtype=torch.float32)
            edited_reconstruction = self.reconstruct(body['edited'])
            original, baseline = self.predict(ids)
            if original.shape != original_reconstruction.shape or original.shape != edited_reconstruction.shape:
                raise ValueError('Wrong reconstruction shape')
            corrected = original+(edited_reconstruction-original_reconstruction)
            if not torch.isfinite(corrected).all():
                raise ValueError('Nonfinite corrected activation')
            return {'original':baseline, 'roundtrip':self.predict(ids, original_reconstruction)[1],
                    'edited_raw':self.predict(ids, edited_reconstruction)[1],
                    'edited_corrected':self.predict(ids, corrected)[1],
                    'error':(original-original_reconstruction).tolist(),
                    'edited_reconstruction':edited_reconstruction.tolist(),
                    'corrected_activation':corrected.tolist(),
                    'metrics':metrics(original.tolist(),original_reconstruction.tolist())}
        raise ValueError('Unknown HF backend action')

    def validate(self):
        torch = self.torch
        start = self.run('start', prompt='Describe a small blue square in one sentence.')
        inspected = self.run('inspect', tokens=start['tokens'])
        explained = self.run('explain', tokens=start['tokens'], h=inspected['h'])
        compared = self.run('compare', tokens=start['tokens'], reconstruction=explained['reconstruction'], edited=explained['text'])
        def probabilities(prediction):
            return {item['id']:item['prob'] for item in prediction['completion_probabilities'][0]['top_logprobs']}
        baseline = probabilities(compared['original'])
        corrected = probabilities(compared['edited_corrected'])
        difference = max(abs(value-corrected.get(index,0)) for index,value in baseline.items())
        report = {'model':self.profile['id'], 'layer':self.profile['layer'], 'dimension':len(inspected['h']),
                  'precision':self.precision,'unchanged_corrected_max_probability_difference':difference,
                  'eos_token_ids':start['eos_token_ids'], 'all_vectors_finite':True,
                  'cuda_devices':[{ 'name':torch.cuda.get_device_name(index),
                     'total_bytes':torch.cuda.get_device_properties(index).total_memory,
                     'peak_allocated_bytes':torch.cuda.max_memory_allocated(index)} for index in range(torch.cuda.device_count())],
                  'passed':difference<1e-5 and len(inspected['h'])==self.profile['dimension']}
        if not report['passed']:
            raise ValueError('Unchanged-text numeric control failed')
        return report
