import unittest
from nla.stepper import run


class FastStepperTests(unittest.TestCase):
    def test_selected_predictions_match_full_comparison(self):
        calls=[]
        def call(role, endpoint, request):
            calls.append((role, endpoint))
            if endpoint=='/reconstruct':return {'activation':[2.,3.]}
            if endpoint=='/extract':return {'activation':[1.,2.]}
            if endpoint=='/completion':return {'synthetic_prediction':request.get('steering')}
            raise AssertionError(endpoint)
        body={'action':'compare','tokens':[1,2],'reconstruction':[.5,1.], 'edited':'Synthetic description.'}
        full=run(body,call,'')
        for variant in ('original','roundtrip','edited_raw','edited_corrected'):
            calls.clear()
            selected=run(dict(body,variants=[variant]),call,'')
            self.assertEqual(selected[variant],full[variant])
            self.assertEqual(calls.count(('base','/completion')),1)
            self.assertEqual(calls.count(('ar','/reconstruct')),int(variant.startswith('edited')))

    def test_inspect_skips_unused_prediction(self):
        def call(role,endpoint,request):
            self.assertEqual(endpoint,'/extract')
            return {'activation':[1.,2.]}
        self.assertEqual(run({'action':'inspect','tokens':[1],'baseline':False},call,''),{'h':[1.,2.]})
