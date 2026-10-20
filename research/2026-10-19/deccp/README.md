# Five-case DECCP review

Download [review.html](review.html) and open it in a browser. It is standalone:
no server, account or network requests are required. GitHub shows HTML source
rather than running the viewer. [review.json](review.json) contains the same data.

Select a case for all four full saved responses. Select a branch and step for
its NLA description and editor proposal. There is a changed-edits-only filter.
Expand the context sections for the exact instruction, generated prefix and
editor message. The prefix is for human review, not separately sent to the editor.

Use **Read in** to switch between **Original**, **English**, and **Original +
English**. The selection applies to questions, responses, NLA descriptions,
edits, generated prefixes, and editor context. Switching language retains the
selected case, branch, step, and changed-edits filter. All translations are
embedded in the standalone HTML, so reading it works offline.

The four Chinese prompts and sixteen Chinese responses have close English
translations checked against the source. Step text uses offline
[OPUS-MT Chinese–English](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)
with reviewed terminology and fragment glosses. Repetition, awkward phrasing,
contradictions, and cutoff fragments are retained. Bracketed glosses explain
isolated tokens. The Original view preserves the saved text and exact editor
message. [review.en.json](review.en.json) holds the English lookup;
[translation-overrides.json](translation-overrides.json) holds the reviewed
translations and corrections. The original `review.json` is unchanged.

This snapshot does not refresh from HPC. All five cases and all 20 branches are finished. Reaching 512 tokens counts as finished. Original has no NLA call;
roundtrip has no editor call. Missing stages are marked. This is not a graded result.

Refresh from operational checkpoints:

```sh
python -m tools.export_deccp_viewer --run PATH_TO_CAMPAIGN --output PATH_TO_REVIEW.html
```

The exporter embeds the adjacent `PATH_TO_REVIEW.en.json` when present and
rejects stale translations. To rebuild English after changing the source data,
use a local CTranslate2 model and an intermediate cache outside the repository:

```sh
uv run --python 3.12 --with ctranslate2 --with sentencepiece \
  python -m tools.translate_deccp_review \
  --review research/2026-10-19/deccp/review.json \
  --model PATH_TO_OPUS_MT_CT2_MODEL \
  --cache PATH_TO_SEGMENT_CACHE.json \
  --overrides research/2026-10-19/deccp/translation-overrides.json
```

Model conversion:
`gaudi/opus-mt-zh-en-ctranslate2` at
`05d8fc158397bae0c65b8d46c858b6c18e094c12`.
Model weights remain outside Git. The translation tool checks coverage and
literal numbers; the exporter checks that translations match the source text.

Keep regenerated experimental data on this development branch, not main.

Click a token in the inspector to select its NLA step and highlight its position in the original response. Step selection and Previous/Next also update the highlight. Token highlighting uses the saved original boundaries; translated prose has no one-to-one token alignment.
