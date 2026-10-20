# Five-case DECCP review

Download [review.html](review.html) and open it in a browser. It is standalone:
no server, account or network requests are required. GitHub shows HTML source
rather than running the viewer. [review.json](review.json) contains the same data.

Select a case for all four full saved responses. Select a branch and step for
its NLA description and editor proposal. There is a changed-edits-only filter.
Expand the context sections for the exact instruction, generated prefix and
editor message. The prefix is for human review, not separately sent to the editor.

This snapshot does not refresh from HPC. All five cases and all 20 branches are finished. Reaching 512 tokens counts as finished. Original has no NLA call;
roundtrip has no editor call. Missing stages are marked. This is not a graded result.

Refresh from operational checkpoints:

```sh
python -m tools.export_deccp_viewer --run PATH_TO_CAMPAIGN --output PATH_TO_REVIEW.html
```

Keep regenerated experimental data on this development branch, not main.
