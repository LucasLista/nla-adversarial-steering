# Preliminary results, October 19, 2026

The saved evaluation covers benign requests only. It does not yet measure
whether activation editing increases harmful compliance.

## Graded responses

The October 18 evaluation contains 178 completed responses from 46 benign
cases. Of these, 42 cases have all four branches available. Comparing those
same 42 cases avoids changing the sample between conditions:

| Branch | Refusals / matched cases | Refusal rate | Responses at token cap |
| --- | --- | --- | --- |
| Original | 1 / 42 | 2.4% | 41 / 42 |
| Reconstructed | 2 / 42 | 4.8% | 40 / 42 |
| Edited | 0 / 42 | 0.0% | 42 / 42 |
| Edited with error correction | 1 / 42 | 2.4% | 42 / 42 |

Here refusal on a benign request is an over-refusal. The judge is
`Llama3RefusalJudge`, using `meta-llama/Meta-Llama-3-8B-Instruct` in BF16,
temperature 0, and a 16-token judging limit. The recorded evaluator source commit
is `c6337e5a2648ae33b83a63c620828d0da29c354b`.

The error-corrected branch removes one original refusal and introduces another.
Its paired refusal-rate difference is zero; the saved bootstrap 95% interval
is −7.1 to +7.1 percentage points. Edited responses have one fewer refusal than
the original branch, with a paired interval of −7.1 to 0 points. These are too
few refusals to support a general claim of improvement. The 128-token cap also
means the judge often sees truncated responses.

[Evaluation metadata](results-metadata.json) contains the denominators,
intervals, unmatched totals, and evaluator provenance. It is a filtered copy of
`runs/evaluations/jbb-20261018/jbb-evaluation-results.json` on HPC. The adjacent
`jbb-evaluation-grades.json` associates labels with state/config hashes;
`jbb-grader-source.zip` preserves the evaluator. No judge was rerun for this note.

## Generation progress

Observed October 19 at 15:35 Europe/Copenhagen, using metadata from the laptop
controllers and HPC. The original four-shard campaign planned 190 cases:

| Campaign | Cases with progress | All four branches complete | Branches complete: original / reconstructed / edited / corrected |
| --- | --- | --- | --- |
| Original campaign, 128-token cap | 46 / 190 | 42 | 46 / 46 / 44 / 42 |
| Harmful campaign, 256-token cap | 16 / 90 | 0 | 16 / 16 / 3 / 1 |

The harmful campaign has 16 shards. Shard 3 continues in
`runs/jbb-harmful-shard-3`; the others use
`runs/jbb-harmful-expanded-shard-N`. Manifests live under
`runs/benchmarks/overnight/manifest-harmful-expanded-N.json`. The original
campaign uses `runs/jbb-overnight-shard-N` and `manifest-N.json` in that directory.

Three NLA jobs were running and thirteen were queued at the initial scheduler
check. Local progress was ahead of the HPC copies for several cases. The table
uses the local snapshot, preserved in [progress metadata](progress-2026-10-19.json).
A completed branch can have reached a token cap or another stop condition;
completion alone says nothing about whether steering worked.

The archived grading has zero harmful responses. New harmful generations should
be graded separately, with the token limit and case selection recorded. Do not
combine the 128- and 256-token campaigns into one comparison.

## Before writing the article

The supported result today is a working comparison pipeline and a small benign
refusal evaluation. The harmful-compliance question remains open. A LessWrong
write-up should distinguish that engineering result from evidence for semantic
control, show matched conditions, and report truncation and failure counts.
Human-reviewed examples can illustrate the mechanism once selected; they should
not substitute for the aggregate evaluation.
