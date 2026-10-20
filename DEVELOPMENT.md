# Development snapshot — 19 October 2026

For Jonathan and Lucas to review the ongoing NLA experiment. This is a working
snapshot, not a proposal to merge preliminary data into main.

- [Five-case DECCP response and NLA viewer](research/2026-10-19/deccp/README.md)
- [DECCP question, observations, selected prompts and setup](docs/research/deccp-pilot.md)
- [Exact five-case plan and editor instruction](config/deccp-refusal-v2.json)
- [Earlier harmful-benchmark responses, NLA texts and edits](research/2026-10-19/harmful/README.md)
- [HPC access and operating guide](docs/hpc.md)

The DECCP notes distinguish screening, early editor review and unresolved causal
claims. The harmful export is an ungraded, blind copy of preserved text, including
incomplete branches. Its publication here was explicitly requested; this does not
authorize publishing additional runs or merging these artifacts into main.

Based on main at ecca45c, with the active experiment's execution order and SSH
recovery changes. Main's newer API-preflight, parsing and steering utilities are
preserved. The live inference checkout is unchanged. The supervisor has a portable
interface here; deployment state, credentials and full checkpoints remain private.
