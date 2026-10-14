# Response logs

These are user-authorized, unread response snapshots from the four-trajectory
experiments. `status.json` identifies partial versus completed runs and records
branch token counts. `responses.json` is for human review only; agents must not
inspect its contents. Prompt inputs and detailed API checkpoints stay in ignored
runtime storage.

To copy a snapshot without printing model content:

```sh
uv run --locked publish_run.py --run runs/RUN_NAME --name RUN_NAME
```

Stage the named log directory explicitly and push to `main`. Do not run a content
diff or preview on the response file.
