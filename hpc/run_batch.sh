# Shared batch driver; batch_args are set by the LSF job.
uv run --locked python -m nla.launch --port "$server_port" --backend-port "$backend_port" --gpu-layers 99 --threads 4 --auth-token-file "runs/token-${LSB_JOBID}.txt" --audit-log &
server=$!
trap 'kill "$server" 2>/dev/null; wait "$server" 2>/dev/null' EXIT
# Self-contained driver: no laptop connection needed. Retries resume from checkpoints.
for attempt in 1 2 3 4 5; do
    if uv run --locked python -m hpc.hpc_batch "${batch_args[@]}" --port "$server_port" \
        --token-file "runs/token-${LSB_JOBID}.txt" \
        --benchmark-manifest "${NLA_BENCHMARK_MANIFEST:-runs/benchmarks/jbb-benign/manifest.json}" \
        --max-tokens "${NLA_MAX_TOKENS:-128}" --fast; then
        printf 'complete\n' > "runs/batch-status-${LSB_JOBID}.txt"; exit 0
    else
        batch_exit=$?
        if [[ "$batch_exit" == 78 ]]; then
            printf 'editor_requires_attention\n' > "runs/batch-status-${LSB_JOBID}.txt"
            exit 78
        fi
    fi
    printf 'attempt %s failed\n' "$attempt" >> "runs/batch-status-${LSB_JOBID}.txt"
    kill -0 "$server" 2>/dev/null || exit 1
    sleep 60
done
exit 1
