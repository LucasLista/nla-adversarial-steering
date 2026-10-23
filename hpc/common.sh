# Shared setup for DTU LSF jobs; source from the repository root.
module load gcc/12.4.0-binutils-2.42 cuda/12.4.1 cmake/3.31.9
export CC="$(command -v gcc)" CXX="$(command -v g++)" CUDAHOSTCXX="$(command -v g++)"
# The CUDA module can be missing on a host that still mounts the toolkit.
export LD_LIBRARY_PATH="/appl/cuda/12.4.1/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export NLA_CUDA_LINK_LIB=/appl/cuda/12.4.1/targets/x86_64-linux/lib
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$PWD/.runtime/uv-cache"
mkdir -p runs .runtime
# Concurrent jobs share one venv and build tree.
(flock 9; uv sync --locked && uv run --locked python -m nla.setup --cuda --cuda-arch 70 --skip-download --jobs 4) 9>.runtime/sync.lock
# Each job gets its own block of four ports and a fresh bearer token.
port=$((30000 + (LSB_JOBID % 5000) * 4))
printf '%s\n' "$port" > "runs/port-$LSB_JOBID.txt"
hostname -f > "runs/host-$LSB_JOBID.txt"
(umask 077; uv run --locked python -c 'import secrets; print(secrets.token_hex(32))' > "runs/token-$LSB_JOBID.txt")
server=(uv run --locked python -m nla.server --resident --gpu-layers 99 --threads 4 --context 16384
        --port "$port" --backend-port $((port + 1)) --token-file "runs/token-$LSB_JOBID.txt")
