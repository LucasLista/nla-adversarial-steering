# Shared Qwen build and per-job server configuration. Source from the repo root.
# Metadata-only startup diagnostics; never record commands, output or credentials.
set -E
nla_startup_phase=toolchain
nla_startup_health() {
    local code=$1 line=$2
    printf '{"phase":"%s","exit_code":%s,"source_line":%s}\n' "$nla_startup_phase" "$code" "$line" > ".runtime/startup-${LSB_JOBID}.json"
}
trap 'nla_startup_health "$?" "$LINENO"' ERR
module load gcc/12.4.0-binutils-2.42
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
module load cuda/12.4.1 cmake/3.31.9
export PATH="$HOME/.local/bin:$PATH"
export CUDAHOSTCXX="$CXX"
# Keep dependency writes off the shared home filesystem.
export UV_CACHE_DIR="$PWD/.runtime/uv-cache"
mkdir -p runs
cache=.runtime/build-cuda/CMakeCache.txt
if [[ -f "$cache" ]]; then
    cached_cxx=$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "$cache")
    if [[ "$cached_cxx" != "$CXX" ]]; then
        # Reconfigure the mixed-compiler cache; retain compiled objects.
        rm "$cache"
    fi
fi
{ "$CXX" --version; nvcc --version; uv --version; } > "runs/toolchain-${LSB_JOBID}.txt"
nvidia-smi > "runs/gpu-${LSB_JOBID}.txt"
env | sort | sed -n '/^CUDA_/p;/^LSB_/p' > "runs/scheduler-${LSB_JOBID}.txt"
export NLA_RESIDENT_MODELS=1
export NLA_CONTEXT_SIZE=16384
nla_startup_phase=dependency_sync
uv sync --locked
nla_startup_phase=backend_build
uv run --locked python -m nla.setup --cuda --cuda-arch 70 --skip-download --jobs 4
# Allocate a per-job port range to avoid unrelated services on shared nodes.
server_port=$((30000 + (LSB_JOBID % 5000) * 4))
backend_port=$((server_port + 1))
printf '%s\n' "$server_port" > "runs/port-${LSB_JOBID}.txt"
hostname -f > "runs/host-${LSB_JOBID}.txt"
umask 077
uv run --locked python -c 'import secrets; print(secrets.token_hex(32))' > "runs/token-${LSB_JOBID}.txt"

nla_startup_phase=ready
nla_startup_health 0 "$LINENO"
trap - ERR
