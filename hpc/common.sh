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
# The CUDA module may be unavailable on a host even while the pinned toolkit is
# mounted. Make its shared libraries explicit for linking and backend loading.
export LD_LIBRARY_PATH="/appl/cuda/12.4.1/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export NLA_CUDA_LINK_LIB=/appl/cuda/12.4.1/targets/x86_64-linux/lib
export PATH="$HOME/.local/bin:$PATH"
export CUDAHOSTCXX="$CXX"
# Keep dependency writes off the shared home filesystem.
export UV_CACHE_DIR="$PWD/.runtime/uv-cache"
mkdir -p runs
# Compiler-cache validation is performed by nla.setup under its build lock.
{ "$CXX" --version; nvcc --version; uv --version; } > "runs/toolchain-${LSB_JOBID}.txt"
nvidia-smi > "runs/gpu-${LSB_JOBID}.txt"
env | sort | sed -n '/^CUDA_/p;/^LSB_/p' > "runs/scheduler-${LSB_JOBID}.txt"
export NLA_RESIDENT_MODELS=1
export NLA_CONTEXT_SIZE=16384
# Parallel jobs share one venv and build tree, so concurrent syncs and cmake
# runs corrupt each other. Serialize both; later jobs find the build cached.
mkdir -p .runtime
exec {nla_build_lock}>.runtime/build.lock
nla_startup_phase=build_lock
flock --timeout 5400 "$nla_build_lock"
nla_startup_phase=dependency_sync
uv sync --locked
nla_startup_phase=backend_build
uv run --locked python -m nla.setup --cuda --cuda-arch 70 --skip-download --jobs 4
flock --unlock "$nla_build_lock"
exec {nla_build_lock}>&-
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
