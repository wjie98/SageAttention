# Turing SM75 Build Notes

This branch can compile SM75 and SM86 extensions on a low-memory WSL setup,
but the CUDA compiler has high peak memory usage. On a 4 GiB WSL instance,
parallel builds can exhaust RAM and swap while compiling the SM75 attention
kernels.

Use conservative build parallelism on low-memory machines:

```bash
source /home/wenjie/.miniconda3/etc/profile.d/conda.sh
conda activate turing-attn-cu128

export CUDA_HOME=/home/wenjie/.local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export TORCH_CUDA_ARCH_LIST="7.5;8.6"
export EXT_PARALLEL=1
export MAX_JOBS=1
export NVCC_APPEND_FLAGS="--threads=1"

pip install -e . --no-build-isolation
```

Observed memory behavior on the local WSL test machine:

- `MAX_JOBS=2` started multiple `cudafe++`/`cc1plus` processes at once and
  pushed the system into heavy swap.
- Single-file CUDA compiler stages such as `cicc` and `ptxas` can still use
  roughly 1.3-1.7 GiB RSS for large SM75 kernels.
- `EXT_PARALLEL=1` and `MAX_JOBS=1` keep the build slow but stable.

For quick SM75-only checks, build only Turing:

```bash
export TORCH_CUDA_ARCH_LIST="7.5"
export EXT_PARALLEL=1
export MAX_JOBS=1
export NVCC_APPEND_FLAGS="--threads=1"
pip install -e . --no-build-isolation
```

Only raise `MAX_JOBS` or `EXT_PARALLEL` on machines with enough RAM. For WSL,
prefer increasing the WSL memory limit before using parallel CUDA builds.
