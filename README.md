# Galaxy two-point angular correlation - CUDA and HIP

[![CI](https://github.com/Avicted/galaxy-correlation-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/Avicted/galaxy-correlation-gpu/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

A GPU solver for the angular two-point correlation function of two 100,000-object galaxy
catalogs, with independently tuned CUDA (NVIDIA Blackwell) and HIP (AMD RDNA 2) backends.

For every pair of galaxies the angular separation is computed,

$$\theta_{12} = \arccos\left(\sin\delta_1\sin\delta_2 + \cos\delta_1\cos\delta_2\cos(\alpha_1-\alpha_2)\right)$$

binned into `360 × 4 = 1440` buckets of 0.25°, for three histogram families - real-real
(**DD**), real-random (**DR**) and random-random (**RR**) - which are combined with the
Landy–Szalay estimator. 3 × 10¹⁰ pair evaluations per run.

Write-ups:

- [CUDA Deep Dive: 10B Galaxy Pairs on an RTX 5080 (23 ms Kernel)](https://victoranderssen.com/blog/galaxy-problem-cuda-rtx5080/) - the NVIDIA work, as a measured ablation ladder
- [The Galaxy Problem](https://victoranderssen.com/blog/galaxy-problem/) - the earlier AMD/RDNA 2 work

## Quick start

```sh
make            # list targets and the current configuration
make cuda       # build for NVIDIA (needs nvcc)
make hip        # build for AMD (needs hipcc)
make verify     # run once, check the output against the committed reference
make bench      # 9 runs; quote the median of runs 2-9
```

`make all` builds whichever backends have a compiler present. Every tunable overrides on
the command line:

```sh
make cuda ARCH=sm_89              # Ada instead of Blackwell
make cuda TILE=256                # different block tile
make hip OFFLOAD_ARCH=gfx1100     # RDNA 3 instead of RDNA 2
make bench BACKEND=hip RUNS=15
```

## Results

RTX 5080 (Blackwell, `sm_120`), CUDA 13.3, driver 610.57.04:

| | |
|---|---|
| Kernel | **22.75 ms** (median of runs 2-9) |
| Wall clock | **0.032 s** |
| Registers | 34, 0 bytes spilled |

Removing the block tiling, the fast `acos`, the DD/RR symmetry and the loop unrolling one
at a time measures 188.7 ms, so the tuning is worth 8.3×.

RX 6900 XT (RDNA 2, `gfx1030`): 116.33 ms kernel, 0.154 s wall clock.

The AMD figures are historical records carried forward from earlier commits, not
measurements taken under the protocol above; there is no AMD GPU in the machine used for
the CUDA work. The two backends ran on different hardware three years apart, so the rows do
not compare.

## Correctness

Every run asserts that all three histograms sum to exactly `N² = 10 000 000 000`.
`make verify` also diffs the output against `results/omega.out` byte for byte.

`fast_acosf` is a minimax polynomial with a measured maximum error of 6.77 × 10⁻⁵ rad
(0.0039°) against `acosf`. That is far below the 0.25° bin width but it does not make the
binning identical: a sample within 0.0039° of a bin edge can move. Over the real catalogs
it shifts counts in 353 of the 360 populated bins, at most 0.03% of any one bin, with a
largest omega change of 0.002257. The totals are invariant to that redistribution, so the
`N²` asserts cannot detect it. Use `acosf` if you need exact bin agreement.

## Optimizations

CUDA (Blackwell):

- Block-tiled pair loop: each thread owns one `i` and walks a shared-memory tile of `TILE`
  `j`-galaxies, instead of one thread per pair.
- Precomputed per-galaxy `sin`/`cos`(δ), removing a `sincos` from the inner loop.
- Fast polynomial `acos` (see the caveat above).
- DD/RR symmetry with whole-block diagonal skipping: only `j >= i` is computed,
  off-diagonal pairs weighted by 2.
- Shared-memory atomic histograms, which are fast on Blackwell.
- `__launch_bounds__` and inner-loop unrolling; `ptxas` reports zero spills.
- `mmap` input parsing with a custom fast ASCII number parser, replacing `fscanf`.
- No histogram padding, unlike the HIP build: an interleaved A/B of 1440 against 1536 bins
  agrees to within 0.06 ms (0.3%) on Blackwell, and 1440 is 1152 bytes per block cheaper in
  shared memory.

HIP (RDNA 2):

- Native HIP, not a `hipify` translation of the CUDA source.
- `-mwavefrontsize64` to fix the wave size at compile time, plus `-munsafe-fp-atomics`.
  RDNA 2 resolves same-address LDS atomics within a wave in hardware.
- Histograms padded 1440 to 1536 bins to avoid LDS bank conflicts, and 32-bit LDS counters
  to halve the shared-memory footprint.
- Block size 32×32, i.e. 1024 threads, 16 wave64s. Measured: 16×16 493 ms, 16×32 167 ms,
  32×16 166 ms, 32×32 147 ms.
- DD/RR symmetry and `mmap` input parsing, as above.

## Layout

```
src/galaxy_cuda.cu     CUDA implementation (NVIDIA, sm_120)
src/galaxy_hip.cpp     HIP implementation  (AMD, gfx1030)
data/                  Input catalogs - see data/README.md for provenance
results/omega.out      Committed reference output for `make verify`
checksums/SHA256SUMS   Freezes the catalogs and the reference output
scripts/               Benchmark helpers used by the Makefile
Makefile               Build, run, benchmark, verify (needs a GPU toolchain)
Makefile.pre-commit    Every check the hooks and CI run (needs no CUDA)
Dockerfile             CUDA development image
Dockerfile.pre-commit  CUDA-free image the hooks run in
```

Each backend is one self-contained translation unit; the host-side catalog reader and
timing scaffolding are deliberately duplicated rather than shared through a header.
[CONTRIBUTING.md](CONTRIBUTING.md) covers that, the two-makefile split, the benchmark
protocol and the frozen files in `data/` and `results/`.

## Data

The input catalogs come from the Åbo Akademi University GPU Programming course and are not
my own work; see [`data/README.md`](data/README.md).

## License

GPL-3.0. See [LICENSE](LICENSE).
