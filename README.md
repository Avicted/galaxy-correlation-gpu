# Galaxy two-point angular correlation - CUDA and HIP

[![CI](https://github.com/Avicted/galaxy-correlation-gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/Avicted/galaxy-correlation-gpu/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

A GPU solver for the angular two-point correlation function of two 100,000-object
galaxy catalogs, with two independently tuned backends: **CUDA** for NVIDIA
Blackwell (RTX 5080) and **HIP** for AMD RDNA 2 (RX 6900 XT).

For every pair of galaxies the angular separation is computed,

$$\theta_{12} = \arccos\left(\sin\delta_1\sin\delta_2 + \cos\delta_1\cos\delta_2\cos(\alpha_1-\alpha_2)\right)$$

binned into `360 × 4 = 1440` buckets of 0.25°, for three histogram families - real-real
(**DD**), real-random (**DR**) and random-random (**RR**) - which are then combined with
the Landy–Szalay estimator. That is 3 × 10¹⁰ pair evaluations per run.

Two write-ups cover the tuning in detail:

- [CUDA Deep Dive: 10B Galaxy Pairs on an RTX 5080 (23 ms Kernel)](https://victoranderssen.com/blog/galaxy-problem-cuda-rtx5080/) - the NVIDIA work, as a measured ablation ladder
- [The Galaxy Problem](https://victoranderssen.com/blog/galaxy-problem/) - the earlier AMD/RDNA 2 work

## Quick start

`make` is the entry point for everything.

```sh
make            # list targets and the current configuration
make cuda       # build for NVIDIA  (needs nvcc)
make hip        # build for AMD     (needs hipcc)
make verify     # run once, check the output against the committed reference
make bench      # the measurement protocol used in the blog post
```

Each backend needs only its own compiler. `make all` builds whichever are present.

### Overriding the target

Every tunable is overridable on the command line:

```sh
make cuda ARCH=sm_89              # Ada instead of Blackwell
make cuda TILE=256                # different block tile
make hip OFFLOAD_ARCH=gfx1100     # RDNA 3 instead of RDNA 2
make bench BACKEND=hip RUNS=15
```

## Results

### NVIDIA - RTX 5080 (Blackwell, `sm_120`)

| | |
|---|---|
| Kernel | **22.75 ms** (median of runs 2–9) |
| Wall clock | **0.032 s** |
| Toolchain | CUDA 13.3, driver 610.57.04 |
| Register use | 34 registers, 0 bytes spilled |

The tuning is worth **8.3×**: the same code with the block tiling, the fast `acos`, the
DD/RR symmetry and the loop unrolling removed one at a time measures 188.7 ms.

### AMD - RX 6900 XT (RDNA 2, `gfx1030`)

| | |
|---|---|
| Kernel | 116.33 ms |
| Wall clock | 0.154 s (best observed 0.152 s) |

**These AMD figures are historical records, not measurements taken under the protocol
below.** They come from the project's own README at successive commits, on different days
and driver versions, and they have not been re-measured - there is no AMD GPU in the
machine used for the CUDA work. Treat them as a record of where the project has been
rather than as a comparison against the NVIDIA row.

Nothing here says CUDA is faster than HIP, or NVIDIA faster than AMD. The two backends ran
on different hardware three years apart. The only like-for-like measurements in this
project are within a single backend on a single GPU.

## Measurement protocol

`make bench` runs the solver 9 times. Run 1 is cold and is reported separately; the figure
to quote is the median of runs 2–9. `make bench` also warns when the CPU governor is not
set to `performance` or when another process holds GPU memory, both of which measurably
skew results:

```sh
sudo cpupower frequency-set -g performance
```

## Correctness

The program asserts on every run that all three histograms sum to exactly
`N² = 10 000 000 000`. A variant failing that assert is a failed variant, not a fast one.

`make verify` goes further and checks the output against the committed
`results/omega.out` byte for byte.

### One caveat worth reading

`fast_acosf` is a polynomial minimax approximation with a measured maximum error of
6.77 × 10⁻⁵ rad (0.0039°) against `acosf`. That is far below the 0.25° bin width, but it
does **not** make the binning identical: a sample only has to fall within 0.0039° of a bin
edge to move. Over the real catalogs it shifts counts in 353 of the 360 populated bins, at
most 0.03% of any one bin, with a largest omega change of 0.002257.

The histogram *totals* are unaffected, so the `N²` asserts cannot detect this - a total is
invariant to redistribution between bins. For studying the shape of a correlation function
this is irrelevant. If you need exact bin agreement, use `acosf` instead.

## Optimizations

### CUDA (Blackwell)

- **Block-tiled pair loop.** Each thread owns one `i` and loops a shared-memory tile of
  `TILE` `j`-galaxies, instead of one thread per pair - large reuse, far fewer loads.
- **Precomputed per-galaxy `sin`/`cos`(δ)**, removing a `sincos` from the inner loop.
- **Fast polynomial `acos`** (see the caveat above).
- **DD/RR symmetry** with whole-block diagonal skipping: only `j >= i` is computed, and
  off-diagonal pairs are weighted by 2.
- **Shared-memory atomic histograms**, which are fast on Blackwell.
- **`__launch_bounds__`** and inner-loop unrolling; `ptxas` reports zero spills.
- **`mmap` input parsing** with a custom fast ASCII number parser, replacing `fscanf`.

Two optimizations that were tried and *did not* pay off are documented in the blog post
rather than silently dropped.

Note that the HIP build's histogram padding to 1536 bins is **not** carried over here: an
interleaved A/B against 1440 agrees to within 0.06 ms (0.3%) on Blackwell, because the
atomics scatter across bins by data rather than by thread index. The CUDA kernel runs
unpadded, which is 1152 bytes per block cheaper in shared memory.

### HIP (RDNA 2)

- **Native HIP**, not a `hipify` translation of the CUDA source.
- **`-mwavefrontsize64`** to fix the wave size at compile time, plus
  `-munsafe-fp-atomics`. RDNA 2 resolves same-address LDS atomics within a wave in
  hardware; no explicit ballot aggregation is done in the source, despite what an
  earlier version of this list and a since-corrected function name implied.
- **Padded histograms** 1440 → 1536 bins to avoid LDS bank conflicts (this one does matter
  on RDNA 2), and 32-bit LDS counters to halve the shared-memory footprint.
- **Block-size tuning.** Measured: 16×16 → 493 ms, 16×32 → 167 ms, 32×16 → 166 ms,
  **32×32 → 147 ms**. The winner is 32×32, i.e. 1024 threads, 16 wave64s per block.
- **DD/RR symmetry** and **`mmap` input parsing**, as above.

## Development

```sh
make install-hooks   # pre-commit install --install-hooks
make lint            # the full no-GPU gate
make pre-commit      # the same, inside the pinned CUDA-free image
```

There are two makefiles: `Makefile` needs a GPU toolchain, `Makefile.pre-commit` holds
every check the hooks and CI run and needs none. [CONTRIBUTING.md](CONTRIBUTING.md)
explains why that split is load-bearing, which hook runs at which stage, and how the
byte-exact files in `data/` and `results/` are protected.

CI measures nothing - GitHub has no GPU runners, so it only proves the code compiles for
six target/tile combinations with zero register spills and that the tree is clean. Every
figure in this README comes from a real run on the RTX 5080, under the protocol above.

## Layout

```
src/galaxy_cuda.cu     CUDA implementation (NVIDIA, sm_120)
src/galaxy_hip.cpp     HIP implementation  (AMD, gfx1030)
data/                  Input catalogs - see data/README.md for provenance
results/omega.out      Committed reference output for `make verify`
checksums/SHA256SUMS   Freezes the catalogs and the reference output
scripts/               Benchmark helpers used by the Makefile
Makefile               Build, run, benchmark, verify  (needs a GPU toolchain)
Makefile.pre-commit    Every check the hooks and CI run  (needs no CUDA)
Dockerfile             CUDA development image
Dockerfile.pre-commit  CUDA-free image the hooks run in
```

Each backend is **one self-contained translation unit**. The host-side catalog reader,
timing scaffolding and histogram assertions are duplicated between the two rather than
shared through a header - roughly 300 lines. That is deliberate: each file compiles with a
single command and can be read end to end, which is what the write-ups quote, and the two
share no device code at all. The duplication is between backends, not within one; nobody
reads both at once.

## Data

The input catalogs come from the Åbo Akademi University GPU Programming course and are not
my own work; see [`data/README.md`](data/README.md).

## License

GPL-3.0. See [LICENSE](LICENSE).
