# Contributing

## The two-makefile contract

There are two makefiles, and the split is deliberate.

| | `Makefile` | `Makefile.pre-commit` |
|---|---|---|
| Owns | build, run, benchmark, verify | every check a hook or CI runs |
| Needs | `nvcc` or `hipcc`; a GPU for `run`/`bench`/`verify` | nothing but the lint tools |
| Entry point for | using the solver | proving the tree is clean |

`Makefile` presumes a GPU toolchain. Its `BACKEND` variable shells out to
`command -v nvcc` at parse time, on *every* invocation, and every build and run
rule needs `nvcc` or `hipcc`. None of that exists on a contributor's laptop or
on a GitHub-hosted runner - but the quality gate has to run there.

So the rule is:

> Every `.pre-commit-config.yaml` local hook invokes
> `make -f Makefile.pre-commit <target>` and nothing else. `Makefile.pre-commit`
> never requires CUDA, ROCm or a GPU. `Makefile` delegates its quality targets
> down to it, so each check has exactly one implementation and the hook, the CI
> step and the local command are the same command.

Two marked exceptions: `tidy` needs the CUDA/ROCm *headers* (still not a GPU)
and skips cleanly without them, and `verify` needs a real GPU and delegates back
up to `Makefile`. Neither gates an automatic hook stage.

If you add a check, add it to `Makefile.pre-commit` and to `lint`, then point a
hook at it. Do not inline a command into `.pre-commit-config.yaml`.

## Setup

```sh
make install-hooks      # pre-commit install --install-hooks
make lint               # the full no-GPU gate
make pre-commit         # the same, inside the pinned container
```

If you have no lint tools installed locally, `make pre-commit` is the path -
it builds `Dockerfile.pre-commit` and runs every hook inside it, at the pinned
versions. `make -f Makefile.pre-commit tools` reports what you have and flags
drift from the pins.

## Before publishing a number

CI cannot measure anything: GitHub has no GPU runners. It proves the code
compiles for six target/tile combinations, that ptxas reports zero register
spills, and that the tree is clean. That is all.

Any figure that goes into the README or a write-up must come from a real run:

```sh
make check-env          # warns about the CPU governor and other GPU tenants
make bench              # 9 runs; quote the median of runs 2-9
make verify             # byte-for-byte against results/omega.out
```

`make check-env` is advisory, not a gate - but a kernel time measured under the
`powersave` governor is not comparable with the published figures, which assume
`performance`.

## The byte-exact files

`data/*.txt` and `results/omega.out` are frozen. The catalogs are third-party
CRLF files, and their exact bytes are an input to `results/omega.out`, which
`make verify` diffs byte for byte. Three things protect them:

- `.gitattributes` marks them `-text`, so git never normalises the line endings;
- `.pre-commit-config.yaml` excludes them from every whitespace hook;
- `checksums/SHA256SUMS` plus the `data-integrity` check turns any drift into a
  loud failure instead of a silent one.

If you genuinely need to change one, regenerate `checksums/SHA256SUMS` in the
same commit and say why in the message.

## Style

`clang-format` is pinned to the version in `Makefile.pre-commit`; a different
major version reformats the tree and fights CI. `make format` applies it.

Each backend is one self-contained translation unit. The host-side catalog
reader is duplicated between `src/galaxy_cuda.cu` and `src/galaxy_hip.cpp`
rather than shared through a header - see the note at the top of each file. That
is a decision, not an oversight; please do not "fix" it without discussing it.
