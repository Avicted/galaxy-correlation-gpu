# Input catalogs

Two catalogs of 100,000 objects each:

| File | Contents |
|---|---|
| `data_100k_arcmin.txt` | Real observed galaxies |
| `flat_100k_arcmin.txt` | A synthetic, uniformly distributed random catalog |

## Format

Plain ASCII. The first line is the object count; every following line is one
object as two whitespace-separated values:

```
100000
4646.98	3749.51
4644.35	3749.52
...
```

The two columns are right ascension $\alpha$ and declination $\delta$, both in
**arcminutes**. The solver converts them to radians on read
(`arcmin2rad = π / (60 · 180)`).

## Provenance

These catalogs are the input data for the Galaxy Problem assignment in the
GPU Programming course at Åbo Akademi University. They are **not** my own data
and I claim no authorship of them. They are included here so that the
benchmark figures reported in the accompanying blog posts can actually be
reproduced rather than merely described - `make verify` checks the solver's
output against `results/omega.out` byte for byte, which is only meaningful with
the same input.

If you are the rights holder and would prefer these not be redistributed, open
an issue and I will replace them with a generator that produces catalogs of
equivalent size and distribution.

## Using your own data

Nothing in the solver is specific to these files. Any two catalogs in the format
above, of equal length, will work:

```sh
make cuda
./bin/galaxy_cuda.out my_real.txt my_random.txt out.txt
```

Note that the correctness assertions check each histogram sums to exactly
$N^2$, so both catalogs must contain the same number of objects.
