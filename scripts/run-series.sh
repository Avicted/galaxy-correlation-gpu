#!/usr/bin/env bash
# Run the solver N times against the two catalogs and report each kernel time.
#
# Usage: run-series.sh <binary> <runs> <output-file> <data> <flat>
#
# Run 1 is cold - the GPU clocks have not ramped and the page cache may be
# empty - so it is reported separately and excluded from the median. That is
# the protocol the published figures use.
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 <binary> <runs> <output-file> <data> <flat>" >&2
    exit 2
fi

binary=$1 runs=$2 output=$3 data=$4 flat=$5

mkdir -p "$(dirname "$output")"

times=()
for ((run = 1; run <= runs; run++)); do
    printf '  ---- Run %d/%d ----\n' "$run" "$runs"
    line=$("$binary" "$data" "$flat" "$output" | tee /dev/stderr \
        | grep -oE 'Kernel execution time: [0-9.]+' || true)
    [[ -n $line ]] && times+=("${line##* }")
done

if [[ ${#times[@]} -lt 2 ]]; then
    exit 0
fi

printf '\n  cold run (excluded): %s ms\n' "${times[0]}"
median=$(printf '%s\n' "${times[@]:1}" | sort -g | awk '{a[NR]=$1}
    END {print (NR % 2) ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2}')
printf '  median of runs 2-%d: %s ms\n' "$runs" "$median"
