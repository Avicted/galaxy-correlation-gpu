#!/usr/bin/env bash
# Warn about anything on this machine that would skew a benchmark.
#
# Advisory only: it never fails the build, because a reader running the code to
# see it work should not be blocked by a CPU governor setting. It exists so
# that a number measured under the wrong conditions is not quoted as if it had
# been measured under the right ones.
set -euo pipefail

governor_file=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
governor=unknown
[[ -r $governor_file ]] && governor=$(<"$governor_file")

if [[ $governor != performance ]]; then
    cat <<MSG
WARNING: CPU governor is '$governor', not 'performance'.
         The published figures assume 'performance'. Expect slower and noisier
         results - the kernel time is sensitive to this. Set it with:
           sudo cpupower frequency-set -g performance

MSG
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    if [[ ${others:-0} -gt 0 ]]; then
        printf "NOTE: %s other process(es) currently hold GPU memory.\n\n" "$others"
    fi
fi
