#!/bin/sh
# galaxy-pre-commit-container: installed by `make install-hooks`; do not edit.
#
# Runs the pre-commit hooks for the stage this file is named after (pre-commit
# or pre-push). With Docker available they run in the pinned pre-commit image;
# without it, with a local pre-commit, which is what the devcontainer has.
set -eu

stage=$(basename "$0")
args=
if [ "$stage" = pre-push ]; then
    args=--all-files
fi
top=$(git rev-parse --show-toplevel)

if command -v docker >/dev/null 2>&1; then
    exec make -C "$top" --no-print-directory pre-commit-staged STAGE="$stage" ARGS="$args"
elif command -v pre-commit >/dev/null 2>&1; then
    cd "$top"
    # shellcheck disable=SC2086 # $args is empty or one flag
    exec pre-commit run --hook-stage "$stage" $args
fi

echo "error: git hooks need docker or pre-commit on PATH" >&2
exit 1
