#!/usr/bin/env bash
# Mutation testing of the audio framing and dispatch core.
#
# Front end for scripts/mutation-test.py, the project's AST mutation harness. Every
# mutant is built in an isolated worker copy under ./.cache/mutation/worker-<n>/src and
# imported ahead of the editable install, so tracked files are only ever read. Mutant
# runs use --no-cov (no coverage data-file contention) and a derandomised Hypothesis
# profile with a private example database.
#
# Usage: sh scripts/mutate.sh [--min-score PCT] [--jobs N] [--target NAME ...]
#   --min-score PCT   fail below PCT percent killed (default 95)
#   --jobs N          parallel workers (default: CPU count)
#   --target NAME     restrict to one target (repeatable; default: all of them)
#
# Targets: barge-in (domain/duplex.py), chunker, frames (core/frames.py), framing
# (transport/framing.py), ring (core/ring.py), supervisor (core/supervisor.py),
# validation (_validation.py). The JSON report with every survivor is written to
# ./.cache/mutation/report.json.

# Re-exec under real bash when launched via `sh` (POSIX mode lacks arrays/[[ ]]).
if [ -z "${BASH_VERSION:-}" ] || shopt -qo posix 2>/dev/null; then exec bash "$0" "$@"; fi
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"

harness_args=()
while (($#)); do
  case "$1" in
    --min-score | --jobs | --target)
      harness_args+=("$1" "${2:?$1 needs a value}")
      shift 2
      ;;
    -h | --help) sed -n '2,18p' "$0"; exit 0 ;;
    *) printf 'mutate.sh: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ " ${harness_args[*]-} " == *" --min-score "* ]] || harness_args+=(--min-score 95)

git check-ignore -q --no-index .cache/mutation/probe || {
  printf 'mutate.sh: .cache/mutation/ is not git-ignored\n' >&2
  exit 1
}
for stray in mutants .cache/mutation/worker-0/src/.git; do
  [[ -e "${stray}" ]] && { printf 'mutate.sh: unexpected %s\n' "${stray}" >&2; exit 1; }
done

# Tracked sources must be byte-identical before and after: mutants live only in the
# worker copies.
before="$(git ls-files -s -- src tests | git hash-object --stdin)$(git diff --stat -- src tests | git hash-object --stdin)"

status=0
uv run --frozen python scripts/mutation-test.py "${harness_args[@]}" || status=$?

after="$(git ls-files -s -- src tests | git hash-object --stdin)$(git diff --stat -- src tests | git hash-object --stdin)"
if [[ "${before}" != "${after}" ]]; then
  printf 'mutate.sh: tracked sources changed during the run\n' >&2
  exit 1
fi

if ((status != 0)); then
  printf '\nmutate: FAILED (see .cache/mutation/report.json)\n' >&2
  exit "${status}"
fi
printf '\nmutate: PASSED\n'
