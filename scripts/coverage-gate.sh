#!/usr/bin/env bash
# 100.00 % line AND branch coverage gate.
#
# Runs the suite from a clean coverage database, then verifies the JSON report
# file-by-file: a single missing line, missing branch or partial branch anywhere fails
# the gate (no rounding: 99.995 % is a failure). `pragma: no cover` is forbidden in
# src/ so coverage cannot be gamed; the only exclusion is `if TYPE_CHECKING:`.
#
# Extra arguments are forwarded to pytest.

# Re-exec under real bash when launched via `sh` (dash on Ubuntu lacks pipefail).
if [ -z "${BASH_VERSION:-}" ] || shopt -qo posix 2>/dev/null; then exec bash "$0" "$@"; fi
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${root}/.cache/pycache}"
readonly COV_DIR=".cache/coverage"
readonly COV_JSON="${COV_DIR}/coverage.json"

mkdir -p "${COV_DIR}"
rm -f "${COV_DIR}/.coverage" "${COV_DIR}"/.coverage.* "${COV_JSON}"

if pragmas="$(git grep --untracked -n -E 'pragma:[[:space:]]*no[[:space:]]+(cover|branch)' -- src)"; then
  printf 'coverage-gate: FAILED (coverage exclusion pragmas in src/)\n%s\n' "${pragmas}" >&2
  exit 1
fi

pytest_status=0
uv run --locked pytest --cov-report=json -q "$@" || pytest_status=$?

if [[ ! -f "${COV_JSON}" ]]; then
  printf 'coverage-gate: FAILED (no coverage report produced; pytest exit %d)\n' "${pytest_status}" >&2
  exit 1
fi

gate_status=0
uv run --locked python -I - "${COV_JSON}" <<'PY' || gate_status=$?
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)

offenders = []
for path, data in sorted(report["files"].items()):
    summary = data["summary"]
    missing_lines = data.get("missing_lines", [])
    missing_branches = data.get("missing_branches", [])
    if missing_lines or missing_branches or summary.get("num_partial_branches", 0):
        offenders.append(
            f"  {path}: missing lines {missing_lines or '-'}; "
            f"missing branches {missing_branches or '-'}"
        )

totals = report["totals"]
statements, covered = totals["num_statements"], totals["covered_lines"]
branches, covered_branches = totals["num_branches"], totals["covered_branches"]
print(
    f"coverage-gate: lines {covered}/{statements}, branches {covered_branches}/{branches}, "
    f"excluded {totals['excluded_lines']}, total {totals['percent_covered']:.2f}%"
)
if statements == 0:
    print("coverage-gate: FAILED (nothing was measured)", file=sys.stderr)
    sys.exit(1)
if offenders or covered != statements or covered_branches != branches:
    print("coverage-gate: FAILED (below 100.00%)", file=sys.stderr)
    print("\n".join(offenders), file=sys.stderr)
    sys.exit(1)
print("coverage-gate: OK (100.00% line + branch)")
PY

if ((pytest_status != 0)); then
  printf 'coverage-gate: FAILED (pytest exit %d)\n' "${pytest_status}" >&2
  exit "${pytest_status}"
fi
exit "${gate_status}"
