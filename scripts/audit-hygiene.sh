#!/usr/bin/env bash
# Repository hygiene gate.
#
#   1. No absolute machine paths (user homes, mounted volumes, temp dirs) in any
#      tracked or untracked-but-not-ignored text file.
#   2. No build artifacts or caches tracked (or pending to be tracked) by git.
#   3. No tool cache outside ./.cache/ and no uncommitted build artifact on disk
#      (proves the redirection actually works and the tree is pristine).
#   4. Every tool's cache redirection is still configured in pyproject.toml.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

failures=0
fail() {
  printf '  \xe2\x9c\x97 %s\n' "$1" >&2
  failures=$((failures + 1))
}

# ---------------------------------------------------------------- 1. absolute paths
# An absolute path is a machine-root directory preceded by start-of-line or a
# non-path character. The alternation is written so this file never matches itself.
readonly ABSOLUTE_PATH_RE='(^|[^A-Za-z0-9_.~/-])/(Users|home|root|Volumes|private/var|var/folders|opt/homebrew|mnt/[a-z])/|[A-Za-z]:[\\/]+(Users|Documents and Settings)[\\/]'
if matches="$(git grep --untracked -I -n -E -e "${ABSOLUTE_PATH_RE}" -- . ':!uv.lock' 2>/dev/null)"; then
  while IFS= read -r line; do
    fail "absolute machine path: ${line}"
  done <<<"${matches}"
fi

# ---------------------------------------------------------------- 2. tracked artifacts
readonly ARTIFACT_RE='(^|/)(__pycache__|build|dist|htmlcov|\.eggs|\.venv|venv|\.cache|\.pytest_cache|\.ruff_cache|\.mypy_cache|\.hypothesis|\.tox|\.nox)/|\.egg-info/|\.py[cod]$|\.so$|(^|/)\.coverage(\..+)?$|(^|/)coverage\.(xml|json)$|(^|/)\.DS_Store$'
while IFS= read -r path; do
  [[ -n "${path}" ]] && fail "build artifact or cache under version control: ${path}"
done < <(git ls-files --cached --others --exclude-standard | grep -E "${ARTIFACT_RE}" || true)

# ---------------------------------------------------------------- 3. cache leaks on disk
while IFS= read -r leak; do
  [[ -n "${leak}" ]] && fail "cache leaked outside ./.cache/: ${leak#./}"
done < <(
  find . \( -path ./.git -o -path ./.venv -o -path ./.cache \) -prune -o \
    \( -name .pytest_cache -o -name .ruff_cache -o -name .mypy_cache -o -name .hypothesis \
    -o -name __pycache__ -o -name htmlcov -o -name .benchmarks -o -name .coverage \
    -o -name '.coverage.*' -o -name coverage.xml \) -print
)

# ---------------------------------------------------------------- 3b. build artifacts on disk
while IFS= read -r artifact; do
  [[ -n "${artifact}" ]] && fail "uncommitted build artifact: ${artifact#./}"
done < <(
  find . \( -path ./.git -o -path ./.venv -o -path ./.cache \) -prune -o \
    \( -path ./dist -o -path ./build -o -name '*.egg-info' -o -name '*.py[co]' -o -name '*.so' \) -print
)

# ---------------------------------------------------------------- 4. redirection config
require_config() {
  grep -qF -- "$1" pyproject.toml || fail "pyproject.toml lost cache redirection: $1"
}
require_config 'cache-dir = ".cache/uv"'
require_config 'cache-dir = ".cache/ruff"'
require_config 'cache_dir = ".cache/mypy"'
require_config 'cache_dir = ".cache/pytest"'
require_config 'data_file = ".cache/coverage/.coverage"'
grep -qxF '.cache/' .gitignore || fail ".gitignore must ignore .cache/"

if ((failures > 0)); then
  printf 'audit-hygiene: FAILED (%d violation(s))\n' "${failures}" >&2
  if find . \( -path ./.git -o -path ./.venv -o -path ./.cache \) -prune -o -name __pycache__ -print | grep -q .; then
    printf '  hint: export PYTHONPYCACHEPREFIX="$(git rev-parse --show-toplevel)/.cache/pycache"\n' >&2
  fi
  exit 1
fi
printf 'audit-hygiene: OK (no absolute paths, no tracked artifacts, all caches in ./.cache/)\n'
