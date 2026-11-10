#!/usr/bin/env bash
# Repository hygiene gate.
#
#   1. No absolute machine paths (user homes, mounted volumes, temp dirs) in any
#      tracked or untracked-but-not-ignored text file.
#   2. No build artifacts or caches tracked (or pending to be tracked) by git.
#   3. No tool cache outside ./.cache/ and no uncommitted build artifact on disk
#      (proves the redirection actually works and the tree is pristine).
#   4. Every tool's cache redirection is still configured in pyproject.toml.
#   5. Every image pulled by deploy/Dockerfile and compose.yaml is pinned by digest.
#   6. With --workspace: a pristine working tree (no modified, staged or untracked
#      non-ignored file), proving tooling such as scripts/mutate.sh touched nothing
#      outside ./.cache/.
#
# Usage: sh scripts/audit_hygiene.sh [--workspace]

# Re-exec under real bash when launched via `sh` (POSIX mode lacks process substitution).
if [ -z "${BASH_VERSION:-}" ] || shopt -qo posix 2>/dev/null; then exec bash "$0" "$@"; fi
set -euo pipefail

workspace=0
for arg in "$@"; do
  case "${arg}" in
    --workspace) workspace=1 ;;
    -h | --help) sed -n '2,17p' "$0"; exit 0 ;;
    *) printf 'audit_hygiene.sh: unknown argument: %s\n' "${arg}" >&2; exit 2 ;;
  esac
done

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
    -o -name '.coverage.*' -o -name coverage.xml -o -name mutants \) -print
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

# ---------------------------------------------------------------- 5. images pinned by digest
# Dockerfile: the syntax frontend, image ARGs and literal FROMs (FROM ${ARG} is covered
# by its ARG). compose.yaml: every pulled image; the locally built
# voice-ai-realtime-gateway/* image has no registry digest.
readonly DIGEST_RE='@sha256:[0-9a-f]{64}'
if [[ -f deploy/Dockerfile ]]; then
  if unpinned="$(grep -nE '^# syntax=|^ARG [A-Z_]+_IMAGE=|^FROM ' deploy/Dockerfile \
      | grep -vE '^[0-9]+:FROM \$\{[A-Z_]+\}' | grep -vE "${DIGEST_RE}")"; then
    fail "deploy/Dockerfile image references without a digest:"$'\n'"${unpinned}"
  fi
fi
if [[ -f compose.yaml ]]; then
  if unpinned="$(grep -nE '^[[:space:]]+image:' compose.yaml \
      | grep -v 'image: voice-ai-realtime-gateway/' | grep -vE "${DIGEST_RE}")"; then
    fail "compose.yaml images without a digest:"$'\n'"${unpinned}"
  fi
fi

# ---------------------------------------------------------------- 6. pristine tree
if ((workspace)); then
  if dirty="$(git status --porcelain --untracked-files=all)" && [[ -n "${dirty}" ]]; then
    fail "working tree is not pristine:"$'\n'"${dirty}"
  fi
fi

if ((failures > 0)); then
  printf 'audit_hygiene: FAILED (%d violation(s))\n' "${failures}" >&2
  if find . \( -path ./.git -o -path ./.venv -o -path ./.cache \) -prune -o -name __pycache__ -print | grep -q .; then
    printf '  hint: export PYTHONPYCACHEPREFIX="$(git rev-parse --show-toplevel)/.cache/pycache"\n' >&2
  fi
  exit 1
fi
printf 'audit_hygiene: OK (no absolute paths, no tracked artifacts, all caches in ./.cache/%s)\n' \
  "$( ((workspace)) && printf ', pristine tree' )"
