#!/usr/bin/env bash
# Sole-committer invariant.
#
# Fails unless EVERY commit reachable from any local ref has author == committer ==
#   mikeZ98 <145975972+mikeZ98@users.noreply.github.com>
# and no commit message carries co-author / foreign sign-off trailers or automated
# tool signatures. Also refuses a git identity configuration that would violate the
# invariant on the next commit.
set -euo pipefail

readonly EXPECTED_NAME="mikeZ98"
readonly EXPECTED_EMAIL="145975972+mikeZ98@users.noreply.github.com"
readonly EXPECTED="${EXPECTED_NAME} <${EXPECTED_EMAIL}>"

cd "$(git rev-parse --show-toplevel)"

failures=0
fail() {
  printf '  \xe2\x9c\x97 %s\n' "$1" >&2
  failures=$((failures + 1))
}

# ---------------------------------------------------------------- next-commit identity
for var in GIT_AUTHOR_NAME GIT_COMMITTER_NAME; do
  value="${!var:-}"
  if [[ -n "${value}" && "${value}" != "${EXPECTED_NAME}" ]]; then
    fail "environment ${var}='${value}' overrides the sole committer"
  fi
done
for var in GIT_AUTHOR_EMAIL GIT_COMMITTER_EMAIL; do
  value="${!var:-}"
  if [[ -n "${value}" && "${value}" != "${EXPECTED_EMAIL}" ]]; then
    fail "environment ${var}='${value}' overrides the sole committer"
  fi
done
configured_name="$(git config --get user.name || true)"
configured_email="$(git config --get user.email || true)"
if [[ "${configured_name}" != "${EXPECTED_NAME}" || "${configured_email}" != "${EXPECTED_EMAIL}" ]]; then
  fail "git config identity is '${configured_name} <${configured_email}>', expected '${EXPECTED}'"
fi

# ---------------------------------------------------------------- history
total=0
if git rev-parse --verify --quiet HEAD >/dev/null; then
  # Raw %an/%ae/%cn/%ce: .mailmap must not be able to launder an identity.
  while IFS=$'\x1f' read -r sha author_name author_email committer_name committer_email; do
    total=$((total + 1))
    short="${sha:0:12}"
    if [[ "${author_name}" != "${EXPECTED_NAME}" || "${author_email}" != "${EXPECTED_EMAIL}" ]]; then
      fail "${short} author is '${author_name} <${author_email}>'"
    fi
    if [[ "${committer_name}" != "${EXPECTED_NAME}" || "${committer_email}" != "${EXPECTED_EMAIL}" ]]; then
      fail "${short} committer is '${committer_name} <${committer_email}>'"
    fi
    message="$(git log -1 --format=%B "${sha}")"
    if grep -qiE '^[[:space:]]*co-authored-by:' <<<"${message}"; then
      fail "${short} has a Co-authored-by trailer (secondary author)"
    fi
    while IFS= read -r signoff; do
      if [[ "${signoff}" != "Signed-off-by: ${EXPECTED}" ]]; then
        fail "${short} has a foreign sign-off: '${signoff}'"
      fi
    done < <(grep -iE '^[[:space:]]*signed-off-by:' <<<"${message}" || true)
    if grep -qE 'Generated with \[|noreply@anthropic\.com|\xf0\x9f\xa4\x96' <<<"${message}"; then
      fail "${short} carries an automated tool signature"
    fi
  done < <(git log --all --format='%H%x1f%an%x1f%ae%x1f%cn%x1f%ce')
fi

if ((failures > 0)); then
  printf 'audit-attribution: FAILED (%d violation(s) across %d commit(s))\n' "${failures}" "${total}" >&2
  exit 1
fi
printf 'audit-attribution: OK (%d commit(s), sole committer %s)\n' "${total}" "${EXPECTED}"
