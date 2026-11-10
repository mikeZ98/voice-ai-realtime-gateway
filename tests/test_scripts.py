"""Shell entry points survive being launched as ``sh scripts/<name>.sh``.

CI (and the README) invoke every script through ``sh``. That is bash in POSIX mode on
macOS but dash on Ubuntu, where ``set -o pipefail``, arrays and ``[[ ]]`` are errors.
Each script therefore re-execs itself under real bash before its first bash-only line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.sh"))
GUARD = 'if [ -z "${BASH_VERSION:-}" ] || shopt -qo posix 2>/dev/null; then exec bash "$0" "$@"; fi'


def test_scripts_are_discovered() -> None:
    assert len(SCRIPTS) >= 5


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_script_re_execs_under_bash_before_any_bash_only_line(script: Path) -> None:
    code = [
        line.strip()
        for line in script.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert code, f"{script.name} has no code"
    assert code[0] == GUARD, f"{script.name}: first statement must be the bash re-exec guard"
