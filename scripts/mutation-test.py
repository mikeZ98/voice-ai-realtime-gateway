#!/usr/bin/env python3
"""AST mutation testing harness for the gateway's critical paths.

For every target module, each mutation point found by the AST walker becomes one mutant:
a copy of ``src/`` in which exactly that one node is altered. The mapped test suite runs
against the mutant. A failing (or hanging) suite *kills* the mutant; a passing suite
means the mutant *survived*, i.e. the tests cannot tell the regression apart from correct
code.

Operators:
  * comparisons    ``<`` <-> ``<=``, ``>`` <-> ``>=``, ``==`` <-> ``!=``,
                   ``is`` <-> ``is not``, ``in`` <-> ``not in``
  * boolean        ``and`` <-> ``or``; ``not x`` -> ``x``
  * arithmetic     ``+`` <-> ``-``, ``*`` <-> ``//``, ``&`` <-> ``|``; same for ``+=`` / ``-=``
  * constants      integer literal ``n`` -> ``n + 1``; ``True`` <-> ``False``

Never mutated: type annotations, ``if TYPE_CHECKING:`` blocks, docstrings, ``__all__``
and ``__slots__``. String literals are left alone.

Isolation: each worker owns ``.cache/mutation/worker-<n>/src`` and imports it ahead of
the editable install through ``PYTHONPATH``; the harness first proves that the copy is
what gets imported and that the unmutated baseline passes. Tests run with
``--no-cov -x`` and a derandomised Hypothesis profile backed by a private database.

Usage::

    uv run python scripts/mutation-test.py                 # all default targets
    uv run python scripts/mutation-test.py --target ring --jobs 8 --min-score 95

The JSON report (score, every survivor with its location) is written to
``.cache/mutation/report.json``. The exit status is non-zero below ``--min-score``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, override

ROOT: Final = Path(__file__).resolve().parent.parent
CACHE: Final = ROOT / ".cache" / "mutation"
PACKAGE: Final = "src/voice_ai_realtime_gateway"

TARGETS: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "barge-in": (
        f"{PACKAGE}/domain/duplex.py",
        ("tests/domain", "tests/vad/test_vad_bridge.py", "tests/pipeline"),
    ),
    "chunker": (f"{PACKAGE}/inference/chunker.py", ("tests/inference/test_chunker.py",)),
    "frames": (f"{PACKAGE}/core/frames.py", ("tests/core/test_frames.py",)),
    "framing": (f"{PACKAGE}/transport/framing.py", ("tests/transport/test_framing.py",)),
    "ring": (f"{PACKAGE}/core/ring.py", ("tests/core",)),
    "supervisor": (
        f"{PACKAGE}/core/supervisor.py",
        ("tests/core/test_supervisor.py", "tests/core/test_cancel_properties.py"),
    ),
    "validation": (
        f"{PACKAGE}/_validation.py",
        ("tests/test_numeric_contracts.py", "tests/deploy/test_lifecycle.py"),
    ),
}

_COMPARE_SWAPS: Final[dict[type[ast.cmpop], type[ast.cmpop]]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}
_BINOP_SWAPS: Final[dict[type[ast.operator], type[ast.operator]]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
    ast.BitAnd: ast.BitOr,
    ast.BitOr: ast.BitAnd,
}
_SKIPPED_NAMES: Final = frozenset({"__all__", "__slots__"})


@dataclass(frozen=True, slots=True)
class Mutant:
    """One mutation point."""

    target: str
    index: int
    line: int
    column: int
    description: str


@dataclass(slots=True)
class Result:
    """Outcome of running the mapped suite against one mutant."""

    mutant: Mutant
    status: str  # "killed", "survived" or "timeout"
    seconds: float


class _Mutator(ast.NodeTransformer):
    """Enumerates mutation points; when ``apply`` matches one, mutates exactly that node."""

    def __init__(self, apply: int | None = None) -> None:
        """Enumerate only, or also mutate point ``apply``."""
        super().__init__()
        self.apply = apply
        self.points: list[tuple[int, int, str]] = []

    # -- traversal policy -------------------------------------------------------------

    def _point(self, node: ast.AST, description: str) -> bool:
        index = len(self.points)
        self.points.append(
            (getattr(node, "lineno", 0), getattr(node, "col_offset", 0), description)
        )
        return index == self.apply

    def _body(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        if (
            statements
            and isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)
            and isinstance(statements[0].value.value, str)
        ):
            return [statements[0], *(self.visit(stmt) for stmt in statements[1:])]
        return [self.visit(stmt) for stmt in statements]

    @override
    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = self._body(node.body)
        return node

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.args.defaults = [self.visit(default) for default in node.args.defaults]
        node.args.kw_defaults = [
            None if d is None else self.visit(d) for d in node.args.kw_defaults
        ]
        node.body = self._body(node.body)
        return node

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node.args.defaults = [self.visit(default) for default in node.args.defaults]
        node.args.kw_defaults = [
            None if d is None else self.visit(d) for d in node.args.kw_defaults
        ]
        node.body = self._body(node.body)
        return node

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.body = self._body(node.body)
        return node

    @override
    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        if node.value is not None:
            node.value = self.visit(node.value)
        return node

    @override
    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if not names & _SKIPPED_NAMES:
            node.value = self.visit(node.value)
        return node

    @override
    def visit_If(self, node: ast.If) -> ast.If:
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            return node
        self.generic_visit(node)
        return node

    @override
    def visit_TypeAlias(self, node: ast.TypeAlias) -> ast.TypeAlias:
        return node

    # -- operators ----------------------------------------------------------------------

    @override
    def visit_Compare(self, node: ast.Compare) -> ast.Compare:
        self.generic_visit(node)
        for position, operator in enumerate(node.ops):
            swap = _COMPARE_SWAPS.get(type(operator))
            if swap is not None and self._point(
                node, f"{type(operator).__name__} -> {swap.__name__}"
            ):
                node.ops[position] = swap()
        return node

    @override
    def visit_BoolOp(self, node: ast.BoolOp) -> ast.BoolOp:
        self.generic_visit(node)
        swap: type[ast.boolop] = ast.Or if isinstance(node.op, ast.And) else ast.And
        if self._point(node, f"{type(node.op).__name__} -> {swap.__name__}"):
            node.op = swap()
        return node

    @override
    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and self._point(node, "drop `not`"):
            return node.operand
        return node

    @override
    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        self.generic_visit(node)
        swap = _BINOP_SWAPS.get(type(node.op))
        if swap is not None and self._point(node, f"{type(node.op).__name__} -> {swap.__name__}"):
            node.op = swap()
        return node

    @override
    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AugAssign:
        self.generic_visit(node)
        swap = _BINOP_SWAPS.get(type(node.op))
        if swap is not None and self._point(node, f"{type(node.op).__name__}= -> {swap.__name__}="):
            node.op = swap()
        return node

    @override
    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        value = node.value
        if isinstance(value, bool):
            if self._point(node, f"{value} -> {not value}"):
                return ast.copy_location(ast.Constant(not value), node)
        elif isinstance(value, int) and self._point(node, f"{value} -> {value + 1}"):
            return ast.copy_location(ast.Constant(value + 1), node)
        return node


def discover(target: str) -> list[Mutant]:
    """Every mutation point of ``target``, in deterministic order."""
    mutator = _Mutator()
    mutator.visit(ast.parse((ROOT / target).read_text(), filename=target))
    return [
        Mutant(target=target, index=index, line=line, column=column, description=description)
        for index, (line, column, description) in enumerate(mutator.points)
    ]


def mutate(target: str, index: int) -> str:
    """Source of ``target`` with mutation point ``index`` applied."""
    tree = ast.parse((ROOT / target).read_text(), filename=target)
    mutated = _Mutator(apply=index).visit(tree)
    return ast.unparse(ast.fix_missing_locations(mutated))


class Worker:
    """Owns one isolated copy of ``src/`` and runs suites against it."""

    def __init__(self, number: int) -> None:
        """Create (or recreate) worker ``number``'s private copy of ``src/``."""
        self.root = CACHE / f"worker-{number}"
        self.src = self.root / "src"
        if self.root.exists():
            shutil.rmtree(self.root)
        shutil.copytree(ROOT / "src", self.src)
        self.env = {
            **os.environ,
            "PYTHONPATH": str(self.src),
            "PYTHONDONTWRITEBYTECODE": "1",
            "VOICE_GATEWAY_HYPOTHESIS_HOME": str(self.root / "hypothesis"),
            "VOICE_GATEWAY_HYPOTHESIS_PROFILE": "mutation",
        }

    def imported_from(self) -> str:
        """Where ``voice_ai_realtime_gateway`` is imported from under this worker's env."""
        probe = "import voice_ai_realtime_gateway as p; print(p.__file__)"
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            env=self.env,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    def install(self, target: str, source: str) -> None:
        """Write ``source`` as ``target`` inside this worker's copy."""
        (self.src / Path(target).relative_to("src")).write_text(source)

    def run(self, tests: tuple[str, ...], timeout: float) -> tuple[str, float]:
        """Run ``tests``; returns ``(status, seconds)``."""
        command = [
            sys.executable, "-m", "pytest", *tests, "-x", "-q", "--no-cov", "-p", "no:randomly",
            "-o", f"cache_dir={self.root / 'pytest'}",
        ]  # fmt: skip
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command, env=self.env, cwd=ROOT, capture_output=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return "timeout", time.monotonic() - started
        return ("survived" if completed.returncode == 0 else "killed"), time.monotonic() - started


def _prepare(name: str, jobs: int) -> tuple[list[Worker], float]:
    """Build workers, prove isolation and time the unmutated baseline."""
    _, tests = TARGETS[name]
    workers = [Worker(number) for number in range(jobs)]
    for worker in workers:
        location = worker.imported_from()
        if not location.startswith(str(worker.src)):
            msg = f"isolation failed: the package was imported from {location}"
            raise SystemExit(msg)
    status, baseline = workers[0].run(tests, timeout=600)
    if status != "survived":
        msg = f"baseline suite for {name} does not pass on unmutated code"
        raise SystemExit(msg)
    return workers, baseline


def run_target(name: str, jobs: int, limit: int | None) -> list[Result]:
    """Mutation-test one target with ``jobs`` parallel workers."""
    target, tests = TARGETS[name]
    mutants = discover(target)[:limit]
    original = (ROOT / target).read_text()
    workers, baseline = _prepare(name, jobs)
    timeout = baseline * 4 + 10
    print(f"[{name}] {len(mutants)} mutants, baseline {baseline:.1f}s, timeout {timeout:.0f}s")
    pending: queue.Queue[Mutant] = queue.Queue()
    for mutant in mutants:
        pending.put(mutant)
    results: list[Result] = []
    lock = threading.Lock()

    def drain(worker: Worker) -> None:
        while True:
            try:
                mutant = pending.get_nowait()
            except queue.Empty:
                worker.install(target, original)
                return
            worker.install(target, mutate(target, mutant.index))
            outcome, seconds = worker.run(tests, timeout)
            with lock:
                results.append(Result(mutant=mutant, status=outcome, seconds=seconds))
                done = len(results)
            if done % 25 == 0:
                print(f"[{name}] {done}/{len(mutants)}")

    threads = [threading.Thread(target=drain, args=(worker,)) for worker in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(results, key=lambda result: result.mutant.index)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--target", action="append", choices=sorted(TARGETS), help="repeatable")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--limit", type=int, default=None, help="first N mutants per target")
    parser.add_argument("--min-score", type=float, default=95.0)
    arguments = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}
    killed_total = mutants_total = 0
    for name in arguments.target or sorted(TARGETS):
        results = run_target(name, max(1, arguments.jobs), arguments.limit)
        killed = sum(result.status != "survived" for result in results)
        survivors = [asdict(result.mutant) for result in results if result.status == "survived"]
        score = 100.0 * killed / len(results) if results else 100.0
        report[name] = {
            "mutants": len(results),
            "killed": killed,
            "score": round(score, 2),
            "survivors": survivors,
        }
        killed_total += killed
        mutants_total += len(results)
        print(f"[{name}] killed {killed}/{len(results)} = {score:.2f}%")
        for survivor in survivors:
            print(
                f"    survived  {survivor['target']}:{survivor['line']}:{survivor['column']}"
                f"  {survivor['description']}"
            )
    overall = 100.0 * killed_total / mutants_total if mutants_total else 100.0
    report["overall"] = {
        "mutants": mutants_total,
        "killed": killed_total,
        "score": round(overall, 2),
    }
    (CACHE / "report.json").write_text(json.dumps(report, indent=2))
    print(f"overall mutation score: {overall:.2f}% ({killed_total}/{mutants_total})")
    for worker_dir in CACHE.glob("worker-*"):
        shutil.rmtree(worker_dir)
    return 0 if overall >= arguments.min_score else 1


if __name__ == "__main__":
    sys.exit(main())
