"""Leak-free supervision of session-scoped asyncio tasks.

* Every spawned task is strongly referenced until it finishes (no GC'd pending tasks).
* Every outcome is retrieved in a done-callback: cancellations are absorbed, failures are
  routed to ``on_failure`` -- so "Task exception was never retrieved" cannot happen.
* :meth:`TaskSupervisor.cancel` is synchronous (O(n), no ``await``), enabling a
  sub-millisecond barge-in path; :meth:`TaskSupervisor.join` then reaps with a deadline
  and reports stragglers that ignored cancellation instead of hanging forever.
* Cancelling the coroutine that awaits :meth:`TaskSupervisor.join` propagates normally
  and never cancels or orphans the supervised tasks (``asyncio.wait`` semantics).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, final

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

__all__ = ["FailureHandler", "JoinReport", "SupervisorClosedError", "TaskSupervisor"]

type FailureHandler = Callable[[asyncio.Task[None], BaseException], None]


class SupervisorClosedError(RuntimeError):
    """The supervisor no longer accepts new tasks."""


@dataclass(frozen=True, slots=True, kw_only=True)
class JoinReport:
    """Outcome of :meth:`TaskSupervisor.join`."""

    finished: int
    stragglers: int


@final
class TaskSupervisor:
    """Owns a dynamic set of ``Task[None]`` objects for one session."""

    __slots__ = ("_closed", "_on_failure", "_tasks")

    def __init__(self, *, on_failure: FailureHandler) -> None:
        """Route every task failure (non-cancellation exception) to ``on_failure``.

        ``on_failure`` runs as a loop callback and must not raise.
        """
        self._on_failure = on_failure
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    def __len__(self) -> int:
        """Number of supervised tasks that have not finished yet."""
        return len(self._tasks)

    @property
    def closed(self) -> bool:
        """Whether :meth:`spawn` is refused."""
        return self._closed

    def spawn(self, coro: Coroutine[Any, Any, None], *, name: str) -> asyncio.Task[None]:
        """Schedule ``coro`` as a supervised task on the running loop.

        Raises:
            SupervisorClosedError: after :meth:`close`; ``coro`` is closed so it never
                triggers a "coroutine was never awaited" warning.
        """
        if self._closed:
            coro.close()
            msg = f"cannot spawn {name!r}: supervisor is closed"
            raise SupervisorClosedError(msg)
        task = asyncio.get_running_loop().create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._reap)
        return task

    def cancel(self) -> int:
        """Request cancellation of every pending task; returns how many were signalled."""
        return sum(task.cancel() for task in self._tasks)

    def close(self) -> None:
        """Refuse further spawns (pending tasks are left to :meth:`cancel`/:meth:`join`)."""
        self._closed = True

    async def join(self, timeout: float) -> JoinReport:  # noqa: ASYNC109 -- deadline is the API
        """Wait up to ``timeout`` seconds for every currently supervised task to finish."""
        pending = set(self._tasks)
        if not pending:
            return JoinReport(finished=0, stragglers=0)
        done, still_pending = await asyncio.wait(pending, timeout=timeout)
        return JoinReport(finished=len(done), stragglers=len(still_pending))

    def _reap(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._on_failure(task, error)
