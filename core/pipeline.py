"""
DeepReality — Parallel Pin Orchestrator
=======================================

The execution core of the PIN Architecture. Every pin is an
independent processing unit, and pins that do not depend on one
another execute concurrently. When a pin does consume another pin's
output (for example, the explainability pins visualise the decisions of
the detection models), the dependency graph places it automatically at
the correct point in the schedule.

Usage:
    pipeline = PinPipeline(max_workers=8)
    pipeline.add_pin(pin_a1)                                  # independent
    pipeline.add_pin(pin_d1, depends_on=["PIN-B1", "PIN-B2"])  # dependent
    run = pipeline.run(image_path, on_pin_complete=callback)

    run.results["PIN-A1"]   -> standard pin JSON envelope
    run.durations["PIN-A1"] -> pin runtime in seconds
    run.total_time          -> wall-clock duration of the whole run
    run.sequential_time     -> summed pin runtime (hypothetical serial run)

Context passed to dependent pins:
    {
        "PIN-B1": {...complete PIN-B1 result...},
        "_pins":  {"PIN-B1": <PinB1Clip instance>}   # for shared model state
    }

Concurrency model: a thread pool is used rather than processes.
PyTorch inference, NumPy/OpenCV kernels and file I/O all release the
GIL, so threads achieve genuine parallelism on this workload, and the
loaded models — several gigabytes in total — are shared from memory
without the cost of process duplication.
"""

import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field


@dataclass
class PipelineRun:
    """Outcome of one pipeline run over a single image."""
    results: dict = field(default_factory=dict)     # pin_id -> pin JSON envelope
    durations: dict = field(default_factory=dict)   # pin_id -> seconds
    total_time: float = 0.0                          # actual (parallel) duration
    sequential_time: float = 0.0                     # duration a serial run would have taken

    @property
    def speedup(self) -> float:
        """Speed-up factor achieved by concurrent execution."""
        if self.total_time <= 0:
            return 1.0
        return self.sequential_time / self.total_time


@dataclass
class _PinNode:
    pin: object                 # BasePin instance
    depends_on: list[str]       # upstream pin_id list


class PinPipeline:
    """
    Dependency-graph (DAG) driven parallel pin executor.

    - Every pin without dependencies starts immediately.
    - A pin starts the moment its own dependencies complete, without
      waiting for unrelated pins.
    - A failed upstream pin does not block its dependants: the failure
      is propagated through the context so the dependent pin can decide
      how to proceed. Partial evidence is more useful than none.
    """

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self._nodes: dict[str, _PinNode] = {}

    def add_pin(self, pin, depends_on: list[str] | None = None):
        """Register a pin. depends_on lists the upstream pin ids."""
        deps = list(depends_on) if depends_on else []
        self._nodes[pin.pin_id] = _PinNode(pin=pin, depends_on=deps)
        return self

    def _validate(self):
        """Reject missing dependencies and dependency cycles."""
        for pin_id, node in self._nodes.items():
            for dep in node.depends_on:
                if dep not in self._nodes:
                    raise ValueError(
                        f"Pin {pin_id} declares a dependency on '{dep}', "
                        f"but '{dep}' was never added to the pipeline."
                    )
        # Cycle detection by topological consumption
        resolved: set[str] = set()
        pending = set(self._nodes)
        while pending:
            ready = {
                pid for pid in pending
                if all(d in resolved for d in self._nodes[pid].depends_on)
            }
            if not ready:
                raise ValueError(f"Dependency cycle detected among: {pending}")
            resolved |= ready
            pending -= ready

    def run(self, file_path: str, on_pin_complete=None) -> PipelineRun:
        """
        Execute every pin for a single image, honouring the dependency
        graph and maximising concurrency.

        Args:
            file_path:        Image to analyse.
            on_pin_complete:  Optional callback invoked on the main
                              thread as each pin finishes, which makes
                              live progress reporting safe.
        """
        self._validate()

        run = PipelineRun()
        t0 = time.perf_counter()

        remaining = set(self._nodes)
        running = {}  # Future -> pin_id

        def make_job(pin_id: str):
            node = self._nodes[pin_id]

            def job():
                context = {dep: run.results[dep] for dep in node.depends_on}
                if node.depends_on:
                    context["_pins"] = {
                        dep: self._nodes[dep].pin for dep in node.depends_on
                    }
                start = time.perf_counter()
                result = node.pin.run(str(file_path), context=context)
                return result, time.perf_counter() - start

            return job

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while remaining or running:
                # Dispatch every pin whose dependencies are now satisfied
                ready = [
                    pid for pid in remaining
                    if all(dep in run.results
                           for dep in self._nodes[pid].depends_on)
                ]
                for pid in ready:
                    remaining.discard(pid)
                    running[executor.submit(make_job(pid))] = pid

                if not running:
                    break  # _validate() already rejects cycles; this is a safety net

                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    pid = running.pop(future)
                    result, duration = future.result()
                    run.results[pid] = result
                    run.durations[pid] = duration
                    if on_pin_complete:
                        on_pin_complete(pid, result, duration)

        run.total_time = time.perf_counter() - t0
        run.sequential_time = sum(run.durations.values())
        return run
