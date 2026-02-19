"""SystemMonitor — background daemon thread for CPU/mem/GPU metrics."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ._run import Run


class SystemMonitor:
    def __init__(self, run: Run, interval: float = 5.0):
        self._run = run
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._step = 0

        # probe capabilities
        self._has_psutil = False
        self._has_pynvml = False
        self._gpu_count = 0

        try:
            import psutil  # noqa: F401
            self._has_psutil = True
        except ImportError:
            pass

        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_count = pynvml.nvmlDeviceGetCount()
            self._has_pynvml = self._gpu_count > 0
        except Exception:
            pass

    def start(self) -> None:
        if not self._has_psutil and not self._has_pynvml:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                metrics = self._sample()
                if metrics:
                    ts = time.time()
                    step = self._step
                    self._step += 1
                    for key, value in metrics.items():
                        self._run._db.execute(
                            "INSERT INTO metrics (run_id, key, step, ts, value) VALUES ($1, $2, $3, $4, $5)",
                            [self._run.id, key, step, ts, float(value)],
                        )
            except Exception:
                pass
            self._stop_event.wait(self._interval)

    def _sample(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if self._has_psutil:
            try:
                import psutil
                metrics["system/cpu_percent"] = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                metrics["system/memory_percent"] = mem.percent
                metrics["system/memory_used_gb"] = mem.used / (1024 ** 3)
            except Exception:
                pass

        if self._has_pynvml:
            try:
                import pynvml
                for i in range(self._gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except Exception:
                        temp = 0
                    prefix = f"system/gpu{i}"
                    metrics[f"{prefix}_util_percent"] = float(util.gpu)
                    metrics[f"{prefix}_memory_used_gb"] = mem_info.used / (1024 ** 3)
                    metrics[f"{prefix}_temp_c"] = float(temp)
            except Exception:
                pass

        return metrics
