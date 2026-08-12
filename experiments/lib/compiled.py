"""Persistent compiled-inference server.

torch.compile(mode="reduce-overhead") keeps its cudagraph state in thread-local
storage, so the compiled forward must always run on the thread that captured it —
but engine service threads are per-collect. This server owns one long-lived thread;
the engine callback forwards each request to it over a queue. Calls are padded to a
fixed row count so capture sees a single shape (cache hits, terminal simulations and
deduplication shrink calls unpredictably, and every distinct row count is a fresh
capture).
"""

import queue
import threading


class CompiledInferServer:
    def __init__(self, module, shape, pad_rows: int, device: str) -> None:
        self._module = module
        self._shape = shape
        self._pad_rows = pad_rows
        self._device = device
        self._requests: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def infer(self, obs):
        done = threading.Event()
        box: dict = {}
        self._requests.put((obs, box, done))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box["result"]

    def close(self) -> None:
        self._requests.put(None)
        self._thread.join()

    def _serve(self) -> None:
        import numpy as np
        import torch

        compiled = torch.compile(self._module, mode="reduce-overhead")
        c, h, w = self._shape
        while True:
            item = self._requests.get()
            if item is None:
                return
            obs, box, done = item
            try:
                n = obs.shape[0]
                if n < self._pad_rows:
                    padded = np.zeros((self._pad_rows, obs.shape[1]), dtype=obs.dtype)
                    padded[:n] = obs
                    obs = padded
                with torch.inference_mode():
                    x = (
                        torch.from_numpy(np.ascontiguousarray(obs))
                        .reshape(-1, c, h, w)
                        .to(self._device)
                    )
                    logits, values = compiled(x)
                    # row slice only — the action axis stays whole (f32 contract)
                    box["result"] = (logits[:n].cpu().numpy(), values[:n].cpu().numpy())
            except BaseException as e:  # noqa: BLE001 — must reach the calling thread
                box["error"] = e
            finally:
                done.set()
