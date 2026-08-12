"""CompiledInferServer: single owner thread, fixed-shape padding, error propagation."""

import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from compiled import CompiledInferServer


class RecordingHeads:
    """Stands in for net.heads; records the thread each forward runs on."""

    def __init__(self) -> None:
        self.threads: set[int] = set()

    def __call__(self, x):
        self.threads.add(threading.get_ident())
        return x.sum(dim=(2, 3)), x.mean(dim=(1, 2, 3))


@pytest.fixture()
def eager(monkeypatch):
    # cudagraph capture needs CUDA; these tests cover threading and padding
    monkeypatch.setattr(torch, "compile", lambda m, **k: m)


def test_pads_to_fixed_shape_and_slices_back(eager) -> None:
    heads = RecordingHeads()
    server = CompiledInferServer(heads, (2, 3, 3), pad_rows=8, device="cpu")
    obs = np.random.default_rng(0).random((3, 18)).astype(np.float32)
    logits, values = server.infer(obs)
    server.close()
    assert logits.shape == (3, 2) and values.shape == (3,)
    want_l, want_v = heads(torch.from_numpy(obs).reshape(-1, 2, 3, 3))
    np.testing.assert_allclose(logits, want_l.numpy(), rtol=1e-6)
    np.testing.assert_allclose(values, want_v.numpy(), rtol=1e-6)


def test_oversize_calls_pass_through_unpadded(eager) -> None:
    server = CompiledInferServer(RecordingHeads(), (1, 2, 2), pad_rows=2, device="cpu")
    logits, values = server.infer(np.ones((5, 4), dtype=np.float32))
    server.close()
    assert logits.shape == (5, 1) and values.shape == (5,)


def test_all_forwards_run_on_one_persistent_thread(eager) -> None:
    heads = RecordingHeads()
    server = CompiledInferServer(heads, (1, 2, 2), pad_rows=4, device="cpu")
    callers = [
        threading.Thread(target=lambda: server.infer(np.ones((4, 4), dtype=np.float32)))
        for _ in range(4)
    ]
    for t in callers:
        t.start()
    for t in callers:
        t.join()
    server.close()
    assert len(heads.threads) == 1
    assert threading.get_ident() not in heads.threads


def test_callback_errors_propagate_to_the_caller(eager) -> None:
    def broken(x):
        raise ValueError("boom")

    server = CompiledInferServer(broken, (1, 2, 2), pad_rows=4, device="cpu")
    with pytest.raises(ValueError, match="boom"):
        server.infer(np.ones((2, 4), dtype=np.float32))
    # the serve loop must survive a failed request
    with pytest.raises(ValueError, match="boom"):
        server.infer(np.ones((2, 4), dtype=np.float32))
    server.close()
