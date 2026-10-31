import sys

import pytest

from datamate.local_inference import LocalModelError, LocalTransformersBackend


def test_local_backend_requires_checkpoint_and_has_no_remote_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    with pytest.raises(LocalModelError, match="local checkpoint"):
        LocalTransformersBackend(tmp_path / "missing")

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setitem(sys.modules, "torch", None)
    backend = LocalTransformersBackend(checkpoint)

    with pytest.raises(LocalModelError, match="no remote API fallback"):
        backend.generate("Tổng doanh thu?")
