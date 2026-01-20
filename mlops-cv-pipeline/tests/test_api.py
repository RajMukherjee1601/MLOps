from __future__ import annotations

import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.serving import app as app_module


class DummyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return stable logits
        return torch.zeros((x.size(0), 10))


def test_health():
    client = TestClient(app_module.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_smoke(monkeypatch):
    monkeypatch.setattr(app_module, "load_model", lambda: DummyModel())
    monkeypatch.setattr(app_module, "append_inference_log", lambda *args, **kwargs: None)

    client = TestClient(app_module.app)

    # Make a simple 32x32 RGB image
    img = Image.new("RGB", (32, 32), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    r = client.post(
        "/predict",
        files={"file": ("test.png", buf.getvalue(), "image/png")},
    )

    assert r.status_code == 200
    body = r.json()
    assert "pred_class" in body
    assert "top5" in body
    assert len(body["top5"]) == 5
