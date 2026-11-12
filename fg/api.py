"""Operator dashboard: grade a fruit, show why, show the sorting decision.

A small FastAPI app behind a login. "Next fruit" renders a fresh synthetic
fruit (or you can upload an image), and the page shows the grade, the
confidence, the heat map over the fruit, and the lane the sorter would send it
to. It is a demonstration of the model in use, not a production line controller.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from . import auth, cam, sort
from .explain import grad_cam
from .model import load as load_model
from .render import render_fruit
from .spec import GRADES

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="FreshGrade")

_model = None


def model():
    global _model
    if _model is None:
        path = ROOT / "models" / "grader.pt"
        if not path.exists():
            raise HTTPException(503, "Model not trained yet. Run `make eval` first.")
        _model = load_model(path)
    return _model


def _png_b64(arr_uint8: np.ndarray, scale: int = 4) -> str:
    im = Image.fromarray(arr_uint8).resize((64 * scale, 64 * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _overlay(img: np.ndarray, cam_map: np.ndarray) -> np.ndarray:
    """Blend a red heat map over the fruit image."""
    base = img.astype(np.float32)
    heat = np.stack([cam_map, cam_map * 0.1, cam_map * 0.1], -1) * 255
    w = (cam_map[..., None] * 0.55)
    return np.clip(base * (1 - w) + heat * w, 0, 255).astype(np.uint8)


def _grade_array(img: np.ndarray) -> dict:
    m = model()
    x = cam.preprocess(img)
    cmap, cls, probs = grad_cam(m, x)
    conf = float(probs[cls])
    dec = sort.decide(cls, conf)
    return {
        "grade": GRADES[cls],
        "confidence": round(conf, 3),
        "probs": {GRADES[i]: round(float(probs[i]), 3) for i in range(3)},
        "lane": dec["lane"], "action": dec["action"],
        "image": _png_b64(img),
        "overlay": _png_b64(_overlay(img, cmap)),
    }


def _auth(token: str | None):
    if not token or not auth.check_token(token):
        raise HTTPException(401, "Please sign in.")


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "web" / "index.html").read_text()


@app.post("/login")
def login(user: str = Form(...), password: str = Form(...)):
    tok = auth.login(user, password)
    if not tok:
        raise HTTPException(401, "Wrong username or password.")
    return {"token": tok}


@app.post("/sample")
def sample(token: str = Form(...)):
    _auth(token)
    rng = np.random.default_rng()
    domain = "line" if rng.random() < 0.4 else "train"
    s = render_fruit(rng, domain)
    out = _grade_array(s["image"])
    out["true_grade"] = GRADES[s["grade"]]
    out["camera"] = "line camera" if domain == "line" else "studio camera"
    return out


@app.post("/grade")
async def grade(token: str = Form(...), file: UploadFile = File(...)):
    _auth(token)
    raw = await file.read()
    im = Image.open(io.BytesIO(raw)).convert("RGB").resize((64, 64))
    return JSONResponse(_grade_array(np.array(im)))
