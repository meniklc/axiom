"""Routes for the Evaluation tab. Mounted with one include_router() line in app.py."""
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.evaluation import store
from src.evaluation.service import SAMPLE_QUESTIONS, evaluate_query

ROOT = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(ROOT / "templates"))
router = APIRouter()


class EvalRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)


@router.get("/evaluation", response_class=HTMLResponse)
def evaluation_page(request: Request):
    return templates.TemplateResponse(request=request, name="evaluation.html", context={})


@router.post("/api/eval/run")
async def eval_run(payload: EvalRequest):
    try:
        return await run_in_threadpool(evaluate_query, payload.question)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/eval/samples")
def eval_samples():
    return SAMPLE_QUESTIONS


@router.get("/api/eval/summary")
def eval_summary():
    return store.summary()


@router.get("/api/eval/history")
def eval_history(limit: int = 50):
    return store.list_runs(min(max(limit, 1), 500))


@router.get("/api/eval/runs/{run_id}")
def eval_run_detail(run_id: int):
    r = store.get_run(run_id)
    if not r:
        raise HTTPException(status_code=404, detail="Run not found")
    return r


@router.delete("/api/eval/history")
def eval_clear():
    store.clear_runs()
    return {"status": "cleared"}
