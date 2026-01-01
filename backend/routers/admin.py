from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend import database, models, auth
from backend.routers.admin_logs import add_log
from fastapi.responses import FileResponse
import os
import requests

router = APIRouter(prefix="/admin", tags=["Admin Management"])

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------
# GET ALL USERS
# -------------------------------
@router.get("/users")
def get_all_users(
    db: Session = Depends(get_db),
    current_user=Depends(auth.get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    users = db.query(models.User).all()
    return [{"id": u.id, "email": u.email, "role": u.role} for u in users]


# -------------------------------
# DELETE USER
# -------------------------------
@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(auth.get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    add_log(db, current_user.email, f"Deleted user: {user.email}")
    return {"message": "User deleted successfully"}


# -------------------------------
# GET ALL WORKSPACES
# -------------------------------
@router.get("/workspaces")
def get_all_workspaces(
    db: Session = Depends(get_db),
    current_user=Depends(auth.get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    bots = db.query(models.Bot).all()
    return [
        {
            "id": b.id,
            "name": b.name,
            "owner_email": b.owner.email if b.owner else "-",
            "datasets": len(b.datasets),
            "annotations": len(b.annotations),
        }
        for b in bots
    ]


# -------------------------------
# GET ALL DATASETS
# -------------------------------
@router.get("/datasets")
def get_all_datasets(
    db: Session = Depends(get_db),
    current_user=Depends(auth.get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    datasets = db.query(models.Dataset).all()
    return [
        {
            "id": d.id,
            "name": d.name,
            "file_path": d.file_path,
            "bot_name": d.bot.name if d.bot else "-",
            "owner_email": d.bot.owner.email if d.bot else "-"
        }
        for d in datasets
    ]


# -------------------------------
# GET ALL MODELS METRICS
# -------------------------------
@router.get("/models")
def get_all_models(
    db: Session = Depends(get_db),
    current_user=Depends(auth.get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    bots = db.query(models.Bot).all()
    MODEL_TYPES = ["spacy", "logreg", "bert"]
    result = []

    for bot in bots:
        bot_name = bot.name.lower().replace(" ", "_").rstrip("_")
        for m in MODEL_TYPES:
            try:
                r = requests.post(f"http://127.0.0.1:8000/evaluate/{bot_name}/{m}")
                if r.status_code == 200:
                    metrics = r.json().get("metrics", {})
                    result.append({
                        "bot_name": bot.name,
                        "model_type": m,
                        "accuracy": metrics.get("accuracy"),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1": metrics.get("f1"),
                        "trained": True
                    })
                else:
                    raise Exception()
            except:
                result.append({
                    "bot_name": bot.name,
                    "model_type": m,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "trained": False
                })
    return result


# -------------------------------
# ✅ GET ALL FEEDBACKS (FIXED)
# -------------------------------
@router.get("/feedbacks")
def get_all_feedbacks(
    db: Session = Depends(get_db),
    current_user=Depends(auth.get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    feedbacks = (
        db.query(models.Feedback, models.Bot)
        .join(models.Bot, models.Feedback.bot_id == models.Bot.id)
        .order_by(models.Feedback.timestamp.desc())   # ✅ FIX
        .all()
    )

    return [
        {
            "user_email": f.user_email,
            "bot_name": bot.name,
            "feedback_text": f.feedback_text,
            "timestamp": f.timestamp
        }
        for f, bot in feedbacks
    ]
