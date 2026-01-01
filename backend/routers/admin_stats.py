from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend import database, models

router = APIRouter()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/admin/stats")
def get_admin_stats(db: Session = Depends(get_db)):
    total_users = db.query(models.User).count()
    total_workspaces = db.query(models.Bot).count()
    total_datasets = db.query(models.Dataset).count()
    total_models = db.query(models.TrainedModel).count() if hasattr(models, "TrainedModel") else 0
    total_annotations = db.query(models.Annotation).count()

    return {
        "total_users": total_users,
        "total_workspaces": total_workspaces,
        "total_datasets": total_datasets,
        "total_models": total_models,
        "total_annotations": total_annotations
    }
