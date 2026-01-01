from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend import database, models, auth
from datetime import datetime

router = APIRouter(prefix="/admin", tags=["Activity Logs"])

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------
# ADD LOG ENTRY
# ---------------------------
def add_log(db, user_email, action):
    log = models.ActivityLog(
        user_email=user_email,
        action=action,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    db.add(log)
    db.commit()

# ---------------------------
# API: GET ALL LOGS
# ---------------------------
@router.get("/logs")
def get_logs(current_user=Depends(auth.get_current_user),
             db: Session = Depends(get_db)):

    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    logs = db.query(models.ActivityLog).order_by(models.ActivityLog.id.desc()).all()

    return [
        {
            "id": log.id,
            "user_email": log.user_email,
            "action": log.action,
            "timestamp": log.timestamp
        } for log in logs
    ]
