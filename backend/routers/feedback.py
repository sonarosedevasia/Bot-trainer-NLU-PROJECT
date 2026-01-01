from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime

from backend import database, models, schemas, auth

router = APIRouter(prefix="/feedback", tags=["Feedback"])


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/", response_model=schemas.FeedbackOut)
def submit_feedback(
    feedback: schemas.FeedbackCreate,
    db: Session = Depends(get_db),
    current_user = Depends(auth.get_current_user)
):
    """
    Save feedback submitted by a user for a specific bot
    """

    new_feedback = models.Feedback(
        user_email=current_user.email,
        bot_id=feedback.bot_id,
        feedback_text=feedback.feedback_text,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    db.add(new_feedback)
    db.commit()
    db.refresh(new_feedback)

    return new_feedback
