import os
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend import database
from backend.routers.admin_logs import add_log

from ..model_modules.spacy_model import train_spacy_model
from ..model_modules.logreg_model import train_logreg_model
from ..model_modules.bert_model import train_bert_model

router = APIRouter(prefix="/train", tags=["Training"])


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def load_train_data(bot_name):
    path = f"backend/split_datasets/{bot_name}/{bot_name}_train_dataset.json"
    if not os.path.exists(path):
        raise Exception("Training dataset not found. Split first.")
    return json.load(open(path, "r", encoding="utf-8"))


@router.post("/{bot_name}/{model_type}")
def train_model(
    bot_name: str,
    model_type: str,
    db: Session = Depends(get_db),
):
    bot_name = bot_name.lower().replace(" ", "_")

    # Load dataset
    try:
        train_data = load_train_data(bot_name)
    except Exception as e:
        return {"error": str(e)}

    # Train the model
    try:
        if model_type == "spacy":
            train_spacy_model(bot_name, train_data)
        elif model_type == "logreg":
            train_logreg_model(bot_name, train_data)
        elif model_type == "bert":
            train_bert_model(bot_name, train_data)
        else:
            return {"error": "Unknown model type"}
    except Exception as e:
        return {"error": f"{model_type} training failed: {str(e)}"}

    # Log action
    add_log(db, "system", f"Trained model '{model_type}' for bot '{bot_name}'")

    return {"message": f"Training complete ({model_type})"}