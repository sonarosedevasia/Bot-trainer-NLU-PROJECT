import os
import json
import shutil
import pandas as pd
import spacy
from typing import List, Dict
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from backend.routers.admin_logs import add_log
from jose import jwt, JWTError
from .. import models, schemas, crud, database, auth
from fastapi.security import OAuth2PasswordBearer
import subprocess  # used for dataset split
from backend.split_dataset import split_dataset_for_bot

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# Load spaCy for annotation suggestions (small model)
# NOTE: this is only used for entity suggestions during annotation
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # Fallback: blank model (still works for tokenization but no pretrained NER)
    nlp = spacy.blank("en")


# ---------------------------
# Database Dependency
# ---------------------------
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------
# Get Current User
# ---------------------------
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = crud.get_user_by_email(db, email)
    if user is None:
        raise credentials_exception
    return user


# ---------------------------
# Bot CRUD
# ---------------------------
@router.post("/bots", response_model=schemas.BotOut)
def create_bot(bot: schemas.BotCreate, db: Session = Depends(get_db), current_user: schemas.UserOut = Depends(get_current_user)):
    return crud.create_bot(db, current_user.id, bot)


@router.get("/bots", response_model=List[schemas.BotOut])
def list_bots(db: Session = Depends(get_db), current_user: schemas.UserOut = Depends(get_current_user)):
    return crud.get_bots(db, current_user.id)


@router.delete("/bots/{bot_id}")
def remove_bot(bot_id: int, db: Session = Depends(get_db), current_user: schemas.UserOut = Depends(get_current_user)):
    if crud.delete_bot(db, bot_id, current_user.id):
        return {"detail": "Bot deleted"}
    raise HTTPException(status_code=404, detail="Bot not found")


# ---------------------------
# Dataset Upload / List / Download
# ---------------------------
@router.post("/bots/{bot_id}/datasets")
def upload_dataset(
    bot_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: schemas.UserOut = Depends(get_current_user)
):
    bot = db.query(models.Bot).filter(
        models.Bot.id == bot_id,
        models.Bot.owner_id == current_user.id
    ).first()

    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    upload_dir = "datasets"
    os.makedirs(upload_dir, exist_ok=True)

    safe_filename = file.filename
    file_path = os.path.join(upload_dir, safe_filename)

    # Prevent duplicate dataset names
    existing = db.query(models.Dataset).filter(
        models.Dataset.bot_id == bot_id,
        models.Dataset.name == safe_filename
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Dataset already exists")

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Save to DB
    dataset_record = crud.add_dataset(
        db,
        bot_id,
        schemas.DatasetCreate(name=safe_filename, file_path=file_path)
    )

    # ⭐ ADD ACTIVITY LOG HERE ⭐
    add_log(db, current_user.email, f"Uploaded dataset: {safe_filename}")

    return {
        "dataset": {
            "id": dataset_record.id,
            "name": dataset_record.name
        }
    }


@router.get("/bots/{bot_id}/datasets")
def list_datasets_for_bot(bot_id: int, db: Session = Depends(get_db), current_user: schemas.UserOut = Depends(get_current_user)):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.owner_id == current_user.id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    datasets = crud.get_datasets(db, bot_id)
    return [{"id": d.id, "name": d.name, "file_path": d.file_path} for d in datasets]


@router.get("/datasets/{dataset_id}/download")
def download_dataset(dataset_id: int, db: Session = Depends(get_db), current_user: schemas.UserOut = Depends(get_current_user)):
    dataset = crud.get_dataset_by_id(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    bot = db.query(models.Bot).filter(models.Bot.id == dataset.bot_id).first()
    if not bot or bot.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    if not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    return FileResponse(path=dataset.file_path, filename=dataset.name, media_type="application/octet-stream")


# ---------------------------
# Analyze Text (Intent + Entity Extraction) - used by annotation UI
# ---------------------------
@router.post("/bots/{bot_id}/analyze_text")
def analyze_text(bot_id: int, text: str, db: Session = Depends(get_db), current_user: schemas.UserOut = Depends(get_current_user)):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.owner_id == current_user.id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    # Use spaCy for entities (pretrained small model); intent suggestion is rule-based
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    lower = text.lower()
    bot_name = bot.name.lower()
    intent = "GeneralQuery"

    # Basic rule-based intent suggestion (same as previous)
    if "travel" in bot_name:
        if any(k in lower for k in ["cancel", "reschedule", "change booking", "postpone", "refund", "modify"]):
            intent = "TravelCancellationIntent"
        elif any(k in lower for k in ["status", "delayed", "arrival", "departure", "boarding", "schedule"]):
            intent = "TravelStatusQuery"
        elif any(k in lower for k in ["price", "cost", "fare", "budget", "rate", "charge"]):
            intent = "TravelPriceInquiry"
        elif any(k in lower for k in ["book", "reserve", "flight", "ticket", "hotel", "journey", "trip", "holiday"]):
            intent = "TravelBookingIntent"
        elif any(k in lower for k in ["destination", "places", "recommend", "plan", "spots"]):
            intent = "TravelRecommendation"

    elif "food" in bot_name:
        if any(k in lower for k in ["cancel", "remove", "change order", "refund", "replace"]):
            intent = "FoodOrderCancelIntent"
        elif any(k in lower for k in ["order", "buy", "deliver", "hungry", "craving", "send food"]):
            intent = "FoodOrderIntent"
        elif any(k in lower for k in ["menu", "dish", "offer", "special", "today"]):
            intent = "MenuInquiryIntent"
        elif any(k in lower for k in ["feedback", "rating", "experience"]):
            intent = "FoodFeedbackIntent"
        elif any(k in lower for k in ["restaurant", "nearby", "open", "location", "branch"]):
            intent = "RestaurantSearchIntent"

    elif "shop" in bot_name:
        if any(k in lower for k in ["cancel", "return", "refund", "replace", "exchange"]):
            intent = "OrderReturnIntent"
        elif any(k in lower for k in ["buy", "order", "purchase", "add to cart", "checkout"]):
            intent = "ProductOrderIntent"
        elif any(k in lower for k in ["track", "delivery", "status", "shipped", "expected"]):
            intent = "OrderTrackingIntent"
        elif any(k in lower for k in ["price", "discount", "offer", "sale", "deal"]):
            intent = "ProductPriceInquiry"
        elif any(k in lower for k in ["search", "find", "looking for", "available", "browse"]):
            intent = "ProductSearchIntent"

    elif "finance" in bot_name:
        if any(k in lower for k in ["balance", "account", "statement", "funds"]):
            intent = "AccountBalanceQuery"
        elif any(k in lower for k in ["loan", "emi", "interest", "apply", "borrow"]):
            intent = "LoanInquiryIntent"
        elif any(k in lower for k in ["credit card", "debit card", "transaction", "fraud"]):
            intent = "CardTransactionIntent"
        elif any(k in lower for k in ["investment", "stocks", "mutual fund", "portfolio"]):
            intent = "InvestmentQuery"
        elif any(k in lower for k in ["tax", "insurance", "policy", "coverage"]):
            intent = "TaxInsuranceQuery"

    elif "edu" in bot_name or "education" in bot_name:
        if any(k in lower for k in ["admission", "enroll", "register", "apply"]):
            intent = "AdmissionIntent"
        elif any(k in lower for k in ["course", "subject", "program", "syllabus"]):
            intent = "CourseInquiryIntent"
        elif any(k in lower for k in ["fees", "payment", "due", "structure"]):
            intent = "FeeInquiryIntent"
        elif any(k in lower for k in ["exam", "test", "result", "mark", "grade"]):
            intent = "ExamResultQuery"
        elif any(k in lower for k in ["faculty", "teacher", "professor", "contact"]):
            intent = "FacultyInfoIntent"

    return {"text": text, "intent": intent, "entities": entities}


# ---------------------------
# Save Annotation
# ---------------------------
@router.post("/bots/{bot_id}/save_annotation")
def save_annotation(bot_id: int, data: dict = Body(...), db: Session = Depends(get_db), current_user: schemas.UserOut = Depends(get_current_user)):
    bot = db.query(models.Bot).filter(models.Bot.id == bot_id, models.Bot.owner_id == current_user.id).first()
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    nlu_dir = os.path.join(base_dir, "..", "nlu_datasets", bot.name.strip())
    os.makedirs(nlu_dir, exist_ok=True)
    json_path = os.path.join(nlu_dir, f"{bot.name.strip()}_nlu.json")

    nlu_data = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                nlu_data = json.load(f)
        except Exception:
            nlu_data = []

    entities = data.get("entities", []) or []
    manual_entities = data.get("manual_entities", [])
    combined_entities = entities + manual_entities
    data["entities"] = combined_entities

    nlu_data.append({"text": data["text"], "intent": data["intent"], "entities": combined_entities})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(nlu_data, f, indent=4)

    ann = models.Annotation(text=data["text"], intent=data["intent"], entities=json.dumps(combined_entities), bot_id=bot_id)
    db.add(ann)
    db.commit()
    db.refresh(ann)

    return {"detail": "Annotation saved", "annotation_id": ann.id}


# ---------------------------
# List Annotated Sentences
# ---------------------------
@router.get("/bots/{bot_id}/annotations")
def get_annotations(bot_id: int, db: Session = Depends(get_db), current_user: schemas.UserOut = Depends(get_current_user)):
    annotations = db.query(models.Annotation).filter(models.Annotation.bot_id == bot_id).all()
    return [{"id": a.id, "text": a.text} for a in annotations]


# ---------------------------
# Start Training: Only triggers dataset split (keeps responsibilities separate)
# ---------------------------


@router.post("/start-training/{bot_name}")
def start_training(bot_name: str):
    bot_name = bot_name.strip().lower().replace(" ", "_").rstrip("_")

    nlu_path = f"backend/nlu_datasets/{bot_name}"
    if not os.path.exists(nlu_path):
        return {"error": f"Annotated dataset folder not found for {bot_name}."}

    try:
        # Direct function call (NO subprocess)
        split_dataset_for_bot(bot_name)
        return {"message": f"✅ Dataset successfully split for {bot_name}!"}
    except Exception as e:
        return {"error": f"❌ Dataset split failed: {str(e)}"}
