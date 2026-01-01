import os
import json
import ast
from fastapi import APIRouter, HTTPException

# ✅ DB imports (relative to backend package)
from ..database import SessionLocal
from .. import models

router = APIRouter(prefix="/active-learning", tags=["Active Learning"])


# -----------------------------------------------------
# Helper: normalize bot name → folder friendly
# -----------------------------------------------------
def normalize_bot_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").rstrip("_")


# -----------------------------------------------------
# GET pending samples  (now uses bot_id)
# -----------------------------------------------------
@router.get("/{bot_id}/{model_type}")
def get_pending(bot_id: int, model_type: str):
    """
    Read low-confidence samples for a specific bot+model from:
    backend/active_learning_cache/<bot_folder>/<model_type>_pending.json
    """

    db = SessionLocal()
    try:
        bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first()
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found in database")

        bot_folder = normalize_bot_name(bot.name)
    finally:
        db.close()

    model = model_type.lower()
    folder = f"backend/active_learning_cache/{bot_folder}"
    file_path = os.path.join(folder, f"{model}_pending.json")

    if not os.path.exists(file_path):
        return {"samples": []}

    try:
        data = json.load(open(file_path, "r", encoding="utf-8"))
    except Exception:
        data = []

    return {"samples": data}


# -----------------------------------------------------
# SAVE correction (JSON + DB, no duplicates)
# -----------------------------------------------------
@router.post("/{bot_id}/{model_type}/save/{row_id}")
def save_correction(bot_id: int, model_type: str, row_id: int, payload: dict):
    """
    - Reads the pending.json entry at index row_id
    - Parses corrected intent & entities
    - Updates backend/nlu_datasets/<bot>/<bot>_nlu.json (replacing any old copy of that text)
    - Inserts a new row into annotations table with correct bot_id
    - Removes the entry from pending.json
    """

    db = SessionLocal()
    try:
        # 1️⃣ Find bot in DB
        bot = db.query(models.Bot).filter(models.Bot.id == bot_id).first()
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found in database")

        bot_folder = normalize_bot_name(bot.name)
        model = model_type.lower()

        # 2️⃣ Load pending file
        folder = f"backend/active_learning_cache/{bot_folder}"
        file_path = os.path.join(folder, f"{model}_pending.json")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Pending file not found")

        data = json.load(open(file_path, "r", encoding="utf-8"))

        if row_id < 0 or row_id >= len(data):
            raise HTTPException(status_code=400, detail="Invalid row_id")

        # 3️⃣ Extract corrected values from payload
        corrected_intent = payload.get("corrected_intent")
        raw_entities = payload.get("corrected_entities", [])

        if not corrected_intent:
            raise HTTPException(status_code=400, detail="Corrected intent missing")

        # ---- Parse entities ----
        if isinstance(raw_entities, str):
            try:
                corrected_entities = ast.literal_eval(raw_entities)
            except Exception:
                corrected_entities = []
        else:
            corrected_entities = raw_entities or []

        # Ensure list
        if corrected_entities is None:
            corrected_entities = []

        text = data[row_id]["text"]

        # 4️⃣ UPDATE NLU JSON (remove duplicates & add clean row)
        nlu_path = f"backend/nlu_datasets/{bot_folder}/{bot_folder}_nlu.json"
        os.makedirs(os.path.dirname(nlu_path), exist_ok=True)

        try:
            nlu_data = json.load(open(nlu_path, "r", encoding="utf-8"))
        except Exception:
            nlu_data = []

        # remove any previous entries with same text
        nlu_data = [item for item in nlu_data if item.get("text") != text]

        # add updated record
        nlu_data.append({
            "text": text,
            "intent": corrected_intent,
            "entities": corrected_entities
        })

        json.dump(nlu_data, open(nlu_path, "w", encoding="utf-8"), indent=4)

        # 5️⃣ INSERT into annotations table (DB)
        new_ann = models.Annotation(
            bot_id=bot_id,
            text=text,
            intent=corrected_intent,
            entities=json.dumps(corrected_entities, ensure_ascii=False),
            manual_entities=None  # keep same style as other annotations
        )

        db.add(new_ann)
        db.commit()
        db.refresh(new_ann)

        # 6️⃣ REMOVE from pending.json
        data.pop(row_id)
        json.dump(data, open(file_path, "w", encoding="utf-8"), indent=4)

        return {
            "detail": "Correction saved successfully (JSON + DB, no duplicates)",
            "annotation_id": new_ann.id,
        }

    finally:
        db.close()
