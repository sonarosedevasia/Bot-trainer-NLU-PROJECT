import os
import json
from fastapi import APIRouter

router = APIRouter(prefix="/compare", tags=["Model Compare"])

@router.get("/{bot_name}")
def compare_models(bot_name: str):

    bot_name = bot_name.lower()
    base = f"backend/models/{bot_name}"

    results = {}

    for model_type in ["spacy", "logreg", "bert"]:
        meta_path = f"{base}/{model_type}/metadata.json"
        if os.path.exists(meta_path):
            results[model_type] = json.load(open(meta_path))

    return results
