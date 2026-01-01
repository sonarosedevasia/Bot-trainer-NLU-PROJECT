import os
import json
import joblib
import torch
from fastapi import APIRouter
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


# -----------------------------------------------------
# Load Test Data
# -----------------------------------------------------
def load_test_data(bot_name: str):
    path = f"backend/split_datasets/{bot_name}/{bot_name}_test_dataset.json"
    if not os.path.exists(path):
        raise Exception("Test dataset missing.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------
# Save pending (low-confidence only)
# -----------------------------------------------------
def save_pending(bot: str, model: str, pending_list):
    folder = f"backend/active_learning_cache/{bot}"
    os.makedirs(folder, exist_ok=True)

    out_path = os.path.join(folder, f"{model}_pending.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pending_list, f, indent=4)

    return out_path


# -----------------------------------------------------
# SPA CY Evaluation
# -----------------------------------------------------
@router.post("/{bot_name}/spacy")
def eval_spacy(bot_name: str):
    import spacy

    bot = bot_name.lower().replace(" ", "_").rstrip("_")
    test_data = load_test_data(bot)

    model_dir = f"backend/models/{bot}/spacy"
    nlp = spacy.load(model_dir)

    y_true, y_pred = [], []
    detailed_rows = []
    threshold_pct = 50.0  # % confidence cutoff

    for item in test_data:
        text = item["text"]
        true_intent = item["intent"]
        ents = item.get("entities", [])

        doc = nlp(text)
        pred = max(doc.cats, key=doc.cats.get)
        conf_raw = float(doc.cats[pred])
        conf_pct = round(conf_raw * 100, 2)

        y_true.append(true_intent)
        y_pred.append(pred)

        detailed_rows.append({
            "text": text,
            "true_intent": true_intent,
            "predicted_intent": pred,
            "confidence": conf_pct,
            "status": "correct" if pred == true_intent else "wrong",
            "entities": ents
        })

    pending = [r for r in detailed_rows if r["confidence"] < threshold_pct]
    save_pending(bot, "spacy", pending)

    report = classification_report(y_true, y_pred, output_dict=True)
    macro = report["macro avg"]

    metrics = {
        "accuracy": report["accuracy"],
        "precision": macro["precision"],
        "recall": macro["recall"],
        "f1": macro["f1-score"],
    }

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "detailed": detailed_rows,
        "pending_count": len(pending)
    }


# -----------------------------------------------------
# Logistic Regression Evaluation
# -----------------------------------------------------
@router.post("/{bot_name}/logreg")
def eval_logreg(bot_name: str):

    bot = bot_name.lower().replace(" ", "_").rstrip("_")
    test_data = load_test_data(bot)

    model_dir = f"backend/models/{bot}/logreg"
    pipeline = joblib.load(f"{model_dir}/model.joblib")

    texts = [x["text"] for x in test_data]
    y_true = [x["intent"] for x in test_data]
    entities = [x.get("entities", []) for x in test_data]

    y_pred = pipeline.predict(texts)
    proba = pipeline.predict_proba(texts)

    detailed_rows = []
    threshold_pct = 50.0

    for i, text in enumerate(texts):
        true_intent = y_true[i]
        predicted = y_pred[i]

        conf_raw = float(proba[i].max())
        conf_pct = round(conf_raw * 100, 2)

        detailed_rows.append({
            "text": text,
            "true_intent": true_intent,
            "predicted_intent": predicted,
            "confidence": conf_pct,
            "status": "correct" if predicted == true_intent else "wrong",
            "entities": entities[i]
        })

    pending = [r for r in detailed_rows if r["confidence"] < threshold_pct]
    save_pending(bot, "logreg", pending)

    report = classification_report(y_true, y_pred, output_dict=True)
    macro = report["macro avg"]

    metrics = {
        "accuracy": report["accuracy"],
        "precision": macro["precision"],
        "recall": macro["recall"],
        "f1": macro["f1-score"],
    }

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "detailed": detailed_rows,
        "pending_count": len(pending)
    }


# -----------------------------------------------------
# BERT Evaluation
# -----------------------------------------------------
@router.post("/{bot_name}/bert")
def eval_bert(bot_name: str):

    bot = bot_name.lower().replace(" ", "_").rstrip("_")
    test_data = load_test_data(bot)

    model_dir = f"backend/models/{bot}/bert"

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)

    texts = [x["text"] for x in test_data]
    y_true = [x["intent"] for x in test_data]
    entities = [x.get("entities", []) for x in test_data]

    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**enc)

    preds = torch.argmax(outputs.logits, dim=1).numpy()

    labels = json.load(open(f"{model_dir}/metadata.json"))["labels"]
    y_pred = [labels[p] for p in preds]

    probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()

    detailed_rows = []
    threshold_pct = 50.0

    for i, text in enumerate(texts):
        true_intent = y_true[i]
        pred = y_pred[i]

        conf_raw = float(probs[i].max())
        conf_pct = round(conf_raw * 100, 2)

        detailed_rows.append({
            "text": text,
            "true_intent": true_intent,
            "predicted_intent": pred,
            "confidence": conf_pct,
            "status": "correct" if pred == true_intent else "wrong",
            "entities": entities[i]
        })

    pending = [r for r in detailed_rows if r["confidence"] < threshold_pct]
    save_pending(bot, "bert", pending)

    report = classification_report(y_true, y_pred, output_dict=True)
    macro = report["macro avg"]

    metrics = {
        "accuracy": report["accuracy"],
        "precision": macro["precision"],
        "recall": macro["recall"],
        "f1": macro["f1-score"],
    }

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "detailed": detailed_rows,
        "pending_count": len(pending)
    }
