import os
import json
import joblib
from typing import List, Dict

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
#  TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

def train_logreg_model(bot_name: str, train_data: List[Dict]) -> bool:
    """
    Train a Logistic Regression model using TF-IDF pipeline.

    Saves:
    - backend/models/<bot_name>/logreg/model.joblib
    - backend/models/<bot_name>/logreg/metadata.json
    """

    model_dir = f"backend/models/{bot_name}/logreg"
    os.makedirs(model_dir, exist_ok=True)

    # Extract data
    texts = [item["text"] for item in train_data]
    intents = [item["intent"] for item in train_data]

    # TF-IDF + Logistic Regression pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])

    # Train
    pipeline.fit(texts, intents)

    # Save model
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(pipeline, model_path)

    # Save metadata
    metadata = {
        "model_type": "logreg",
        "train_samples": len(train_data)
    }

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    return True


# ============================================================
#  ACTIVE LEARNING — RETURN LOW CONFIDENCE SAMPLES
# ============================================================

def get_logreg_active_learning_samples(
    bot_name: str,
    threshold: float = 0.5
) -> List[Dict]:
    """
    Safe version:
    - Checks files exist 
    - Returns [] instead of crashing
    """

    base_dir = os.path.dirname(os.path.dirname(__file__))

    # Model path
    model_path = os.path.join(base_dir, "models", bot_name, "logreg", "model.joblib")
    if not os.path.exists(model_path):
        return []   # << SAFE RETURN

    # Test dataset path
    test_json_path = os.path.join(
        base_dir,
        "split_datasets",
        bot_name,
        f"{bot_name}_test_dataset.json"
    )
    if not os.path.exists(test_json_path):
        return []   # << SAFE RETURN

    # Load model
    pipeline = joblib.load(model_path)

    # Load test dataset
    try:
        with open(test_json_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except:
        return []   # << SAFE RETURN

    texts = [item.get("text", "") for item in test_data]
    true_intents = [item.get("intent") for item in test_data]
    entities_list = [item.get("entities", []) for item in test_data]

    if not texts:
        return []

    # Predict probabilities
    proba = pipeline.predict_proba(texts)
    class_labels = pipeline.classes_

    low_conf_samples = []

    for idx, (text, true_intent, ents, probs) in enumerate(
        zip(texts, true_intents, entities_list, proba)
    ):
        max_idx = probs.argmax()
        confidence = float(probs[max_idx])
        predicted_intent = str(class_labels[max_idx])

        if confidence < threshold:
            low_conf_samples.append({
                "row_id": idx,
                "text": text,
                "true_intent": true_intent,
                "predicted_intent": predicted_intent,
                "confidence": confidence,
                "entities": ents,
            })

    return low_conf_samples
