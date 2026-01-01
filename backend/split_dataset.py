import os
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset_for_bot(bot_name):
    """
    Splits the annotated dataset for a specific bot (80/20).
    Input JSON: backend/nlu_datasets/{bot_name}/{bot_name}_nlu.json
    Output: backend/split_datasets/{bot_name}/
       - {bot_name}_train_dataset.json
        bot_name}_test_dataset.json
       - {bot_name}_train_dataset.csv
       - {bot_name}_test_dataset.csv
    """

    base_path = f"backend/nlu_datasets/{bot_name}"
    output_path = f"backend/split_datasets/{bot_name}"
    os.makedirs(output_path, exist_ok=True)

    json_file = os.path.join(base_path, f"{bot_name}_nlu.json")
    if not os.path.exists(json_file):
        print(f"❌ No annotated dataset found for {bot_name}")
        return

    # Load JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"❌ Invalid data format in {json_file}. Expected list.")
        return

    # 80/20 split
    train_ds, test_ds = train_test_split(data, test_size=0.2, random_state=42)

    # Save JSON
    train_json = os.path.join(output_path, f"{bot_name}_train_dataset.json")
    test_json = os.path.join(output_path, f"{bot_name}_test_dataset.json")

    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_ds, f, indent=4, ensure_ascii=False)
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(test_ds, f, indent=4, ensure_ascii=False)

    # Convert to CSV
    def json_to_csv(data, csv_path):
        rows = []
        for item in data:
            text = item.get("text", "")
            intent = item.get("intent", "")
            entities = item.get("entities", [])
            entity_str = "; ".join([f"{e['text']}:{e['label']}" for e in entities])
            rows.append({"text": text, "intent": intent, "entities": entity_str})

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding="utf-8")

    train_csv = os.path.join(output_path, f"{bot_name}_train_dataset.csv")
    test_csv = os.path.join(output_path, f"{bot_name}_test_dataset.csv")

    json_to_csv(train_ds, train_csv)
    json_to_csv(test_ds, test_csv)

    print(f"✅ Split complete for {bot_name}")
    print(f"   Train JSON: {train_json}")
    print(f"   Test JSON: {test_json}")
    print(f"   Train CSV: {train_csv}")
    print(f"   Test CSV: {test_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Provide bot name. Example: python split_dataset.py travel_bot")
    else:
        bot_name = sys.argv[1].strip().lower().replace(" ", "_").rstrip("_")
        split_dataset_for_bot(bot_name)
