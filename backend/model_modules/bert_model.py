import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

def train_bert_model(bot_name, train_data):
    model_dir = f"backend/models/{bot_name}/bert"
    os.makedirs(model_dir, exist_ok=True)

    texts = [x["text"] for x in train_data]
    intents = [x["intent"] for x in train_data]

    labels = sorted(list(set(intents)))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    y = [label2id[x] for x in intents]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    encodings = tokenizer(texts, truncation=True, padding=True)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    dataset = Dataset(encodings, y)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=10,
        save_steps=30,
        save_total_limit=1,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    metadata = {
        "model_type": "bert",
        "train_samples": len(train_data),
        "labels": labels
    }
    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return True
