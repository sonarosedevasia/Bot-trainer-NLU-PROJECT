import os
import spacy
from spacy.training.example import Example
from spacy.util import minibatch

def train_spacy_model(bot_name, train_data):

    output_dir = f"backend/models/{bot_name}/spacy"
    os.makedirs(output_dir, exist_ok=True)

    # Create blank model
    nlp = spacy.blank("en")

    # Add text categorizer
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat")
    else:
        textcat = nlp.get_pipe("textcat")

    # Collect all unique intents
    LABELS = sorted(list({item["intent"] for item in train_data}))

    # Register labels
    for label in LABELS:
        textcat.add_label(label)

    # Convert training data into Example objects
    examples = []
    for item in train_data:
        text = item["text"]
        intent = item["intent"]

        doc = nlp.make_doc(text)
        cats = {label: 0.0 for label in LABELS}
        cats[intent] = 1.0

        examples.append(Example.from_dict(doc, {"cats": cats}))

    # Initialize model
    nlp.initialize(lambda: examples)

    # Training loop
    for epoch in range(10):
        losses = {}
        batches = minibatch(examples, size=4)

        for batch in batches:
            nlp.update(batch, losses=losses)

        print(f"spaCy Training Epoch {epoch+1} Loss = {losses.get('textcat', 0)}")

    # Save model
    nlp.to_disk(output_dir)
    print("spaCy model saved at:", output_dir)
