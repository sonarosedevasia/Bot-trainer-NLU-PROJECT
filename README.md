NLU Model Trainer and Evaluator for Chatbots:

This is a web-based platform that helps users build, train, evaluate, and continuously improve chatbot Natural Language Understanding (NLU) models.
Users can create chatbots, annotate datasets, train multiple NLU models, compare their performance, and improve accuracy using active learning.
The backend is developed using FastAPI, and the frontend interface is built using Streamlit.


Main Aim:

To provide an end-to-end system for training and evaluating chatbot intent classification and entity recognition models in an interactive and efficient way.

Key Features:

User authentication (Login & Register)
Create and manage multiple chatbots
Upload datasets (CSV / JSON)
Sentence annotation (Intent & Entities)

Train multiple NLU models:
spaCy
Logistic Regression (TF-IDF based)
BERT (Hugging Face)

Model evaluation using standard metrics
Model comparison
Active learning using low-confidence predictions
Continuous model improvement

Tech Stack:

Backend: FastAPI

Frontend: Streamlit

Database: SQLite

NLP Models: spaCy, Hugging Face (BERT), Logistic Regression

Model Evaluation Metrics:
Accuracy
Precision
Recall
F1-score
These metrics are used to measure the overall performance of each trained chatbot model.


⚙️ Installation & Setup

Create virtual environment
python -m venv venv

Activate virtual environment (Windows)
venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Run backend
uvicorn backend.main:app --reload

Run frontend
streamlit run frontend/app.py


Project Outcome:
This project simplifies chatbot NLU development by providing a complete pipeline for training, evaluation, and continuous improvement using active learning, making chatbot systems more accurate and reliable over time.
