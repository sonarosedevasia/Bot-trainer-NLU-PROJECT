
import streamlit as st
import requests
import pandas as pd
import io
import json
import os
import matplotlib.pyplot as plt   
import seaborn as sns           
import numpy as np 

# ---------------------------
# Backend URL
# ---------------------------
BACKEND_URL = "http://127.0.0.1:8000"

# ---------------------------
# Session State (initial)
# ---------------------------
def render_admin_sidebar():
    # Show sidebar only for admin
    if st.session_state.get("role") != "admin":
        return

    st.sidebar.title("👑 Admin Menu")

    # Dashboard button
    if st.sidebar.button("🏠 Dashboard"):
        st.session_state.page = "admin_dashboard"
        st.rerun()

    # Admin panel button  ✅ FIXED
    if st.sidebar.button("🛠 Admin Panel"):
        st.session_state.page = "admin_panel"
        st.rerun()

    # Logout
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.session_state.page = "login"
        st.rerun()

if "page" not in st.session_state:
    st.session_state.page = "login"
if "role" not in st.session_state:
    st.session_state.role = "user"

if "token" not in st.session_state:
    st.session_state.token = None
if "email" not in st.session_state:
    st.session_state.email = None
if "selected_bot" not in st.session_state:
    st.session_state.selected_bot = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()
if "entities" not in st.session_state:
    st.session_state.entities = []
if "manual_entities" not in st.session_state:
    st.session_state.manual_entities = []
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
# For "quick test & save to training" inside model_train_page
if "train_preview" not in st.session_state:
    st.session_state.train_preview = None
if "train_entities" not in st.session_state:
    st.session_state.train_entities = []
if "train_manual_entities" not in st.session_state:
    st.session_state.train_manual_entities = []
if "train_last_sentence" not in st.session_state:
    st.session_state.train_last_sentence = ""


# ---------------------------
# Helpers
# ---------------------------
def api_headers():
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    return headers

# ---------------------------
# API Calls
# ---------------------------

# ✅ FIXED login_user()
def login_user(email, password):
    try:
        data = {
            "username": email,   
            "password": password
        }

        r = requests.post(f"{BACKEND_URL}/login", data=data)

        if r.status_code == 200:
            res_data = r.json()

            st.session_state.token = res_data["access_token"]
            st.session_state.email = email
            st.session_state.role = res_data.get("role", "user")

            return True

        else:
            try:
                st.error(r.json())
            except Exception:
                st.error("Login failed.")
            return False

    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
        return False




def register_user(email, password):
    r = requests.post(f"{BACKEND_URL}/register",
                      json={"email": email, "password": password})
    return r.status_code in [200, 201]



def create_bot_api(name, domain):
    headers = api_headers()
    response = requests.post(f"{BACKEND_URL}/bots", json={"name": name, "domain": domain}, headers=headers)
    return response.json() if response.status_code in [200, 201] else None


def list_bots_api():
    headers = api_headers()
    response = requests.get(f"{BACKEND_URL}/bots", headers=headers)
    return response.json() if response.status_code == 200 else []


def delete_bot_api(bot_id):
    headers = api_headers()
    requests.delete(f"{BACKEND_URL}/bots/{bot_id}", headers=headers)
    st.success("Bot deleted successfully!")


def upload_dataset_api(bot_id, uploaded_file):
    headers = api_headers()
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    return requests.post(f"{BACKEND_URL}/bots/{bot_id}/datasets", files=files, headers=headers)


def list_datasets_for_bot_api(bot_id):
    headers = api_headers()
    response = requests.get(f"{BACKEND_URL}/bots/{bot_id}/datasets", headers=headers)
    return response.json() if response.status_code == 200 else []


def download_dataset_api(dataset_id):
    headers = api_headers()
    return requests.get(f"{BACKEND_URL}/datasets/{dataset_id}/download", headers=headers, stream=True)


def analyze_text_api(bot_id, text):
    headers = api_headers()
    return requests.post(f"{BACKEND_URL}/bots/{bot_id}/analyze_text", params={"text": text}, headers=headers)


def save_annotation_api(bot_id, text, intent, entities, manual_entities):
    headers = api_headers()
    payload = {"text": text, "intent": intent, "entities": entities, "manual_entities": manual_entities}
    return requests.post(f"{BACKEND_URL}/bots/{bot_id}/save_annotation", json=payload, headers=headers)


def get_annotations_api(bot_id):
    headers = api_headers()
    response = requests.get(f"{BACKEND_URL}/bots/{bot_id}/annotations", headers=headers)
    return response.json() if response.status_code == 200 else []


def start_training_api(bot_name):
    try:
        response = requests.post(f"{BACKEND_URL}/start-training/{bot_name}")
        return response.json()
    except Exception as e:
        return {"error": f"⚠ Failed to connect to backend: {e}"}


def train_model_api(bot_name, model_type):
    try:
        response = requests.post(f"{BACKEND_URL}/train/{bot_name}/{model_type}")
        return response.json()
    except Exception as e:
        return {"error": f"⚠ Failed to connect to backend: {e}"}


def evaluate_model_api(bot_name, model_type):
    try:
        response = requests.post(f"{BACKEND_URL}/evaluate/{bot_name}/{model_type}")
        return response.json()
    except Exception as e:
        return {"error": f"⚠ Failed to connect to backend: {e}"}


def compare_models_api(bot_name):
    try:
        response = requests.get(f"{BACKEND_URL}/compare/{bot_name}")
        return response.json() if response.status_code == 200 else {}
    except Exception as e:
        return {"error": str(e)}
def get_admin_stats_api():
    headers = api_headers()
    res = requests.get(f"{BACKEND_URL}/admin/stats", headers=headers)
    if res.status_code == 200:
        return res.json()
    return {
        "total_users": 0,
        "total_workspaces": 0,
        "total_datasets": 0,
        "total_models": 0,
        "total_annotations": 0
    }
def get_all_feedbacks_api():
    headers = api_headers()
    r = requests.get(f"{BACKEND_URL}/admin/feedbacks", headers=headers)
    return r.json() if r.status_code == 200 else []

def get_all_users_api():
    headers = api_headers()
    r = requests.get(f"{BACKEND_URL}/admin/users", headers=headers)
    return r.json() if r.status_code == 200 else []

def delete_user_api(user_id):
    headers = api_headers()
    r = requests.delete(f"{BACKEND_URL}/admin/users/{user_id}", headers=headers)
    return r.status_code == 200
def get_all_workspaces_api():
    headers = api_headers()
    r = requests.get(f"{BACKEND_URL}/admin/workspaces", headers=headers)
    return r.json() if r.status_code == 200 else []


def delete_workspace_api(bot_id):
    headers = api_headers()
    r = requests.delete(f"{BACKEND_URL}/admin/workspaces/{bot_id}", headers=headers)
    return r.status_code == 200

def submit_feedback_api(bot_id, feedback_text):
    headers = api_headers()
    payload = {
        "bot_id": bot_id,
        "feedback_text": feedback_text
    }
    return requests.post(
        f"{BACKEND_URL}/feedback/",
        json=payload,
        headers=headers
    )

# ----------------------------------------------------
# ADMIN — GET ALL DATASETS
# ----------------------------------------------------
def get_all_datasets_api():
    headers = api_headers()
    r = requests.get(f"{BACKEND_URL}/admin/datasets", headers=headers)
    return r.json() if r.status_code == 200 else []
def get_activity_logs_api():
    headers = api_headers()
    r = requests.get(f"{BACKEND_URL}/admin/logs", headers=headers)
    return r.json() if r.status_code == 200 else []



# ----------------------------------------------------
# ADMIN — DELETE A SOLITARY DATASET
# ----------------------------------------------------
def delete_dataset_api(dataset_id):
    headers = api_headers()
    r = requests.delete(f"{BACKEND_URL}/admin/datasets/{dataset_id}", headers=headers)
    return r.status_code == 200


# ----------------------------------------------------
# ADMIN — DOWNLOAD DATASET
# ----------------------------------------------------
def download_dataset_admin_api(dataset_id):
    headers = api_headers()
    r = requests.get(f"{BACKEND_URL}/admin/datasets/{dataset_id}/download",
                     headers=headers)

    if r.status_code == 200:
        return r.content  # bytes
    return None
def get_all_models_api():
    headers = api_headers()
    r = requests.get(f"{BACKEND_URL}/admin/models", headers=headers)
    return r.json() if r.status_code == 200 else []
def get_confusion_matrix_api(bot_name, model_type):
    headers = api_headers()
    r = requests.get(f"{BACKEND_URL}/admin/models/{bot_name}/{model_type}/confusion", headers=headers)
    if r.status_code == 200:
        return r.json()
    return None

# ---------------------------
# Pages
# ---------------------------
def login_page():
    st.markdown("<h1 style='text-align:center;color:#00FFFF;'>CHATBOT NLU TRAINER & EVALUATOR</h1>", unsafe_allow_html=True)

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if login_user(email, password):

            # Redirect based on role
            if st.session_state.role == "admin":
                st.session_state.page = "admin_dashboard"
            else:
                st.session_state.page = "workspace"

            st.rerun()

    if st.button("Register"):
        st.session_state.page = "register"
        st.rerun()



def register_page():
    st.title("Register")
    email = st.text_input("Email", key="reg_email")
    password = st.text_input("Password", type="password", key="reg_pass")
    confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
    if st.button("Register"):
        if password == confirm and register_user(email, password):
            st.success("Account created successfully!")
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error("Failed or passwords do not match")
    if st.button("Go to Login"):
        st.session_state.page = "login"
        st.rerun()


def workspace_page():
    if st.session_state.role == "admin":
        st.session_state.page = "admin_dashboard"
        st.rerun()

    col1, col2 = st.columns([5, 1])
    with col1:
        st.title(f"Welcome, {st.session_state.email}")
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.session_state.page = "login"
            st.rerun()

    bots = list_bots_api()
    for b in bots:
        st.markdown(f"### 🤖 {b['name']} ({b.get('domain', '')})")
        c1, c2, c3, c4, c5, c6 = st.columns([1,1.2,1,1.5,1,1])

        with c1:
            if st.button("📁 Upload Dataset", key=f"upload_{b['id']}"):
                st.session_state.selected_bot = b
                st.session_state.page = "dataset_page"
                st.rerun()
        with c2:
            if st.button("📝 Annotation Interface", key=f"annot_{b['id']}"):
                st.session_state.selected_bot = b
                st.session_state.page = "train_page"
                st.rerun()
        with c3:
            if st.button("🧠 Train Models", key=f"train_{b['id']}"):
                st.session_state.selected_bot = b
                st.session_state.page = "model_train_page"
                st.rerun()
        with c4:
            if st.button("🔁 Model Comparison", key=f"compare_{b['id']}"):
                st.session_state.selected_bot = b
                st.session_state.page = "model_compare_page"
                st.rerun()
        with c5:
            if st.button("🔍 Active Learning", key=f"active_{b['id']}"):
                st.session_state.selected_bot = b
                st.session_state.page = "active_learning_page"
                st.rerun()
        with c6:
            if st.button("💬 Feedback", key=f"feedback_{b['id']}"):
                st.session_state.selected_bot = b
                st.session_state.page = "feedback_page"
                st.rerun()


       

    st.divider()
    st.subheader("Create Bot")
    name = st.text_input("Bot Name", key="bot_name")
    domain = st.text_input("Bot Domain", key="bot_domain")
    if st.button("Create Bot"):
        bot = create_bot_api(name, domain)
        if bot:
            st.success(f"Bot '{name}' created!")
            st.rerun()


def bot_dashboard():
    bot = st.session_state.selected_bot
    if not bot:
        st.error("No bot selected.")
        return

    st.title(f"🤖 {bot['name']}")
    st.caption(f"Domain: {bot.get('domain', 'N/A')}")

    c1, c2, c3, c4,c5= st.columns(5)
    with c1:
        if st.button("📁 Upload Dataset"):
            st.session_state.page = "dataset_page"
            st.rerun()
    with c2:
        if st.button("📝 Annotation Interface"):
            st.session_state.page = "train_page"
            st.rerun()
    with c3:
        if st.button("🧠 Train Models"):
            st.session_state.page = "model_train_page"
            st.rerun()
    with c4:
        if st.button("🔁 Model Comparison"):
            st.session_state.page = "model_compare_page"
            st.rerun()
    with c5:
        if st.button("🔍 Active Learning"):
            st.session_state.page = "active_learning_page"
            st.rerun()

    if st.button("⬅ Back to Workspace"):
        st.session_state.page = "workspace"
        st.rerun()


def dataset_page():
    bot = st.session_state.selected_bot
    st.title("📁 Upload & Manage Datasets")

    uploaded_file = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
    if uploaded_file:
        if uploaded_file.name not in st.session_state.uploaded_files:
            resp = upload_dataset_api(bot["id"], uploaded_file)
            if getattr(resp, "status_code", None) in [200, 201] or (isinstance(resp, dict) and resp.get("dataset")):
                st.success("Dataset uploaded successfully!")
                st.session_state.uploaded_files.add(uploaded_file.name)
                st.rerun()
            else:
                st.error("Upload failed.")
        else:
            st.info("This file is already uploaded.")

    st.subheader("Uploaded Datasets")
    datasets = list_datasets_for_bot_api(bot["id"])
    for ds in datasets:
        st.markdown(f"{ds['name']}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Preview", key=f"preview_{ds['id']}"):
                data = download_dataset_api(ds["id"]).content
                try:
                    df = pd.read_csv(io.BytesIO(data), encoding="utf-8")
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(io.BytesIO(data), encoding="latin1")
                    except Exception:
                        try:
                            df = pd.read_csv(io.BytesIO(data), header=0, names=["sentences"], encoding="latin1")
                        except Exception:
                            df = pd.read_csv(io.BytesIO(data), sep=",", on_bad_lines="skip", engine="python", encoding="latin1")
                except Exception:
                    try:
                        df = pd.read_json(io.BytesIO(data))
                    except Exception:
                        st.error("⚠ Unable to preview this dataset format.")
                        continue
                st.dataframe(df.head())
        with c2:
            data = download_dataset_api(ds["id"]).content
            st.download_button("⬇ Download", data=data, file_name=ds["name"])

    if st.button("⬅ Back"):
        st.session_state.page = "bot_dashboard"
        st.rerun()


def train_page():
    bot = st.session_state.selected_bot
    st.title("🧠 Train Model / Annotate (Annotation Interface)")

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🚀 Split Annotated Dataset (80/20)"):
            with st.spinner("⏳ Splitting annotated dataset..."):
                bot_name = bot["name"].lower().replace(" ", "").rstrip("")
                result = start_training_api(bot_name)
                if "message" in result:
                    st.success(result["message"])
                else:
                    st.error(result.get("error", "Unknown error occurred"))

    user_input = st.text_input("Enter a sentence")

    if user_input != st.session_state.last_input:
        st.session_state.pop("annot_result", None)
        st.session_state["entities"] = []
        st.session_state["manual_entities"] = []
        st.session_state.last_input = user_input

    if st.button("Auto-Annotate"):
        if not user_input.strip():
            st.warning("Please enter a sentence to analyze.")
        else:
            resp = analyze_text_api(bot["id"], user_input)
            if getattr(resp, "status_code", None) == 200:
                st.session_state["annot_result"] = resp.json()
                st.session_state["entities"] = st.session_state["annot_result"]["entities"]
                st.success("Analyzed successfully!")
                st.rerun()
            elif isinstance(resp, dict):
                st.session_state["annot_result"] = resp
                st.session_state["entities"] = resp.get("entities", [])
                st.success("Analyzed successfully!")
                st.rerun()
            else:
                st.error("Failed to analyze.")

    if "annot_result" in st.session_state:
        result = st.session_state["annot_result"]
        intent = st.text_input("Intent", value=result.get("intent", ""), key="intent_input")

        st.markdown("Auto-detected Entities (Editable)")
        delete_index = None
        for i, ent in enumerate(st.session_state["entities"]):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                text = st.text_input(f"Entity Text {i}", value=ent.get("text", ""), key=f"text_{i}")
            with c2:
                label = st.text_input(f"Entity Label {i}", value=ent.get("label", ""), key=f"label_{i}")
            with c3:
                if st.button("🗑", key=f"del_ent_{i}"):
                    delete_index = ("entities", i)
            st.session_state["entities"][i] = {"text": text, "label": label}

        if delete_index and delete_index[0] == "entities":
            del st.session_state["entities"][delete_index[1]]
            st.rerun()

        st.markdown("### ➕ Add Manual Entities")
        delete_manual = None
        for j, ment in enumerate(st.session_state["manual_entities"]):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                text = st.text_input(f"Manual Entity Text {j}", value=ment.get("text", ""), key=f"m_text_{j}")
            with c2:
                label = st.text_input(f"Manual Entity Label {j}", value=ment.get("label", ""), key=f"m_label_{j}")
            with c3:
                if st.button("🗑", key=f"del_m_ent_{j}"):
                    delete_manual = j
            st.session_state["manual_entities"][j] = {"text": text, "label": label}

        if delete_manual is not None:
            del st.session_state["manual_entities"][delete_manual]
            st.rerun()

        if st.button("Add New Entity"):
            st.session_state["manual_entities"].append({"text": "", "label": ""})
            st.rerun()

        if st.button("💾 Save Annotation"):
            res = save_annotation_api(bot["id"], user_input, intent, st.session_state["entities"], st.session_state["manual_entities"])
            if getattr(res, "status_code", None) in [200, 201] or (isinstance(res, dict) and res.get("detail")):
                st.success("Annotation saved successfully!")
                st.session_state.pop("annot_result", None)
                st.session_state["entities"] = []
                st.session_state["manual_entities"] = []
                st.rerun()
            else:
                st.error("Failed to save annotation.")

    st.markdown("### 📋 Annotated Sentences")
    data = get_annotations_api(bot["id"])
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df)
    else:
        st.info("No annotated sentences yet.")

    if st.button("⬅ Back"):
        st.session_state.page = "bot_dashboard"
        st.rerun()

def model_train_page():
    bot = st.session_state.selected_bot
    if not bot:
        st.error("No bot selected.")
        return

    st.title("🧠 Train Models (select model)")
    st.write(f"Selected bot: {bot['name']}")

    # -------------------------------------------------
    # 🔍 Quick test + save sentence to training data
    # -------------------------------------------------
    st.markdown("### 🔍 Test a sentence and add it to training data")

    test_sentence = st.text_input(
        "Enter a sentence to test & optionally save for training",
        key="train_test_sentence"
    )

    # Reset preview when user types a new sentence
    if test_sentence != st.session_state.train_last_sentence:
        st.session_state.train_preview = None
        st.session_state.train_entities = []
        st.session_state.train_manual_entities = []
        st.session_state.train_last_sentence = test_sentence

    # ---- Predict intent & entities ----
    if st.button("✨ Predict intent & entities", key="btn_predict_train"):
        if not test_sentence.strip():
            st.warning("Please enter a sentence first.")
        else:
            resp = analyze_text_api(bot["id"], test_sentence)
            data = None
            if getattr(resp, "status_code", None) == 200:
                data = resp.json()
            elif isinstance(resp, dict):
                data = resp
            if not data:
                st.error("Failed to analyze sentence.")
            else:
                st.session_state.train_preview = {
                    "text": test_sentence,
                    "intent": data.get("intent", ""),
                }
                st.session_state.train_entities = data.get("entities", []) or []
                st.session_state.train_manual_entities = []
                st.success("Prediction ready. You can edit and save below.")
                st.rerun()

    # ---- If we have a preview, show editable UI ----
    if st.session_state.train_preview:
        prev = st.session_state.train_preview

        st.markdown("#### ✅ Review & edit prediction")

        # Show sentence (read-only)
        st.text_area("Sentence", value=prev["text"], disabled=True)

        # Editable intent
        intent_value = st.text_input(
            "Intent",
            value=prev.get("intent", ""),
            key="train_preview_intent"
        )

        # -------- Auto-detected entities (editable) --------
        st.markdown("*Auto-detected entities (you can edit them):*")
        delete_auto_idx = None
        for i, ent in enumerate(st.session_state.train_entities):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                text = st.text_input(
                    f"Entity Text {i}",
                    value=ent.get("text", ""),
                    key=f"train_ent_text_{i}"
                )
            with c2:
                label = st.text_input(
                    f"Entity Label {i}",
                    value=ent.get("label", ""),
                    key=f"train_ent_label_{i}"
                )
            with c3:
                if st.button("🗑", key=f"btn_del_train_ent_{i}"):
                    delete_auto_idx = i
            st.session_state.train_entities[i] = {"text": text, "label": label}

        if delete_auto_idx is not None:
            del st.session_state.train_entities[delete_auto_idx]
            st.rerun()

        # -------- Manual entities --------
        st.markdown("➕ Add manual entities (optional):")
        delete_manual_idx = None
        for j, ment in enumerate(st.session_state.train_manual_entities):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                text = st.text_input(
                    f"Manual Entity Text {j}",
                    value=ment.get("text", ""),
                    key=f"train_m_ent_text_{j}"
                )
            with c2:
                label = st.text_input(
                    f"Manual Entity Label {j}",
                    value=ment.get("label", ""),
                    key=f"train_m_ent_label_{j}"
                )
            with c3:
                if st.button("🗑", key=f"btn_del_train_m_ent_{j}"):
                    delete_manual_idx = j
            st.session_state.train_manual_entities[j] = {"text": text, "label": label}

        if delete_manual_idx is not None:
            del st.session_state.train_manual_entities[delete_manual_idx]
            st.rerun()

        if st.button("➕ Add another manual entity", key="btn_add_train_m_ent"):
            st.session_state.train_manual_entities.append({"text": "", "label": ""})
            st.rerun()

        # ---- Save to annotation DB + JSON ----
        if st.button("💾 Save this sentence to training data", key="btn_save_train_sentence"):
            combined_entities = st.session_state.train_entities
            manual_ents = st.session_state.train_manual_entities

            res = save_annotation_api(
                bot["id"],
                prev["text"],
                intent_value,
                combined_entities,
                manual_ents,
            )

            if getattr(res, "status_code", None) in [200, 201] or (
                isinstance(res, dict) and res.get("detail")
            ):
                st.success(
                    "Sentence saved! It is stored like other annotations and "
                    "will be included next time you split the dataset."
                )
                st.session_state.train_preview = None
                st.session_state.train_entities = []
                st.session_state.train_manual_entities = []
            else:
                st.error("Failed to save the sentence.")

    st.markdown("---")
    st.markdown("### 🧠 Train Model")

    st.markdown("*Choose model to train*")
    model_type = st.selectbox(
        "Model Type",
        ["spacy", "logreg", "bert"],
        key="train_model_select"
    )

    # -------- Train button --------
    if st.button("Train Selected Model", key="btn_train_model"):
        with st.spinner("Training..."):
            bot_name = bot["name"].lower().replace(" ", "").rstrip("")
            result = train_model_api(bot_name, model_type)
            if isinstance(result, dict) and result.get("message"):
                st.success(result["message"])
            else:
                st.error(result.get("error", "Training failed."))

    st.markdown("---")
    st.subheader("📊 Evaluate Models")

    # -------- Helper to show metrics + CM + table --------
    def show_results(model_type: str):
        bot_name = bot["name"].lower().replace(" ", "").rstrip("")
        res = evaluate_model_api(bot_name, model_type)

        if not isinstance(res, dict):
            st.error("Unexpected response from backend.")
            st.write(res)
            return
        if res.get("error"):
            st.error(res["error"])
            return

        metrics = res.get("metrics", {}) or {}
        cm = res.get("confusion_matrix")
        detailed = res.get("detailed", []) or []
        pending_count = res.get("pending_count", 0)

        accuracy = metrics.get("accuracy")
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        f1 = metrics.get("f1")

        st.markdown("### ⚙ Overall Performance")

        if accuracy is not None:
            st.write(f"*Accuracy:* {accuracy:.3f}")
        if precision is not None:
            st.write(f"*Precision (macro):* {precision:.3f}")
        if recall is not None:
            st.write(f"*Recall (macro):* {recall:.3f}")
        if f1 is not None:
            st.write(f"*F1-score (macro):* {f1:.3f}")

        st.info(f"Low-confidence samples (< 50% confidence) sent to Active Learning: *{pending_count}*")

        # ----- Confusion Matrix -----
        if cm:
            st.markdown("### 📉 Confusion Matrix")
            cm_array = np.array(cm)

            fig, ax = plt.subplots()
            sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)
        else:
            st.info("No confusion matrix returned for this model.")

        # ----- Detailed Predictions Table -----
        if detailed:
            st.markdown("### 🔍 Detailed Predictions")

            df = pd.DataFrame(detailed)

            # Only keep the 4 columns you want
            cols = [c for c in ["text", "true_intent", "predicted_intent", "confidence", "status"] if c in df.columns]
            df = df[cols]

            # Ensure confidence is float and display as %
            if "confidence" in df.columns:
                df["confidence"] = df["confidence"].astype(float)

            st.dataframe(df)
            st.caption("Confidence is shown in %, only low-confidence (< 50%) are used in Active Learning.")
        else:
            st.info("No detailed predictions returned by backend.")

    # -------- Evaluate buttons --------
    if st.button("Evaluate spaCy", key="btn_eval_spacy"):
        show_results("spacy")

    if st.button("Evaluate Logistic Regression", key="btn_eval_logreg"):
        show_results("logreg")

    if st.button("Evaluate BERT", key="btn_eval_bert"):
        show_results("bert")

    # -------- Back button --------
    if st.button("⬅ Back", key="btn_back_model_train"):
        st.session_state.page = "bot_dashboard"
        st.rerun()

def model_compare_page():
    bot = st.session_state.selected_bot
    if not bot:
        st.error("No bot selected.")
        return

    st.title("🔁 Model Comparison")

    # Normalized bot name
    bot_name = bot["name"].lower().replace(" ", "").rstrip("")

    # ------------------------------------------------
    # 1️⃣ Get train/test sample counts
    # ------------------------------------------------
    train_samples = None
    test_samples = None

    train_path = f"backend/split_datasets/{bot_name}/{bot_name}_train_dataset.json"
    test_path = f"backend/split_datasets/{bot_name}/{bot_name}_test_dataset.json"

    try:
        if os.path.exists(train_path):
            with open(train_path, "r", encoding="utf-8") as f:
                train_samples = len(json.load(f))
    except:
        train_samples = None

    try:
        if os.path.exists(test_path):
            with open(test_path, "r", encoding="utf-8") as f:
                test_samples = len(json.load(f))
    except:
        test_samples = None

    # ------------------------------------------------
    # 2️⃣ Evaluate all models (spaCy, LogReg, BERT)
    # ------------------------------------------------
    model_defs = [
        ("spaCy", "spacy"),
        ("Logistic Regression", "logreg"),
        ("BERT", "bert"),
    ]

    results = []

    for nice_name, key in model_defs:
        res = evaluate_model_api(bot_name, key)

        if not isinstance(res, dict) or res.get("error"):
            continue

        metrics = res.get("metrics", {})
        if not isinstance(metrics, dict):
            continue

        # NEW METRICS FORMAT
        accuracy = metrics.get("accuracy")
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        f1 = metrics.get("f1")

        if any(v is None for v in [accuracy, precision, recall, f1]):
            continue

        results.append({
            "Model": nice_name,
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1 Score": float(f1),
        })

    if not results:
        st.error("No models could be evaluated. Train at least one model first.")
        if st.button("⬅ Back"):
            st.session_state.page = "bot_dashboard"
            st.rerun()
        return

    df = pd.DataFrame(results)

    # ------------------------------------------------
    # 3️⃣ Performance Overview
    # ------------------------------------------------
    st.markdown("### 📊 Performance Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Test Samples", test_samples if test_samples is not None else "-")
    with c2:
        st.metric("Training Samples", train_samples if train_samples is not None else "-")
    with c3:
        st.metric("Models Compared", len(df))

    # ------------------------------------------------
    # 4️⃣ Bar Chart
    # ------------------------------------------------
    st.markdown("### 📈 Performance Comparison")

    x = np.arange(len(df))
    width = 0.2

    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(x - 1.5 * width, df["Accuracy"], width, label="Accuracy")
    ax_bar.bar(x - 0.5 * width, df["Precision"], width, label="Precision")
    ax_bar.bar(x + 0.5 * width, df["Recall"], width, label="Recall")
    ax_bar.bar(x + 1.5 * width, df["F1 Score"], width, label="F1 Score")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(df["Model"])
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_ylabel("Scores")
    ax_bar.set_title("Model Performance Comparison")
    ax_bar.legend()

    st.pyplot(fig_bar)

    # ------------------------------------------------
    # 5️⃣ Pie Chart
    # ------------------------------------------------
    st.markdown("### 🥧 F1 Score Distribution")

    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(df["F1 Score"], labels=df["Model"], autopct="%1.3f")
    ax_pie.axis("equal")
    st.pyplot(fig_pie)

    # ------------------------------------------------
    # 6️⃣ Summary Table
    # ------------------------------------------------
    st.markdown("### 🏆 Performance Summary")

    st.dataframe(df.style.format({
        "Accuracy": "{:.3f}",
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1 Score": "{:.3f}",
    }))

    # Best model
    best_idx = df["F1 Score"].idxmax()
    best_row = df.loc[best_idx]

    st.success(
        f"🏆 Best Performing Model: *{best_row['Model']}* "
        f"(F1 Score: {best_row['F1 Score']:.3f})"
    )

    # ------------------------------------------------
    # 7️⃣ Downloads
    # ------------------------------------------------
    st.markdown("### ⬇ Download Results")

    st.download_button(
        "Download Comparison CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{bot_name}_model_comparison.csv",
        mime="text/csv",
    )

    buf_bar = io.BytesIO()
    fig_bar.savefig(buf_bar, format="png", bbox_inches="tight")
    buf_bar.seek(0)
    st.download_button(
        "Download Bar Chart (PNG)",
        data=buf_bar,
        file_name=f"{bot_name}_comparison_bar.png",
        mime="image/png",
    )

    buf_pie = io.BytesIO()
    fig_pie.savefig(buf_pie, format="png", bbox_inches="tight")
    buf_pie.seek(0)
    st.download_button(
        "Download Pie Chart (PNG)",
        data=buf_pie,
        file_name=f"{bot_name}_comparison_pie.png",
        mime="image/png",
    )

    # ------------------------------------------------
    # 8️⃣ Back Button
    # ------------------------------------------------
    if st.button("⬅ Back"):
        st.session_state.page = "bot_dashboard"
        st.rerun()
def active_learning_page():
    bot = st.session_state.selected_bot
    if not bot:
        st.error("No bot selected.")
        return

    st.title("🔍 Active Learning")

    # ✅ Use bot_id for API calls
    bot_id = bot["id"]

    model_type = st.selectbox("Select model", ["logreg", "spacy", "bert"])

    # Fetch data
    url = f"{BACKEND_URL}/active-learning/{bot_id}/{model_type}"
    res = requests.get(url, headers=api_headers())

    if res.status_code != 200:
        st.error("Failed to load active learning samples.")
        try:
            st.json(res.json())
        except Exception:
            st.write(res.text)
        return

    samples = res.json().get("samples", [])

    if not samples:
        st.success("🎉 No low-confidence samples! Model performing well.")
        if st.button("⬅ Back"):
            st.session_state.page = "bot_dashboard"
            st.rerun()
        return

    # Render each sample as a card
    st.markdown(
        """
        <style>
        .sample-card {
            padding: 20px;
            border-radius: 12px;
            background-color: #1e1e1e;
            border: 1px solid #333;
            margin-bottom: 25px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for idx, row in enumerate(samples):
        st.markdown("<div class='sample-card'>", unsafe_allow_html=True)

        st.markdown(f"### 📝 Review Sample {idx + 1}")

        st.write(f"*Sentence:* {row['text']}")
        st.write(f"*True Intent:* {row['true_intent']}")
        st.write(f"*Predicted Intent:* {row['predicted_intent']}")
        st.write(f"*Confidence:* {row['confidence']}%")

        corrected_intent = st.text_input(
            "Correct Intent",
            value=row.get("corrected_intent", row["predicted_intent"]),
            key=f"intent_{idx}",
        )

        corrected_entities = st.text_area(
            "Correct Entities (optional)",
            value=str(row.get("corrected_entities", row.get("entities", []))),
            key=f"entities_{idx}",
        )

        if st.button("💾 Save", key=f"save_{idx}"):
            payload = {
                "corrected_intent": corrected_intent,
                "corrected_entities": corrected_entities,
            }

            save_url = f"{BACKEND_URL}/active-learning/{bot_id}/{model_type}/save/{idx}"
            r = requests.post(save_url, json=payload, headers=api_headers())

            if r.status_code == 200:
                st.success("Saved correction!")
                st.rerun()
            else:
                st.error("Failed to save correction.")
                try:
                    st.json(r.json())
                except Exception:
                    st.write(r.text)

        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⬅ Back"):
        st.session_state.page = "bot_dashboard"
        st.rerun()
def feedback_page():
    bot = st.session_state.selected_bot
    if not bot:
        st.error("No bot selected.")
        return

    st.title("💬 Feedback")
    st.subheader(f"Bot: {bot['name']}")

    st.markdown(
        """
        Please share your feedback about this bot.  
        You can write about usability, performance, issues, or suggestions for improvement.
        """
    )

    feedback_text = st.text_area(
        "Write your feedback here",
        height=200,
        placeholder="Enter your feedback..."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("💾 Save Feedback"):
            if not feedback_text.strip():
                st.warning("Please write some feedback before saving.")
            else:
                res = submit_feedback_api(bot["id"], feedback_text)

                if res.status_code in [200, 201]:
                    st.success("✅ Feedback submitted successfully!")
                else:
                    st.error("❌ Failed to submit feedback.")

    with col2:
        if st.button("⬅ Back"):
            st.session_state.page = "workspace"
            st.rerun()


def admin_dashboard_page():
    render_admin_sidebar()

    # ---- Header ----
    st.markdown(
        "<h1 style='text-align:center;color:#4FC3F7;'>🛠 Admin Dashboard</h1>",
        unsafe_allow_html=True
    )
    st.write(
        f"<p style='text-align:center;color:#B3E5FC;'>Logged in as <b>{st.session_state.email}</b></p>",
        unsafe_allow_html=True
    )

    # ---- Fetch stats from backend ----
    stats = get_admin_stats_api()

    # Fix: remove admin from user count
    total_users = max(stats["total_users"] - 1, 0)

    # Fix: Your system has exactly 3 models
    total_models = 3

    st.markdown("### 📊 <span style='color:#4FC3F7;'>Overview</span>", unsafe_allow_html=True)

    # ---- Card CSS ----
    st.markdown("""
        <style>
        .card {
            padding: 22px;
            border-radius: 12px;
            background-color: #0A1A2F;
            border: 1px solid #4FC3F7;
            text-align: center;
            box-shadow: 0px 0px 15px rgba(79,195,247,0.3);
            transition: 0.3s;
            color: #E1F5FE;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0px 0px 20px rgba(79,195,247,0.6);
        }
        .icon {
            font-size: 40px;
            margin-bottom: -5px;
            color: #4FC3F7;
        }
        .value {
            font-size: 34px;
            font-weight: bold;
            margin-top: -5px;
            color: #81D4FA;
        }
        .label {
            font-size: 15px;
            margin-top: 5px;
            color: #B3E5FC;
        }
        </style>
    """, unsafe_allow_html=True)

    # ---- Cards Layout ----
    c1, c2, c3 = st.columns(3)
    c4, c5 = st.columns(2)

    with c1:
        st.markdown(f"""
            <div class="card">
                <div class="icon">👤</div>
                <div class="label">Total Users</div>
                <div class="value">{total_users}</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="card">
                <div class="icon">🗂</div>
                <div class="label">Total Workspaces</div>
                <div class="value">{stats['total_workspaces']}</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class="card">
                <div class="icon">📄</div>
                <div class="label">Total Datasets</div>
                <div class="value">{stats['total_datasets']}</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class="card">
                <div class="icon">🤖</div>
                <div class="label">Total Models</div>
                <div class="value">{total_models}</div>
            </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
            <div class="card">
                <div class="icon">📝</div>
                <div class="label">Total Annotations</div>
                <div class="value">{stats['total_annotations']}</div>
            </div>
        """, unsafe_allow_html=True)
def confirm_delete_page():
    user = st.session_state.get("delete_target")

    st.error(f"Are you sure you want to DELETE user: {user['email']} ?")

    col1, col2 = st.columns(2)
    if col1.button("Yes, Delete"):
        delete_user_api(user["id"])
        st.success("User deleted successfully!")
        st.session_state.page = "admin_panel"
        st.rerun()

    if col2.button("Cancel"):
        st.session_state.page = "admin_panel"
        st.rerun()
def confirm_delete_workspace_page():
    ws = st.session_state.get("delete_workspace_target")

    st.error(f"Are you sure you want to DELETE workspace: {ws['name']} ?")

    col1, col2 = st.columns(2)

    if col1.button("Yes, Delete"):
        delete_workspace_api(ws["id"])
        st.success("Workspace deleted successfully!")
        st.session_state.page = "admin_panel"
        st.rerun()

    if col2.button("Cancel"):
        st.session_state.page = "admin_panel"
        st.rerun()
def view_confusion_matrix_page():
    bot_name, model_type = st.session_state.get("view_cm", (None, None))

    if not bot_name:
        st.session_state.page = "admin_panel"
        st.rerun()

    st.markdown(f"<h3 style='color:#4FC3F7;'>📉 Confusion Matrix<br>{bot_name} – {model_type.upper()}</h3>", unsafe_allow_html=True)

    data = get_confusion_matrix_api(bot_name, model_type)

    if not data or not data.get("confusion_matrix"):
        st.error("Model not trained — no confusion matrix available.")
        if st.button("Back"):
            st.session_state.page = "admin_panel"
            st.rerun()
    else:
        cm = data["confusion_matrix"]
        labels = data.get("labels", [])

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        st.pyplot(fig)

        if st.button("Back"):
            st.session_state.page = "admin_panel"
            st.rerun()


def admin_panel_page():
    render_admin_sidebar()

    st.markdown("<h1 style='color:#4FC3F7;'>🛠 Admin Panel</h1>", unsafe_allow_html=True)

    # Top Tabs
    tabs = st.tabs([
    "Users Management",
    "Workspaces",
    "Datasets",
    "Models",
    "Activity Logs",
    "Feedback"
    ])


    # ----------------------------------------------------------
    # 1️⃣ USERS MANAGEMENT TAB
    # ----------------------------------------------------------
    with tabs[0]:
        st.markdown("<h3 style='color:#4FC3F7;'>Users Management</h3>", unsafe_allow_html=True)
        st.write("View registered users and remove unwanted accounts.")

        users = get_all_users_api()

        if not users:
            st.info("No users found.")
        else:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write("*Email*")
            col2.write("*Role*")
            col3.write("*Actions*")

            for user in users:
                c1, c2, c3 = st.columns([3, 1, 1])

                c1.write(user["email"])
                c2.write("Admin" if user["role"] == "admin" else "User")

                if c3.button("🗑 Delete", key=f"del_{user['id']}"):
                    st.session_state["delete_target"] = user
                    st.session_state.page = "confirm_delete"
                    st.rerun()

    # ----------------------------------------------------------
    # 2️⃣ WORKSPACES TAB
    # ----------------------------------------------------------
    # CSS for nowrap
    st.markdown("""
        <style>
        .no-wrap {
            white-space: nowrap;
        }
        </style>
    """, unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("<h3 style='color:#4FC3F7;'>Workspaces</h3>", unsafe_allow_html=True)
        st.write("View all created workspaces (bots) by each user.")

        bots = get_all_workspaces_api()

        if not bots:
            st.info("No workspaces found.")
        else:
            col1, col2, col3, col4, col5 = st.columns([3, 4, 1.2, 1.4, 1.4])

            col1.write("*Workspace Name*")
            col2.markdown("<span class='no-wrap'><b>Owner Email</b></span>", unsafe_allow_html=True)
            col3.markdown("<span class='no-wrap'><b>Datasets</b></span>", unsafe_allow_html=True)
            col4.markdown("<span class='no-wrap'><b>Annotations</b></span>", unsafe_allow_html=True)
            col5.markdown("<span class='no-wrap'><b>Actions</b></span>", unsafe_allow_html=True)

            for b in bots:
                c1, c2, c3, c4, c5 = st.columns([3, 4, 1.2, 1.4, 1.4])

                c1.write(b["name"])
                c2.markdown(f"<span class='no-wrap'>{b['owner_email']}</span>", unsafe_allow_html=True)
                c3.write(b["datasets"])
                c4.write(b["annotations"])

                if c5.button("🗑 Delete", key=f"ws_del_{b['id']}"):
                    st.session_state["delete_workspace_target"] = b
                    st.session_state.page = "confirm_delete_workspace"
                    st.rerun()

    # ----------------------------------------------------------
    # 3️⃣ DATASETS TAB
    # ----------------------------------------------------------
    with tabs[2]:
        st.markdown("<h3 style='color:#4FC3F7;'>Datasets</h3>", unsafe_allow_html=True)
        st.write("View and manage uploaded datasets.")

        datasets = get_all_datasets_api()

    # Prevent breaking of words
        st.markdown("""
            <style>
                .no-wrap {
                    white-space: nowrap !important;
                }
            </style>
        """, unsafe_allow_html=True)

        if not datasets:
            st.info("No datasets found.")
        else:
        # Better spacing between columns
            col1, col2, col3, col4, col5 = st.columns([3, 2.5, 3.5, 1.3, 1.3])

            col1.markdown("<span class='no-wrap'><b>Dataset Name</b></span>", unsafe_allow_html=True)
            col2.markdown("<span class='no-wrap'><b>Bot Name</b></span>", unsafe_allow_html=True)
            col3.markdown("<span class='no-wrap'><b>Owner Email</b></span>", unsafe_allow_html=True)
            col4.markdown("<span class='no-wrap'><b>Download</b></span>", unsafe_allow_html=True)
            col5.markdown("<span class='no-wrap'><b>Delete</b></span>", unsafe_allow_html=True)

        for d in datasets:
            c1, c2, c3, c4, c5 = st.columns([3, 2.5, 3.5, 1.3, 1.3])

            c1.markdown(f"<span class='no-wrap'>{d['name']}</span>", unsafe_allow_html=True)
            c2.markdown(f"<span class='no-wrap'>{d['bot_name']}</span>", unsafe_allow_html=True)
            c3.markdown(f"<span class='no-wrap'>{d['owner_email']}</span>", unsafe_allow_html=True)

            if c4.button("⬇", key=f"dl_{d['id']}"):
                st.session_state["download_target"] = d["id"]
                st.session_state.page = "download_dataset"
                st.rerun()

            if c5.button("🗑", key=f"del_ds_{d['id']}"):
                st.session_state["delete_dataset_target"] = d
                st.session_state.page = "confirm_delete_dataset"
                st.rerun()



    # ----------------------------------------------------------
    # 4️⃣ MODELS TAB
    # ----------------------------------------------------------
    with tabs[3]:

        st.markdown("<h3 style='color:#4FC3F7;'>Models</h3>", unsafe_allow_html=True)
        st.write("View performance of all models for every workspace.")

        models_data = get_all_models_api()

        if not models_data:
            st.info("No model information available.")
        else:
            col1, col2, col3, col4, col5, col6, col7 = st.columns([2.5, 2, 1.2, 1.2, 1.2, 1.2, 1])

            col1.write("*Bot Name*")
            col2.write("*Model*")
            col3.write("*Accuracy*")
            col4.write("*F1 Score*")
            col5.write("*Precision*")
            col6.write("*Recall*")
            col7.write("*View*")

        for m in models_data:
            c1, c2, c3, c4, c5, c6, c7 = st.columns([2.5, 2, 1.2, 1.2, 1.2, 1.2, 1])

            c1.write(m["bot_name"])
            c2.write(m["model_type"].upper())

            # If trained => show metrics
            if m["trained"]:
                c3.write(round(m["accuracy"], 3))
                c4.write(round(m["f1"], 3))
                c5.write(round(m["precision"], 3))
                c6.write(round(m["recall"], 3))

                if c7.button("🔍", key=f"view_{m['bot_name']}_{m['model_type']}"):
                    st.session_state["view_cm"] = (m["bot_name"], m["model_type"])
                    st.session_state.page = "view_confusion_matrix"
                    st.rerun()

            else:
                c3.write("–")
                c4.write("–")
                c5.write("–")
                c6.write("–")
                c7.write("❌")

    # ----------------------------------------------------------
    # 5️⃣ ACTIVITY LOGS TAB
    # ----------------------------------------------------------
    with tabs[4]:
        st.markdown("<h3 style='color:#4FC3F7;'>Activity Logs</h3>", unsafe_allow_html=True)
        st.write("View system activities performed by users and admin.")

        # Prevent email wrapping
        st.markdown("""
            <style>
                .no-wrap {
                    white-space: nowrap !important;
                }
            </style>
        """, unsafe_allow_html=True)

        logs = get_activity_logs_api()

        if not logs:
            st.info("No activity logs found.")
        else:
            col1, col2, col3 = st.columns([2, 2, 5])
            col1.write("*Timestamp*")
            col2.write("*User*")
            col3.write("*Action*")

        for log in logs:
            c1, c2, c3 = st.columns([2, 2, 5])

            c1.write(log["timestamp"])

            # Email stays in one single line
            c2.markdown(f"<span class='no-wrap'>{log['user_email']}</span>", unsafe_allow_html=True)

            c3.write(log["action"])
    # ----------------------------------------------------------
# 6️⃣ FEEDBACK TAB
# ----------------------------------------------------------
    with tabs[5]:
        st.markdown("<h3 style='color:#4FC3F7;'>💬 User Feedback</h3>", unsafe_allow_html=True)
        st.write("View feedback submitted by users for their bots.")

        feedbacks = get_all_feedbacks_api()

        if not feedbacks:
            st.info("No feedback submitted yet.")
        else:
        # Prevent text wrapping issues
            st.markdown("""
                <style>
                    .no-wrap {
                        white-space: nowrap !important;
                    }
                </style>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns([4, 2, 5, 2])

            col1.write("*User Email*")
            col2.write("*Bot Name*")
            col3.write("*Feedback*")
            col4.write("*Timestamp*")

        for fb in feedbacks:
            c1, c2, c3, c4 = st.columns([4, 2, 5, 2])

            c1.markdown(f"<span class='no-wrap'>{fb['user_email']}</span>", unsafe_allow_html=True)
            c2.write(fb["bot_name"])
            c3.write(fb["feedback_text"])
            c4.write(fb["timestamp"])



# ----------------------------------------------------------
#  HANDLER: Confirm Delete User
# ----------------------------------------------------------
def confirm_delete_page():
    user = st.session_state.get("delete_target")

    if not user:
        st.session_state.page = "admin_panel"
        st.rerun()

    st.error(f"Are you sure you want to DELETE user: {user['email']} ?")

    c1, c2 = st.columns(2)
    if c1.button("Yes, Delete"):
        delete_user_api(user["id"])
        st.success("User deleted successfully!")
        st.session_state.page = "admin_panel"
        st.rerun()

    if c2.button("Cancel"):
        st.session_state.page = "admin_panel"
        st.rerun()


# ----------------------------------------------------------
#  HANDLER: Confirm Delete Workspace
# ----------------------------------------------------------
def confirm_delete_workspace_page():
    ws = st.session_state.get("delete_workspace_target")

    if not ws:
        st.session_state.page = "admin_panel"
        st.rerun()

    st.error(f"Delete workspace '{ws['name']}' owned by {ws['owner_email']}?")

    c1, c2 = st.columns(2)
    if c1.button("Yes, Delete Workspace"):
        delete_workspace_api(ws["id"])
        st.success("Workspace deleted successfully!")
        st.session_state.page = "admin_panel"
        st.rerun()

    if c2.button("Cancel"):
        st.session_state.page = "admin_panel"
        st.rerun()


# ----------------------------------------------------------
#  HANDLER: Confirm Delete Dataset
# ----------------------------------------------------------
def confirm_delete_dataset_page():
    ds = st.session_state.get("delete_dataset_target")

    if not ds:
        st.session_state.page = "admin_panel"
        st.rerun()

    st.error(f"Are you sure you want to DELETE dataset: {ds['name']} ?")

    c1, c2 = st.columns(2)
    if c1.button("Yes, Delete Dataset"):
        delete_dataset_api(ds["id"])
        st.success("Dataset deleted successfully!")
        st.session_state.page = "admin_panel"
        st.rerun()

    if c2.button("Cancel"):
        st.session_state.page = "admin_panel"
        st.rerun()


# ----------------------------------------------------------
#  HANDLER: Download Dataset
# ----------------------------------------------------------
def download_dataset_page():
    ds_id = st.session_state.get("download_target")

    if not ds_id:
        st.session_state.page = "admin_panel"
        st.rerun()

    st.markdown("<h3 style='color:#4FC3F7;'>⬇ Download Dataset</h3>", unsafe_allow_html=True)

    file_bytes = download_dataset_admin_api(ds_id)

    if file_bytes is None:
        st.error("Failed to download dataset.")
        if st.button("Back"):
            st.session_state.page = "admin_panel"
            st.rerun()
        return

    st.success("Click the button below to download the dataset.")

    st.download_button(
        "⬇ Download File",
        data=file_bytes,
        file_name="dataset.csv",
        mime="text/csv",
    )

    if st.button("Back"):
        st.session_state.page = "admin_panel"
        st.rerun()



# ---------------------------
# Router / Page switch
# ---------------------------
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "register":
    register_page()
elif st.session_state.page == "workspace":
    workspace_page()
elif st.session_state.page == "bot_dashboard":
    bot_dashboard()
elif st.session_state.page == "dataset_page":
    dataset_page()
elif st.session_state.page == "train_page":
    train_page()
elif st.session_state.page == "model_train_page":
    model_train_page()
elif st.session_state.page == "model_compare_page":
    model_compare_page()
elif st.session_state.page == "active_learning_page":
    active_learning_page()
elif st.session_state.page == "feedback_page":
    feedback_page()

elif st.session_state.page == "admin_dashboard":
    admin_dashboard_page()
elif st.session_state.page == "confirm_delete":
    confirm_delete_page()
elif st.session_state.page == "confirm_delete_workspace":
    confirm_delete_workspace_page()
elif st.session_state.page == "confirm_delete_dataset":
    confirm_delete_dataset_page()

elif st.session_state.page == "download_dataset":
    download_dataset_page()
elif st.session_state.page == "view_confusion_matrix":
    view_confusion_matrix_page()


elif st.session_state.page == "admin_panel":
    admin_panel_page()





else:
    st.write("Unknown page. Resetting.")
    st.session_state.page = "login"
    st.rerun()