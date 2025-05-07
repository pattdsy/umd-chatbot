import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
from textblob import TextBlob
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

#  TRAINING CONTROL
RETRAIN = False

#  Load responses
with open("responses.json", "r") as f:
    responses_dict = json.load(f)

#  TRAINING BLOCK
if RETRAIN:
    # Load training data
    df = pd.read_csv("intents.csv")

    label_encoder = LabelEncoder()
    df['label_id'] = label_encoder.fit_transform(df["label"])
    num_labels = len(label_encoder.classes_)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"], df["label_id"],
        test_size=0.2, random_state=42, stratify=df["label_id"]
    )

    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, return_tensors="tf")
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, return_tensors="tf")

    train_dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"]},
        train_labels
    )).batch(8)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        {"input_ids": val_encodings["input_ids"], "attention_mask": val_encodings["attention_mask"]},
        val_labels
    )).batch(8)

    model = TFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(label_encoder.classes_)
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    model.fit(train_dataset, validation_data=val_dataset, epochs=10)

    model.save_pretrained("intent_model")
    tokenizer.save_pretrained("tokenizer/")
    
    with open("label_encoder_classes.json", "w") as f:
        json.dump(label_encoder.classes_.tolist(), f)

else:
    tokenizer = BertTokenizer.from_pretrained("tokenizer")
    model = TFBertForSequenceClassification.from_pretrained("intent_model")

    with open("label_encoder_classes.json", "r") as f:
        label_classes = json.load(f)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_classes)

#  Predict intent
def predict_intent(user_input):
    encoded = tokenizer([user_input], padding=True, truncation=True, return_tensors="tf")
    output = model(encoded)
    logits = output.logits
    probs = tf.nn.softmax(logits, axis=1)
    pred_id = tf.argmax(probs, axis=1).numpy()[0]
    confidence = probs[0][pred_id].numpy()
    predicted_label = label_encoder.inverse_transform([pred_id])[0]
    return predicted_label, confidence

def autocorrect_input(text):
    # skip terms like "UMD"
    safe_text = re.sub(r'\bumd\b', 'UMD', text.lower())  # preserve key terms
    blob = TextBlob(safe_text)
    corrected = blob.correct()
    return str(corrected)

# Basic intent response retrieval
def get_basic_intent_response(intent):
    return random.choice(responses_dict.get(intent, ["I'm not sure how to answer that yet."]))

#  Get response
def get_response(user_input, session):
    user_input_corrected = autocorrect_input(user_input)
    user_input_lower = user_input_corrected.lower().strip()
    
    # Direct greetings
    if user_input_lower in ["hi", "hello", "kamusta", "hey", "kamusta po"]:
        return random.choice(responses_dict["greeting"])

    # UMD-specific phrase matching
    umd_definition_phrases = [
        "what is a umd", "define umd", "explain umd", "what counts as a umd", "umd meaning", "what's a umd"
    ]
    umd_check_phrases = [
        "is this a umd", "is my file a umd", "is my sheet a umd", "is it a umd", "does this count as a umd"
    ]
    if any(phrase in user_input_lower for phrase in umd_definition_phrases):
        return get_basic_intent_response("umd_guidelines")
    if any(phrase in user_input_lower for phrase in umd_check_phrases):
        session["context"] = "umd_checking"
        return get_basic_intent_response("umd_check_start")

    #  Hardcoded keyword-based responses
    keyword_responses = {
        ("table naming", "table standards", "table names", "data tables"): (
            "The table naming standards used in database design and management ensure a consistent and clear structure for organizing and accessing internal databases.\n"
            "You can access the standards by navigating to:\n\n"
            "DG - Data Policy > Policies and Frameworks > Cloud Data Platform Table Naming Standards"
        ),
        ("column naming standard", "column names", "columns", "column naming", "data columns"): (
            "Naming standards play a crucial role in database design and management. These standards ensure uniform naming conventions for all database objects, enabling clear communication of each object's role and functionality.\n"
            "You can access the standards by navigating to:\n\n"
            "DG Data Policy > Policies and Frameworks > Column Naming Standards"
        ),
        ("data type standards", "data types"): (
            "Data type standards are essential in database design and management. The Cloud Data Platform Data Type Standards document outlines best practices for selecting appropriate data types—such as numerical, textual, date/time, and binary.\n"
            "You can access the standards by navigating to:\n\n"
            "DG Policy > Policies and Frameworks > Cloud Data Platform Data Type Standards"
        ),
    }
    for keywords, reply in keyword_responses.items():
        if any(keyword in user_input_lower for keyword in keywords):
            return reply

    # Contextual Checks
    if session.get("context") == "umd_checking":
        if any(negation in user_input_lower for negation in ["not manual", "not manually", "wasn't manual", "was not manual", "no", "not edited", "not modified"]):
            session["context"] = None
            return "Since it was not manually created and was produced by a process that underwent SDLC, it likely does not qualify as a UMD. This typically refers to files generated by automated ETL jobs, system exports, or scheduled database queries built and maintained through formal IT development procedures. Please confirm with Data Governance."
        elif any(word in user_input_lower for word in ["manual", "excel", "access", "created", "consolidated"]):
            session["context"] = "umd_check_transformed"
            return "Got it. Was the file enriched or modified after creation? (yes/no)"
        elif any(word in user_input_lower for word in ["enrich", "edit", "append", "override", "modified", "added", "changed", "yes", "y", "correct", "it was"]):
            session["context"] = "awaiting_registration"
            return "Since it was enriched or edited, it qualifies as a UMD. Would you like the registration link?"
        elif any(word in user_input_lower for word in ["raw", "extraction", "no modification", "untouched", "unedited", "unmodified"]):
            session["context"] = None
            return "Since it was extracted raw and untouched, it does NOT qualify as a UMD. No further action needed."
        else:
            return "Please specify clearly if it was manually created, enriched, extracted raw, or answer yes/no."

    elif session.get("context") == "umd_check_transformed":
        if any(negation in user_input_lower for negation in ["no", "not modified", "no changes", "not enriched", "wasn't enriched", "no modification"]):
            session["context"] = None
            return "Since it wasn’t modified after creation, it likely doesn't qualify as a UMD. Review the policy to confirm."
        elif any(word in user_input_lower for word in ["yes", "modified", "enriched", "consolidated", "changed", "edited", "updated"]):
            session["context"] = "awaiting_registration"
            return "Thanks! This file qualifies as a UMD. Would you like the registration link?"
        else:
            return "Please clearly state if the file was enriched or modified after creation. (yes/no)"

    elif session.get("context") == "awaiting_registration":
        if any(word in user_input_lower for word in ["yes", "ok", "sure", "please", "link", "registration", "register"]):
            session["context"] = None
            return get_basic_intent_response("umd_registration")
        elif "no" in user_input_lower:
            session["context"] = None
            return "Okay, let me know if you need anything else about UMDs."
        else:
            return "Would you like the registration link? Please answer clearly (yes/no)."

    # Fallback for low confidence predictions
    intent, confidence = predict_intent(user_input)
    if confidence < 0.45:
        if any(word in user_input_lower for word in ["excel", "access", "manual", "consolidated"]):
            session["context"] = "umd_check_transformed"
            return "Got it. Was the file enriched or modified after creation? (yes/no)"
        fallback_keywords = {
            "dataiku": "Dataiku is our cloud-based platform for ML workflows. For governance, refer to the Dataiku Governance Process under DG Policy.",
            "what is dataiku": "Dataiku is our cloud-based platform for ML workflows. For governance, refer to the Dataiku Governance Process under DG Policy.",
            "data asset": "Data assets are prioritized using the EDO 5-tier classification. Ask me about the Data Asset Prioritisation Guidelines.",
            "classification": "We use sensitivity classification standards managed by EDO. Ask about 'EDO classification' or 'Data Governance Standards'.",
            "cdp": (
                "The Cloud Data Platform (CDP) is the central environment where we store, process, and govern critical enterprise data. "
                "It supports analytics, model deployment, and business workflows with full compliance to EDO governance standards."
            ),
            "active customer": ("The Customer Status Framework standardizes customer lifecycle definitions (onboarding to offboarding).\n\nIt covers:\n\n- 'Active', 'Inactive', and 'Deactivated' customer statuses\n- Open/Closed account status definitions\n\nThis supports 360° customer analytics and unified reporting across systems.\n\n For more definitions and rules, please navigate to DG Data Policy > Policies and Frameworks > Customer Data Standards."
            ),
            "inactive customer": ("The Customer Status Framework standardizes customer lifecycle definitions (onboarding to offboarding).\n\nIt covers:\n\n- 'Active', 'Inactive', and 'Deactivated' customer statuses\n- Open/Closed account status definitions\n\nThis supports 360° customer analytics and unified reporting across systems.\n\n For more definitions and rules, please navigate to DG Data Policy > Policies and Frameworks > Customer Data Standards."
            ), 
            "closed account": ("The Customer Status Framework standardizes customer lifecycle definitions (onboarding to offboarding).\n\nIt covers:\n\n- 'Active', 'Inactive', and 'Deactivated' customer statuses\n- Open/Closed account status definitions\n\nThis supports 360° customer analytics and unified reporting across systems.\n\n For more definitions and rules, please navigate to DG Data Policy > Policies and Frameworks > Customer Data Standards."
            ),
            "deactivated": ("The Customer Status Framework standardizes customer lifecycle definitions (onboarding to offboarding).\n\nIt covers:\n\n- 'Active', 'Inactive', and 'Deactivated' customer statuses\n- Open/Closed account status definitions\n\nThis supports 360° customer analytics and unified reporting across systems.\n\n For more definitions and rules, please navigate to DG Data Policy > Policies and Frameworks > Customer Data Standards."
            ),
            "cloud data platform": (
                "The Cloud Data Platform (CDP) is our secure, scalable data ecosystem for managing enterprise datasets. "
                "It’s governed under EDO's Data Governance Framework and includes policies for naming standards, data retention, and access management."
            )}
        for keyword, reply in fallback_keywords.items():
            if keyword in user_input_lower:
                return reply
        return get_basic_intent_response("chatbot_help")

    # High confidence predictions — handle intent
    if intent == "umd_check_start":
        session["context"] = "umd_checking"
        return get_basic_intent_response(intent)
    if intent == "umd_check_created":
        session["context"] = "umd_check_transformed"
        return "Okay, did you manually consolidate, transform, or enrich the data? (yes/no)"
    if intent == "umd_check_transformed":
        session["context"] = "awaiting_registration"
        return get_basic_intent_response(intent)
    if intent == "umd_check_not_transformed":
        session["context"] = None
        return get_basic_intent_response(intent)

    return get_basic_intent_response(intent)
