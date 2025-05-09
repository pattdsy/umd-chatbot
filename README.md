

 
# UMD Buddy Chatbot

A BERT-powered chatbot built using Python (TensorFlow, Transformers, Pandas, NumPy) to enhance employee understanding and management of **User Managed Data (UMD)** — helping users identify, classify, and register files under enterprise governance policies.

---

## Project Summary

In large organizations, files created outside formal IT-managed systems (like Excel, Access, or manually updated reports) are considered **User Managed Data**. These files require registration and governance under official bank policy.

**UMD Buddy (Databot)** is an AI-powered assistant that:

- Classifies queries using a fine-tuned BERT model
- Handles multi-turn conversations using context tracking
- Explains governance standards (AI/ML, UMD, Customer Lifecycle)
- Autocorrects typos with TextBlob
- Falls back to keyword detection when model confidence is low

---

## Key Features

| Feature              | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| Intent Classification | Fine-tuned `bert-base-uncased` using `TFBertForSequenceClassification`     |
| Contextual Dialogue   | Tracks multi-turn logic for UMD status checks                              |
| Fallback Handling     | Recognizes key phrases (e.g., "CDP", "Dataiku") even with low model confidence |
| Autocorrection        | Corrects spelling using `TextBlob` before prediction                       |
| Chat Knowledge Base   | Responses stored in structured `responses.json`                            |

---

## Tech Stack

- Python (Flask, JSON, re, TextBlob)
- TensorFlow + Hugging Face Transformers
- BERT: `bert-base-uncased`
- scikit-learn (Label Encoding, Evaluation)
- Pandas, NumPy

---

## Model Training Summary

- **Training Data**: `intents.csv`
- **Label Encoding**: Convert textual tags to numeric `label_id`
- **Tokenizer**: `bert-base-uncased` (Hugging Face)
- **Model**: Fine-tuned BERT with softmax output
- **Validation**: 80/20 split, 10 epochs
- **Artifacts Saved**:
  - `intent_model/` *(BERT model folder)*  
  - `tokenizer/` *(Tokenizer folder)*  
  - `label_encoder_classes.json` *(Encoded label classes)*

> Due to GitHub file size limits, the model files are uploaded as ZIPs to Google Drive.

---

## 💻 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/pattdsy/umd-chatbot.git
cd umd-chatbot
```

### 2. Set up a virtual environment

**Mac/Linux:**

```bash
python3 -m venv chatbot_env
source chatbot_env/bin/activate
```

**Windows:**

```bash
python -m venv chatbot_env
chatbot_env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download and extract model folders

Download from Google Drive:  
https://drive.google.com/drive/u/0/folders/1HgtySCBgFahqvRQnoTZqPyqZQKT-VeEO

Files needed:
- `intent_model.zip`
- `tokenizer.zip`

### 5. Extract both into the project root so it looks like this:

```
umd-chatbot/
├── app.py
├── umd_bot.py
├── intents.csv
├── responses.json
├── intent_model/
├── tokenizer/
├── requirements.txt
```

### 6. Run the chatbot

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

## Sample Intents Covered

| User Query                             | Intent                      |
|----------------------------------------|-----------------------------|
| What is a UMD?                         | `umd_guidelines`            |
| Is this file a UMD?                    | `umd_check_start`           |
| How do I register my file?             | `umd_registration`          |
| Who owns the UMD?                      | `umd_ownership`             |
| What is the CDP?                       | `fallback_keywords`         |
| What is Dataiku?                       | `dataiku_guidelines`        |
| Dataiku model lifecycle?               | `dataiku_model_lifecycle`   |
| What are Dataiku assets?               | `dataiku_asset_definitions` |
| Define active/inactive customer        | `customer_status_framework` |
| What is AI/ML governance?              | `ai_framework`              |
| Data Asset Prioritization Guidelines?  | `data_asset_prioritisation` |
| Column/table/data type standards?      | `hardcoded keyword response`|
| What does UMD status mean?             | `umd_status_explained`      |
| I don’t understand / Help              | `chatbot_help`              |

---

## Example Response Flow

```text
User: "Is my file a UMD?"  
Bot: "Let’s start. Was this file created outside official IT systems, like Excel, Access, or through manual processes?"  
User: "Yes, it's from Excel."  
Bot: "Got it. Was the file enriched or modified after creation? (yes/no)"
```

---

## Internal Deployment Plan

This bot can be registered with **Microsoft Teams** using the **Microsoft Bot Framework**, and integrated into internal systems via a secure API gateway.

The model and logic can be hosted on enterprise infrastructure (e.g., **Azure** or **AWS**) to provide real-time UMD guidance at scale.

**Another possibility is Powerapps integration with the UMD Registration portal.

This chatbot Flask app can also be hosted on Render, Azure App Service, or a public server like Railway or Heroku. This involves creating a Power Automate flow that calls the endpoint and connecting that flow to Power Apps.**

---

## Future Improvements

- Replace `responses.json` with a dynamic knowledge base (e.g., SharePoint or REST API)
- Add chat history tracking and user analytics
- Deploy on Microsoft Teams or Slack using webhook integration
- Implement multilingual support (e.g., Google Translate API)
