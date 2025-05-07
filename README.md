# umd-chatbot
User-Managed Data (UMD) Governance Chatbot using Python (TensorFlow, NLP with BERT, Pandas, and NumPy) to enhance employee navigation of data governance frameworks, umd identification and registration help.

A BERT-powered chatbot designed to help employees identify, classify, and register User Managed Data (UMD) within enterprise environments. Built to support data governance awareness and policy compliance through natural language interaction. 

Project Summary 

In large organizations, files created outside formal IT systems (e.g., Excel, Access, manually updated reports) are often considered User Managed Data. These UMDs must be registered and governed under bank policy. 

UMD Buddy (Databot)  is an AI chatbot that: 

Classifies user queries using BERT (fine-tuned on enterprise intents) 
Handles follow-up logic using context tracking 
Offers UMD definitions, AI/ML policy explanations, and customer lifecycle standards 
Auto-corrects user inputs with TextBlob 
Falls back to keyword detection when model confidence is low 
 

🤖 Key Features 🤖 
Feature 
Description 
Intent classification 
Fine-tuned bert-base-uncased model using TFBertForSequenceClassification 
Contextual dialogue 
Tracks multi-turn flows for UMD checks (manual creation → enrichment → registration) 
Fallback logic 
Recognizes keywords (e.g., "CDP", "Dataiku") even with low model confidence 
Autocorrection 
Typos are corrected before prediction using TextBlob 
Chatbot knowledge base 
Stores structured answers in responses.json 


Tech Stack 
Python (Flask, JSON, re, TextBlob) 
TensorFlow (TFBertForSequenceClassification) 
Hugging Face Transformers 
Scikit-learn for label encoding and model evaluation 
Model Training Summary 

Input: intents.csv file with labeled chatbot training phrases 

Label encoding: Convert labels to numeric label_id 
Tokenizer: bert-base-uncased from Hugging Face 

Model: Fine-tuned BERT with softmax output for classification 
Evaluation: 80/20 train-validation split, 10 epochs 

Saved Artifacts: 
intent_model/ (BERT model) *
tokenizer/ (BERT tokenizer) *
label_encoder_classes.json (intent classes) *
* Note that these artifacts are separate zip files uploaded into Google Drive due to Github size limits.


How to run locally
How to Run Locally
Clone the repository
git clone https://github.com/pattdsy/umd-chatbot.git
cd umd-chatbot
Set up a virtual environment (recommended)
python3 -m venv chatbot_env
source chatbot_env/bin/activate  # Mac/Linux
# For Windows:
# chatbot_env\Scripts\activate
Install dependencies
pip install -r requirements.txt
Download and extract the model files
Download these two ZIP files from Google Drive:

intent_model.zip
tokenizer.zip
After downloading, extract them into the project root so the folder structure looks like:

umd-chatbot/
├── app.py
├── umd_bot.py
├── intent_model/
├── tokenizer/
Run the chatbot
python app.py
Then open your browser and go to:
👉 http://127.0.0.1:5000


Sample Intents Covered 

“What is a UMD?” → umd_guidelines 
“Is this file a UMD?” / “Does this count as a UMD?” → umd_check_start 
“How do I register my file?” → umd_registration 
“Who owns the UMD?” → umd_ownership 
“What is the CDP?” / “Cloud Data Platform?” → fallback_keywords → CDP explanation 
“What is Dataiku?” → dataiku_guidelines 
“Dataiku model lifecycle” → dataiku_model_lifecycle 
“What are Dataiku assets?” → dataiku_asset_definitions 
“Define active customers” / “Closed account?” / “Inactive customer?” → customer_status_framework 
“What is AI Framework?” / “ML governance policy?” → ai_framework 
“What are the Data Asset Prioritisation Guidelines?” → data_asset_prioritisation 
“What are column naming standards?” / “Column names?” → hardcoded keyword response 
“What are table naming standards?” / “Table names?” → hardcoded keyword response 
“What are data type standards?” → hardcoded keyword response 
“What does UMD status mean?” → umd_status_explained 
“Help” / “What can you do?” / “I don’t understand” → chatbot_help fallback 


How to Run Locally (only for SBC Enterprise Data Office) 
 

Example Response Flow 
User: "Is my file a UMD?"  

Bot: "Let’s start. Was this file created outside official IT systems, like Excel, Access, or through manual processes?"  

User: "Yes, it's from Excel."  

Bot: "Got it. Was the file enriched or modified after creation? (yes/no)" 


This project may be registered as a bot in Microsoft Teams, integrated through the Microsoft Bot Framework, and connected to internal users via a secure API gateway. This will allow employees to interact with UMD Buddy directly in their enterprise chat environment. The intent classification model, session logic, and governance content will be hosted on an internal server or cloud platform (e.g., Azure or AWS), enabling scalable, real-time guidance for UMD identification and registration.  

Future Improvements 

Replace responses.json with external knowledge base integration (e.g., SharePoint links, REST APIs) 
Add chat history tracking and user analytics 
Deploy on Microsoft Teams or Slack via webhook 
Implement multilingual support via translation APIs 
 

Acknowledgments 

Developed as part of a Data Science and Governance internship to promote responsible data governance. 

Special thanks to the Data Governance team for support on policy integration and domain knowledge. 

 

Related Projects 

See also: Data validator for data quality assessment 
