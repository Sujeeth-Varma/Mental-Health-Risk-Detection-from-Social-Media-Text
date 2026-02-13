# An Explainable AI Framework for Multi-Level Mental Health Risk Detection from Social Media Text

## Overview

This project is a scalable, non-intrusive, and explainable AI system that analyzes social media text and classifies mental health risk into multiple levels (Low, Medium, High). It integrates NLP, supervised machine learning, and Explainable AI (XAI) techniques.

## Tech Stack

### Backend

- Python 3.10+
- Flask + Flask-CORS
- NLTK / spaCy
- Scikit-learn
- SHAP / LIME
- VADER / TextBlob / Gensim

### Frontend

- React.js 18
- Axios
- Tailwind CSS
- Recharts

## Project Structure

```
text-based-mental-health-detector/
├── Dockerfile                    # Multi-stage Docker build
├── nginx.conf                    # Nginx reverse-proxy config
├── supervisord.conf              # Process manager config
├── backend/
│   ├── app.py                    # Flask application entry point
│   ├── config.py                 # Configuration settings
│   ├── requirements.txt          # Python dependencies
│   ├── models/                   # Saved ML models & vectorizers
│   ├── data/                     # Datasets
│   ├── pipeline/
│   │   ├── preprocessor.py       # Text cleaning & preprocessing
│   │   ├── feature_extractor.py  # Feature extraction (sentiment, emotion, n-grams)
│   │   ├── model_trainer.py      # Model training & evaluation
│   │   ├── explainer.py          # SHAP/LIME explainability
│   │   └── topic_modeler.py      # LDA/NMF topic modeling
│   ├── utils/
│   │   ├── dataset_generator.py  # Synthetic dataset generation
│   │   └── helpers.py            # Utility functions
│   └── train.py                  # Training script
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.js
│   │   ├── index.css
│   │   ├── components/
│   │   │   ├── TextInput.jsx
│   │   │   ├── PredictionResult.jsx
│   │   │   ├── ExplainabilityView.jsx
│   │   │   ├── SentimentGauge.jsx
│   │   │   ├── FeatureChart.jsx
│   │   │   └── Header.jsx
│   │   └── services/
│   │       └── api.js
│   ├── package.json
│   └── tailwind.config.js
└── README.md
```

## Setup & Installation

### Docker (Recommended)

The easiest way to run the full stack is with Docker. A single container serves both the React frontend and the Flask backend with pre-trained models.

```bash
# Build the image
docker build -t mental-health-detector .

# Run the container
docker run -p 80:80 mental-health-detector
```

Open **http://localhost** — the frontend and backend are both ready to use.

> **Note:** The Docker image bundles the trained models from `backend/models/`. If you retrain, rebuild the image to pick up the new models.

### Manual Setup

#### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger')"
python train.py
python app.py
```

Backend runs on `http://localhost:5000`

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:3000`

## API Endpoints

| Endpoint   | Method | Description                        |
| ---------- | ------ | ---------------------------------- |
| `/predict` | POST   | Returns risk level and explanation |
| `/analyze` | POST   | Returns sentiment, emotion, topics |
| `/health`  | GET    | System status                      |

## Risk Levels

| Level  | Color  | Description                                    |
| ------ | ------ | ---------------------------------------------- |
| Low    | Green  | Neutral or mildly emotional language           |
| Medium | Yellow | Persistent negative emotions, distress signals |
| High   | Red    | Strong negative sentiment, crisis indicators   |

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Explainability quality (qualitative)
