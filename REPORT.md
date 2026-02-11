# Project Report

## An Explainable AI Framework for Multi-Level Mental Health Risk Detection from Social Media Text

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [System Architecture](#4-system-architecture)
5. [Technology Stack](#5-technology-stack)
6. [Data Description](#6-data-description)
7. [NLP Pipeline & Text Preprocessing](#7-nlp-pipeline--text-preprocessing)
8. [Feature Engineering](#8-feature-engineering)
9. [Machine Learning Algorithms](#9-machine-learning-algorithms)
10. [Explainable AI (XAI)](#10-explainable-ai-xai)
11. [Topic Modeling](#11-topic-modeling)
12. [Backend Architecture (Flask REST API)](#12-backend-architecture-flask-rest-api)
13. [Frontend Architecture (React + Vite)](#13-frontend-architecture-react--vite)
14. [Model Evaluation Results](#14-model-evaluation-results)
15. [Ethical Considerations](#15-ethical-considerations)
16. [Limitations & Future Work](#16-limitations--future-work)
17. [Conclusion](#17-conclusion)
18. [References](#18-references)

---

## 1. Introduction

Mental health disorders such as depression, anxiety, and suicidal ideation are increasing globally. The World Health Organization (WHO) estimates that one in four people will be affected by a mental or neurological disorder at some point in their lives. Early detection remains a critical challenge because traditional clinical assessments rely on intrusive, time-consuming, and often inaccessible diagnostic procedures.

Social media platforms have emerged as a rich, organic source of textual data where users frequently express emotions, thoughts, and psychological distress. This project leverages that natural language data to build a **scalable, non-intrusive, and explainable AI system** capable of classifying mental health risk from social media text into three levels: **Low**, **Medium**, and **High**.

The system combines **Natural Language Processing (NLP)**, **supervised machine learning classification**, and **Explainable AI (XAI)** techniques to provide transparent, interpretable predictions — moving beyond black-box models toward a decision-support tool that clinicians and researchers can trust and understand.

---

## 2. Problem Statement

Despite the growing volume of mental health expressions on social media, there is a lack of accessible, automated, and transparent tools that can:

- Detect varying levels of psychological distress from unstructured text.
- Provide human-understandable explanations for each risk prediction.
- Operate without storing personal data or identifying users.

This project addresses that gap by developing a full-stack web application that accepts social media text as input and returns a multi-level risk classification along with explainable linguistic features.

---

## 3. Objectives

1. **Detect mental health risk levels** from social media text using NLP and machine learning.
2. **Classify risk into three tiers** — Low, Medium, and High — to support graded intervention.
3. **Provide explainable outputs** showing which words, sentiments, emotions, and topics drove the prediction.
4. **Support early identification** of psychological distress signals before clinical escalation.
5. **Build an ethical, non-intrusive decision-support tool** with no data retention.
6. **Deliver a production-ready full-stack web application** with a Flask backend and React frontend.

---

## 4. System Architecture

### 4.1 High-Level Architecture Diagram

```
┌──────────────────────────────────┐
│        React Frontend            │
│   (Vite + Tailwind CSS v4)       │
│                                  │
│  ┌────────┐ ┌──────────────────┐ │
│  │TextInput│ │PredictionResult  │ │
│  └────┬───┘ │ExplainabilityView│ │
│       │     │SentimentGauge    │ │
│       │     │FeatureChart      │ │
│       │     └──────────────────┘ │
└───────┼──────────────────────────┘
        │  REST API (Axios)
        │  POST /predict
        │  POST /analyze
        │  GET  /health
        ▼
┌──────────────────────────────────┐
│        Flask Backend             │
│      (Python 3.12, Flask 3.0)    │
│                                  │
│  ┌──────────────────────────┐    │
│  │    Text Preprocessing    │    │
│  │  (NLTK: tokenize,       │    │
│  │   stopwords, lemmatize)  │    │
│  └────────────┬─────────────┘    │
│               ▼                  │
│  ┌──────────────────────────┐    │
│  │   Feature Extraction     │    │
│  │  • VADER Sentiment       │    │
│  │  • TextBlob Polarity     │    │
│  │  • NRC Emotion Lexicon   │    │
│  │  • TF-IDF Vectorization  │    │
│  │  • N-gram Extraction     │    │
│  └────────────┬─────────────┘    │
│               ▼                  │
│  ┌──────────────────────────┐    │
│  │   ML Classification      │    │
│  │  • Logistic Regression   │    │
│  │  • Random Forest         │    │
│  │  • Support Vector Machine│    │
│  └────────────┬─────────────┘    │
│               ▼                  │
│  ┌──────────────────────────┐    │
│  │   Explainable AI (XAI)   │    │
│  │  • LIME Text Explainer   │    │
│  │  • Feature Importance    │    │
│  └────────────┬─────────────┘    │
│               ▼                  │
│  ┌──────────────────────────┐    │
│  │   Topic Modeling (LDA)   │    │
│  └──────────────────────────┘    │
└──────────────────────────────────┘
        │
        ▼
   JSON Response
   (risk level, explanation,
    sentiment, emotions, topics)
```

### 4.2 Data Flow

1. **User Input** → The user pastes social media text into the React frontend.
2. **API Call** → The frontend sends a POST request with the text to `/predict`.
3. **Preprocessing** → The Flask backend cleans, tokenizes, removes stopwords, and lemmatizes the text.
4. **Feature Extraction** → VADER sentiment, TextBlob polarity/subjectivity, NRC emotional tone, and TF-IDF vectors are computed.
5. **Classification** → The best-performing ML model (selected automatically during training) predicts the risk level.
6. **Explainability** → LIME generates word-level contribution weights explaining the prediction.
7. **Topic Modeling** → LDA identifies the dominant topic and keywords in the text.
8. **JSON Response** → All results are bundled into a structured JSON response and returned to the frontend.
9. **Visualization** → The frontend renders the risk badge, sentiment gauge, emotion charts, word contributions, and topic keywords.

### 4.3 Module Structure

```
backend/
├── app.py                    # Flask application & API endpoints
├── config.py                 # Centralized configuration constants
├── train.py                  # Model training orchestration script
├── requirements.txt          # Python dependency manifest
├── pipeline/
│   ├── preprocessor.py       # Text cleaning & NLP preprocessing
│   ├── feature_extractor.py  # Sentiment, emotion, n-gram, TF-IDF extraction
│   ├── model_trainer.py      # Multi-model training & evaluation
│   ├── explainer.py          # LIME-based explainability engine
│   └── topic_modeler.py      # LDA topic modeling
├── utils/
│   ├── dataset_generator.py  # Synthetic dataset creation
│   └── helpers.py            # Input validation & response formatting
├── models/                   # Serialized model artifacts (.joblib)
└── data/                     # Training dataset (.csv)

frontend/
├── index.html                # HTML entry point
├── vite.config.js            # Vite build & dev-server configuration
├── package.json              # Node.js dependency manifest
└── src/
    ├── main.jsx              # React DOM mount point
    ├── App.jsx               # Root application component
    ├── index.css             # Tailwind CSS v4 entry
    ├── components/
    │   ├── Header.jsx        # App header with API health indicator
    │   ├── TextInput.jsx     # Text area with validation & examples
    │   ├── PredictionResult.jsx  # Risk badge & probability bars
    │   ├── ExplainabilityView.jsx # LIME word pills & n-gram chips
    │   ├── SentimentGauge.jsx    # Gradient sentiment gauge
    │   └── FeatureChart.jsx      # Emotion bar chart & topic display
    └── services/
        └── api.js            # Axios HTTP client wrapper
```

---

## 5. Technology Stack

### 5.1 Backend Technologies

| Technology          | Version | Purpose                                                                                  |
| ------------------- | ------- | ---------------------------------------------------------------------------------------- |
| **Python**          | 3.12    | Core programming language for the backend, ML pipeline, and NLP processing               |
| **Flask**           | 3.0.0   | Lightweight WSGI micro-framework for building the REST API server                        |
| **Flask-CORS**      | 4.0.0   | Cross-Origin Resource Sharing middleware enabling frontend-backend communication         |
| **NLTK**            | 3.8.1   | Natural Language Toolkit — tokenization, stopword lists, WordNet lemmatization           |
| **Scikit-learn**    | 1.3.2   | Machine learning library — classifiers, TF-IDF vectorizer, train/test splitting, metrics |
| **VADER Sentiment** | 3.3.2   | Rule-based sentiment analysis tuned for social media text                                |
| **TextBlob**        | 0.17.1  | Simplified NLP API for polarity and subjectivity analysis                                |
| **Gensim**          | 4.3.2   | Topic modeling and document similarity (LDA support)                                     |
| **LIME**            | 0.2.0.1 | Local Interpretable Model-agnostic Explanations for prediction transparency              |
| **NumPy**           | 1.26.2  | Numerical computing for array operations and feature matrices                            |
| **Pandas**          | 2.1.4   | Data manipulation and CSV dataset loading                                                |
| **SciPy**           | 1.16.3  | Sparse matrix operations for feature combination                                         |
| **Joblib**          | 1.3.2   | Model serialization and persistence (`.joblib` format)                                   |

### 5.2 Frontend Technologies

| Technology            | Version | Purpose                                                                         |
| --------------------- | ------- | ------------------------------------------------------------------------------- |
| **React**             | 18.3.1  | Component-based UI library for building the interactive single-page application |
| **Vite**              | 6.0.7   | Next-generation frontend build tool with instant HMR and optimized bundling     |
| **Tailwind CSS**      | 4.0.0   | Utility-first CSS framework (v4 with `@import "tailwindcss"` directive)         |
| **@tailwindcss/vite** | 4.0.0   | Official Tailwind CSS Vite plugin for zero-config integration                   |
| **Axios**             | 1.7.9   | Promise-based HTTP client for API communication                                 |
| **Recharts**          | 2.15.0  | React-based composable charting library for data visualization                  |

### 5.3 Why These Technologies Were Chosen

- **Flask** over Django: Flask's micro-framework philosophy provides lightweight, flexible API creation without the overhead of a full ORM, admin panel, or template engine — ideal for a single-purpose ML inference server.
- **Vite** over Create React App: Vite offers dramatically faster cold starts (~150ms vs ~10s), native ES module support, and a leaner build pipeline. CRA is officially deprecated.
- **Tailwind CSS v4** over traditional CSS: Utility-first styling eliminates context-switching between CSS and JSX files, ensures consistent design tokens, and reduces bundle size through automatic purging of unused classes.
- **LIME** over SHAP: LIME provides model-agnostic, text-specific explanations with intuitive word-level contributions. Its `LimeTextExplainer` is purpose-built for NLP use cases and produces easily visualizable output.
- **VADER** over transformer-based sentiment models: VADER is rule-based, requires no GPU, runs in milliseconds, and is specifically tuned for social media text (handling slang, emoticons, capitalization, punctuation emphasis).

---

## 6. Data Description

### 6.1 Dataset Generation

The system uses a synthetically generated dataset to ensure balanced class distribution and controlled experimentation. The dataset generator (`utils/dataset_generator.py`) creates **1,500 labeled samples** (500 per class) using:

- **50 base text templates per class** — authored to reflect authentic social media language patterns for each risk level.
- **Augmentation strategies** — random prefix injection (e.g., "honestly", "tbh", "lately"), suffix appending (e.g., "i guess", "anyone else feel this way"), and word-level perturbation (random duplication or dropout).

### 6.2 Risk Level Definitions

| Risk Level          | Code | Description                                    | Example Indicators                                                         |
| ------------------- | ---- | ---------------------------------------------- | -------------------------------------------------------------------------- |
| **Low** (Green)     | 0    | Neutral or mildly emotional language           | Gratitude, daily activities, positive experiences, contentment             |
| **Medium** (Yellow) | 1    | Persistent negative emotions, distress signals | Sleep issues, anxiety, loss of interest, social withdrawal, feeling stuck  |
| **High** (Red)      | 2    | Strong negative sentiment, crisis indicators   | Hopelessness, suicidal ideation, self-harm references, desire to disappear |

### 6.3 Dataset Statistics

| Property          | Value                    |
| ----------------- | ------------------------ |
| Total samples     | 1,500                    |
| Samples per class | 500 (perfectly balanced) |
| Train/test split  | 80/20 (stratified)       |
| Training samples  | 1,200                    |
| Test samples      | 300                      |

---

## 7. NLP Pipeline & Text Preprocessing

The preprocessing module (`pipeline/preprocessor.py`) implements a multi-stage text cleaning pipeline optimized for social media content.

### 7.1 Preprocessing Stages

#### Stage 1: Text Cleaning

| Operation                | Technique          | Regex Pattern        | Rationale                                         |
| ------------------------ | ------------------ | -------------------- | ------------------------------------------------- |
| Lowercasing              | `str.lower()`      | —                    | Normalizes case for consistent feature extraction |
| URL removal              | Regex substitution | `http\S+\|www\.\S+`  | URLs carry no sentiment signal                    |
| Mention removal          | Regex substitution | `@\w+`               | User handles are noise                            |
| Hashtag normalization    | Regex substitution | `#` → (removed)      | Keeps hashtag text, removes symbol                |
| HTML stripping           | Regex substitution | `<.*?>`              | Removes markup artifacts                          |
| Non-alphabetic removal   | Regex substitution | `[^a-zA-Z\s]`        | Removes numbers, punctuation, emojis              |
| Whitespace normalization | Regex substitution | `\s+` → single space | Collapses multiple spaces                         |

#### Stage 2: Tokenization

Tokenization is performed using **NLTK's `word_tokenize()`**, which implements the Penn Treebank tokenization standard. It correctly handles:

- Contractions ("can't" → "ca", "n't")
- Punctuation boundaries
- Hyphenated words

#### Stage 3: Stopword Removal

Standard English stopwords from NLTK are removed **except for negation words**, which are critical for sentiment analysis:

**Retained negation words:** `not`, `no`, `never`, `neither`, `nobody`, `nothing`, `nowhere`, `nor`, `cannot`, `without`, `hardly`, `barely`

This is a deliberate design decision — removing "not" from "not happy" would invert the sentiment entirely.

#### Stage 4: Lemmatization

**WordNet Lemmatizer** from NLTK reduces words to their dictionary base form:

- "running" → "run"
- "feelings" → "feeling"
- "hopelessness" → "hopelessness" (noun form preserved)

Lemmatization was chosen over stemming (e.g., Porter Stemmer) because it produces valid English words, which improves both TF-IDF quality and explainability readability.

### 7.2 Pipeline Visualization

```
Raw Text: "I can't stop crying... @therapist Nothing feels right anymore. #mentalhealth"
    │
    ▼  [Clean]
"i cant stop crying therapist nothing feels right anymore mentalhealth"
    │
    ▼  [Tokenize]
["i", "cant", "stop", "crying", "therapist", "nothing", "feels", "right", "anymore", "mentalhealth"]
    │
    ▼  [Remove Stopwords — keep "nothing"]
["cant", "stop", "crying", "therapist", "nothing", "feels", "right", "anymore", "mentalhealth"]
    │
    ▼  [Lemmatize]
["cant", "stop", "cry", "therapist", "nothing", "feel", "right", "anymore", "mentalhealth"]
    │
    ▼  [Join]
"cant stop cry therapist nothing feel right anymore mentalhealth"
```

---

## 8. Feature Engineering

The feature extraction module (`pipeline/feature_extractor.py`) constructs a rich, multi-dimensional feature representation by combining **TF-IDF vectors** with **handcrafted linguistic features**.

### 8.1 TF-IDF Vectorization

**Term Frequency–Inverse Document Frequency (TF-IDF)** converts preprocessed text into a numerical vector by weighing terms that are frequent in a document but rare across the corpus.

**Formula:**

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{1 + \text{DF}(t)}\right)$$

Where:

- $\text{TF}(t, d)$ = frequency of term $t$ in document $d$
- $N$ = total number of documents
- $\text{DF}(t)$ = number of documents containing term $t$

**Configuration used:**

| Parameter      | Value  | Rationale                                                            |
| -------------- | ------ | -------------------------------------------------------------------- |
| `max_features` | 5,000  | Limits vocabulary to top 5K terms by TF-IDF score                    |
| `ngram_range`  | (1, 2) | Captures both unigrams and bigrams                                   |
| `min_df`       | 2      | Ignores terms appearing in fewer than 2 documents                    |
| `max_df`       | 0.95   | Ignores terms appearing in more than 95% of documents                |
| `sublinear_tf` | True   | Applies $1 + \log(\text{TF})$ scaling to dampen high-frequency terms |

This produces a sparse matrix of shape $(n\_samples, 5000)$ that captures the most discriminative vocabulary for each risk level.

### 8.2 VADER Sentiment Analysis

**VADER (Valence Aware Dictionary and sEntiment Reasoner)** is a rule-based sentiment analysis tool specifically attuned to social media text. Unlike ML-based sentiment models, VADER:

- Handles **capitalization** ("GREAT" is more intense than "great")
- Recognizes **punctuation emphasis** ("good!!!" is more positive than "good")
- Understands **degree modifiers** ("extremely bad" vs. "slightly bad")
- Processes **negation** ("not good" inverts polarity)
- Works with **slang and emoticons** (":)" is positive)

**Output features (4 values):**

| Feature    | Range    | Description                          |
| ---------- | -------- | ------------------------------------ |
| `compound` | [−1, +1] | Normalized, weighted composite score |
| `positive` | [0, 1]   | Proportion of positive sentiment     |
| `negative` | [0, 1]   | Proportion of negative sentiment     |
| `neutral`  | [0, 1]   | Proportion of neutral sentiment      |

### 8.3 TextBlob Sentiment Analysis

**TextBlob** provides a complementary sentiment analysis using a pattern-based approach:

| Feature        | Range    | Description                                       |
| -------------- | -------- | ------------------------------------------------- |
| `polarity`     | [−1, +1] | Negative to positive sentiment scale              |
| `subjectivity` | [0, 1]   | Objective (factual) to subjective (opinion) scale |

Using both VADER and TextBlob provides **dual-source sentiment triangulation** — if both tools agree on strong negativity, the classifier has stronger evidence.

### 8.4 NRC Emotion Lexicon (Keyword-Based)

The system implements a custom keyword-based emotion detector modeled after the **NRC Word-Emotion Association Lexicon** (Mohammad & Turney, 2013). It maps input words to eight fundamental emotions from **Plutchik's Wheel of Emotions**:

| Emotion          | Example Keywords                      | Typical Risk Association |
| ---------------- | ------------------------------------- | ------------------------ |
| **Joy**          | happy, love, wonderful, grateful      | Low risk                 |
| **Trust**        | believe, faith, support, safe         | Low risk                 |
| **Anticipation** | hope, eager, future, goal             | Low risk                 |
| **Surprise**     | shocked, amazed, unexpected           | Variable                 |
| **Sadness**      | depressed, hopeless, lonely, empty    | Medium–High risk         |
| **Anger**        | furious, hate, frustrated, hostile    | Medium risk              |
| **Fear**         | terrified, anxious, panic, trapped    | Medium–High risk         |
| **Disgust**      | disgusting, awful, terrible, pathetic | Medium risk              |

Each emotion score is normalized by dividing the matched word count by the total matched count across all emotions, producing a probability distribution over eight emotion categories.

### 8.5 N-gram Extraction

The system extracts **unigrams** (single words) and **bigrams** (two-word phrases) from preprocessed text, ranked by frequency. These provide:

- **Unigrams** — Most frequent individual terms (e.g., "hopeless", "alone", "happy").
- **Bigrams** — Contextual word pairs (e.g., "not happy", "feeling lost", "want disappear").

Bigrams are particularly valuable because they capture **negation patterns** and **phrase-level meaning** that unigrams alone would miss.

### 8.6 Combined Feature Vector

The final feature vector concatenates TF-IDF features with handcrafted linguistic features:

```
Feature Vector = [TF-IDF (5000 dims)] ⊕ [VADER (4)] ⊕ [TextBlob (2)] ⊕ [Emotions (8)]
Total dimensions = 5,014 (approximately, varies with vocabulary)
Actual dimensions in this system = 2,007
```

The combination of **high-dimensional sparse features** (TF-IDF) with **low-dimensional dense features** (sentiment + emotion) gives classifiers both broad lexical coverage and concentrated psychological signal.

---

## 9. Machine Learning Algorithms

Three supervised classification algorithms were trained and evaluated. The system automatically selects the best-performing model based on weighted F1-score.

### 9.1 Logistic Regression

**Algorithm:** Multinomial logistic regression (softmax regression) extends binary logistic regression to multi-class problems.

**Decision Function:**

$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x}}}$$

Where $\mathbf{w}_k$ is the weight vector for class $k$, and $\mathbf{x}$ is the feature vector.

**Hyperparameters used:**

| Parameter      | Value       | Rationale                                                 |
| -------------- | ----------- | --------------------------------------------------------- |
| `max_iter`     | 1000        | Ensures convergence for high-dimensional features         |
| `C`            | 1.0         | Inverse regularization strength (default balance)         |
| `multi_class`  | multinomial | Native multi-class with softmax (not one-vs-rest)         |
| `class_weight` | balanced    | Adjusts weights inversely proportional to class frequency |
| `random_state` | 42          | Reproducibility                                           |

**Strengths:** Produces calibrated probability estimates; coefficients are directly interpretable as feature importance; fast training and inference.

### 9.2 Random Forest

**Algorithm:** An ensemble of decision trees where each tree is trained on a bootstrap sample of the data with random feature subsets. Final prediction is by majority vote.

**Key Mechanism:**

$$\hat{y} = \text{mode}\left(\{h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_B(\mathbf{x})\}\right)$$

Where $h_b$ is the $b$-th decision tree and $B$ is the total number of trees.

**Hyperparameters used:**

| Parameter      | Value    | Rationale                                |
| -------------- | -------- | ---------------------------------------- |
| `n_estimators` | 200      | Number of trees in the forest            |
| `max_depth`    | 20       | Limits tree depth to prevent overfitting |
| `class_weight` | balanced | Handles class imbalance                  |
| `n_jobs`       | -1       | Parallelizes across all CPU cores        |
| `random_state` | 42       | Reproducibility                          |

**Strengths:** Handles non-linear relationships; provides built-in `feature_importances_` via Gini impurity reduction; robust to outliers; minimal hyperparameter tuning needed.

### 9.3 Support Vector Machine (SVM)

**Algorithm:** SVM finds the optimal hyperplane that maximizes the margin between classes in the feature space. The linear kernel computes:

$$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$

For multi-class classification, scikit-learn uses a **one-vs-one** strategy, training $\frac{K(K-1)}{2}$ binary classifiers.

**Hyperparameters used:**

| Parameter      | Value    | Rationale                                                             |
| -------------- | -------- | --------------------------------------------------------------------- |
| `kernel`       | linear   | Effective for high-dimensional text data (TF-IDF)                     |
| `C`            | 1.0      | Regularization parameter balancing margin width and misclassification |
| `probability`  | True     | Enables Platt scaling for probability estimates (needed by LIME)      |
| `class_weight` | balanced | Handles class imbalance                                               |
| `random_state` | 42       | Reproducibility                                                       |

**Strengths:** Excels in high-dimensional spaces where $d > n$; maximizes margin for robust generalization; linear SVM is memory-efficient with sparse TF-IDF matrices.

### 9.4 Model Selection Strategy

All three models are trained on the same 80/20 stratified train/test split. Additionally, **5-fold stratified cross-validation** is performed on the full dataset. The model with the highest **weighted F1-score on the test set** is automatically selected as the production model and serialized for deployment.

---

## 10. Explainable AI (XAI)

### 10.1 Why Explainability Matters

In mental health applications, a black-box "High Risk" label without justification is:

- **Clinically useless** — professionals need to understand _why_ the system flagged a text.
- **Ethically problematic** — opaque decisions affecting mental health assessment require accountability.
- **Trust-eroding** — users and evaluators cannot validate or correct the system without insight.

### 10.2 LIME (Local Interpretable Model-Agnostic Explanations)

**LIME** (Ribeiro et al., 2016) is the primary explainability technique used in this system. It explains individual predictions by approximating the complex model locally with an interpretable linear model.

**Algorithm:**

1. **Perturb** the input text by randomly removing words to create $N$ neighborhood samples (200 in our configuration).
2. **Predict** the risk level for each perturbed sample using the trained classifier.
3. **Weight** each perturbed sample by its cosine similarity to the original input.
4. **Fit** a weighted linear regression on the perturbed samples to approximate the model's local decision boundary.
5. **Extract** feature coefficients from the linear model — these become the word-level contribution weights.

**Mathematical Formulation:**

$$\xi(\mathbf{x}) = \arg\min_{g \in G} \; \mathcal{L}(f, g, \pi_{\mathbf{x}}) + \Omega(g)$$

Where:

- $f$ is the original black-box model
- $g$ is the interpretable surrogate model (linear)
- $\pi_{\mathbf{x}}$ is the locality kernel (proximity measure)
- $\Omega(g)$ is a complexity penalty favoring simpler explanations

**Configuration used:**

| Parameter          | Value                                |
| ------------------ | ------------------------------------ |
| `class_names`      | ["Low", "Medium", "High"]            |
| `num_features`     | 10 (top contributing words shown)    |
| `num_samples`      | 200 (perturbation neighborhood size) |
| `split_expression` | `\W+` (word-boundary splitting)      |

**Output:** A list of (word, weight) pairs where:

- **Positive weight** → pushes prediction toward higher risk.
- **Negative weight** → pushes prediction toward lower risk.

### 10.3 Feature Importance (Global Explainability)

In addition to LIME's local explanations, the system supports **global feature importance**:

- **Random Forest** → Gini impurity-based `feature_importances_`.
- **Logistic Regression / SVM** → Absolute coefficient magnitudes `|coef_|` averaged across classes.

This reveals which features are consistently important across all predictions, not just one.

---

## 11. Topic Modeling

### 11.1 Latent Dirichlet Allocation (LDA)

The system uses **LDA** (Blei et al., 2003) for unsupervised topic discovery. LDA is a generative probabilistic model that assumes:

1. Each document is a **mixture of topics**.
2. Each topic is a **distribution over words**.

**Generative Process:**

$$P(w \mid d) = \sum_{k=1}^{K} P(w \mid z=k) \cdot P(z=k \mid d)$$

Where:

- $P(w \mid z=k)$ is the probability of word $w$ given topic $k$
- $P(z=k \mid d)$ is the probability of topic $k$ in document $d$

**Configuration used:**

| Parameter                   | Value  |
| --------------------------- | ------ |
| `n_components` (topics)     | 5      |
| `learning_method`           | online |
| `max_iter`                  | 20     |
| `max_features` (vocabulary) | 2,000  |

### 11.2 Topic Prediction

For each user input, the system:

1. Transforms the text using the fitted `CountVectorizer`.
2. Computes the topic distribution via `model.transform()`.
3. Returns the **dominant topic**, its **top 5 keywords**, and the **confidence score**.

This helps users understand _what the text is about_ — distinct from _what sentiment it carries_.

---

## 12. Backend Architecture (Flask REST API)

### 12.1 Application Design

The Flask application (`app.py`) follows a **load-once, serve-many** pattern:

- All ML models (classifier, TF-IDF vectorizer, topic model) are loaded into memory at server startup.
- Subsequent requests reuse the in-memory models with zero deserialization overhead.
- This ensures sub-second response times (typically < 2 seconds even with LIME explanations).

### 12.2 API Endpoints

#### `GET /health`

Returns system status and model availability.

**Response:**

```json
{
  "status": "healthy",
  "models_loaded": true,
  "model_name": "svm",
  "disclaimer": "This tool is NOT a clinical diagnosis system."
}
```

#### `POST /predict`

Accepts text and returns risk prediction with full explainability.

**Request:**

```json
{ "text": "I feel completely alone and hopeless..." }
```

**Response structure:**

```json
{
  "prediction": {
    "risk_level": "High",
    "risk_code": 2,
    "color": "#ef4444",
    "description": "Strong negative sentiment, crisis indicators",
    "probabilities": { "low": 0.0, "medium": 0.0, "high": 1.0 },
    "model_used": "svm"
  },
  "explanation": {
    "word_contributions": [
      { "word": "hopeless", "weight": -0.0699, "impact": "negative" },
      { "word": "alone", "weight": -0.0412, "impact": "negative" }
    ]
  },
  "features": {
    "vader_sentiment": { "compound": -0.8555, "positive": 0.0, ... },
    "textblob_sentiment": { "polarity": -0.65, "subjectivity": 0.9 },
    "emotional_tone": { "sadness": 0.4, "fear": 0.3, ... },
    "top_unigrams": [...],
    "top_bigrams": [...]
  },
  "topic": { "topic_id": 2, "keywords": ["feel", "hopeless", ...] },
  "disclaimer": "This is NOT a clinical diagnosis."
}
```

#### `POST /analyze`

Returns only linguistic analysis (sentiment, emotion, topics) without risk prediction.

#### `GET /evaluation`

Returns the model evaluation report (accuracy, precision, recall, F1, confusion matrices).

### 12.3 Input Validation

All text inputs are validated before processing:

| Check          | Constraint                 |
| -------------- | -------------------------- |
| Minimum length | ≥ 10 characters            |
| Maximum length | ≤ 5,000 characters         |
| Minimum words  | ≥ 3 actual words           |
| Type check     | Must be a non-empty string |

### 12.4 CORS Configuration

Flask-CORS is configured with `resources={r"/*": {"origins": "*"}}` to allow cross-origin requests from the Vite development server (port 3000) to the Flask API (port 5000).

---

## 13. Frontend Architecture (React + Vite)

### 13.1 Build Tooling

**Vite 6.0** was chosen as the frontend build tool for:

- **Instant server start** (~150ms cold start vs. ~10s for Webpack-based CRA).
- **Lightning-fast HMR** (Hot Module Replacement) — changes reflect in the browser in < 50ms.
- **Native ES module support** — no bundling needed during development.
- **Optimized production builds** via Rollup with tree-shaking and code splitting.

**Dev server proxy** — Vite's proxy configuration forwards API routes (`/predict`, `/analyze`, `/health`, `/evaluation`) to Flask at `localhost:5000`, eliminating CORS issues during development.

### 13.2 Tailwind CSS v4

Tailwind CSS v4 introduces a completely new engine with:

- **Zero-config** — no `tailwind.config.js` file needed.
- **CSS-first configuration** — all customization via `@import "tailwindcss"` in CSS.
- **Native `@tailwindcss/vite` plugin** — integrated directly into the Vite pipeline.
- **Smaller output** — automatic tree-shaking of unused utilities.

### 13.3 Component Architecture

| Component            | Props                     | Responsibility                                                                                       |
| -------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------- |
| `App`                | —                         | Root component; manages global state (result, loading, error); orchestrates API calls                |
| `Header`             | `healthy`                 | Displays app title, subtitle, and live API connection status badge (green pulse / red)               |
| `TextInput`          | `onSubmit`, `loading`     | Textarea with character counter, input validation, example text buttons, submit with loading spinner |
| `PredictionResult`   | `prediction`              | Risk level badge (color-coded), risk description, probability bar chart for all 3 classes            |
| `SentimentGauge`     | `sentiment`               | Gradient bar (red → gray → green) with needle indicator, VADER + TextBlob score cards                |
| `ExplainabilityView` | `explanation`, `features` | LIME word contribution pills (color-coded by impact), unigram/bigram frequency chips                 |
| `FeatureChart`       | `emotions`, `topic`       | Recharts horizontal bar chart for emotion distribution, topic keyword pills with confidence          |

### 13.4 State Management

The application uses **React's built-in `useState` and `useEffect` hooks** — no external state management library (Redux, Zustand) is needed because:

- State is localized to a single page.
- Only 4 state variables: `healthy`, `loading`, `error`, `result`.
- Data flows unidirectionally: API response → `App` state → child component props.

### 13.5 Visualization Library

**Recharts 2.15** was chosen for the emotion bar chart because:

- Built on React components (no DOM manipulation conflicts).
- Declarative API with `<BarChart>`, `<Bar>`, `<XAxis>`, `<YAxis>`, `<Tooltip>`.
- `<ResponsiveContainer>` adapts to parent width automatically.
- Supports horizontal layout for emotion labels.

---

## 14. Model Evaluation Results

### 14.1 Performance Summary

| Model                   | Accuracy   | Precision  | Recall     | F1-Score   | CV F1 (5-fold)      |
| ----------------------- | ---------- | ---------- | ---------- | ---------- | ------------------- |
| **Logistic Regression** | 0.9833     | 0.9837     | 0.9833     | 0.9834     | 0.9953 ± 0.0050     |
| **Random Forest**       | 0.9600     | 0.9607     | 0.9600     | 0.9601     | 0.9834 ± 0.0107     |
| **SVM (Linear)**        | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.9993 ± 0.0013** |

**Best model selected: SVM (Linear Kernel)**

### 14.2 Confusion Matrices

**Logistic Regression:**

|                   | Predicted Low | Predicted Medium | Predicted High |
| ----------------- | ------------- | ---------------- | -------------- |
| **Actual Low**    | 97            | 3                | 0              |
| **Actual Medium** | 1             | 99               | 0              |
| **Actual High**   | 0             | 1                | 99             |

**Random Forest:**

|                   | Predicted Low | Predicted Medium | Predicted High |
| ----------------- | ------------- | ---------------- | -------------- |
| **Actual Low**    | 98            | 2                | 0              |
| **Actual Medium** | 2             | 96               | 2              |
| **Actual High**   | 0             | 6                | 94             |

**SVM (Linear):**

|                   | Predicted Low | Predicted Medium | Predicted High |
| ----------------- | ------------- | ---------------- | -------------- |
| **Actual Low**    | 100           | 0                | 0              |
| **Actual Medium** | 0             | 100              | 0              |
| **Actual High**   | 0             | 0                | 100            |

### 14.3 Per-Class Performance (Best Model — SVM)

| Class            | Precision  | Recall     | F1-Score   | Support |
| ---------------- | ---------- | ---------- | ---------- | ------- |
| Low              | 1.0000     | 1.0000     | 1.0000     | 100     |
| Medium           | 1.0000     | 1.0000     | 1.0000     | 100     |
| High             | 1.0000     | 1.0000     | 1.0000     | 100     |
| **Weighted Avg** | **1.0000** | **1.0000** | **1.0000** | **300** |

### 14.4 Cross-Validation Analysis

5-fold stratified cross-validation confirms that the high test-set performance is not due to a lucky split:

- SVM achieves **0.9993 ± 0.0013** mean F1 across folds.
- Low standard deviation (0.0013) indicates consistent generalization.

### 14.5 Evaluation Metrics Explained

| Metric        | Formula                             | What It Measures                             |
| ------------- | ----------------------------------- | -------------------------------------------- |
| **Accuracy**  | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall correctness                          |
| **Precision** | $\frac{TP}{TP + FP}$                | Of predicted positives, how many are correct |
| **Recall**    | $\frac{TP}{TP + FN}$                | Of actual positives, how many were detected  |
| **F1-Score**  | $2 \cdot \frac{P \cdot R}{P + R}$   | Harmonic mean of precision and recall        |

---

## 15. Ethical Considerations

### 15.1 Disclaimer

> **⚠️ This tool is NOT a clinical diagnosis system.** It is intended for educational and research purposes only. If you or someone you know is in crisis, please contact a mental health professional or a crisis helpline.

### 15.2 Design Principles

| Principle                  | Implementation                                                               |
| -------------------------- | ---------------------------------------------------------------------------- |
| **Non-intrusive**          | Users voluntarily paste text; no scraping or monitoring                      |
| **No data retention**      | Zero storage of user inputs; each request is stateless                       |
| **No user identification** | No login, no tracking, no cookies, no PII collection                         |
| **Transparency**           | Every prediction is accompanied by LIME explanations showing _why_           |
| **Decision support only**  | System explicitly states it is not a replacement for professional assessment |
| **Balanced training**      | Equal samples per class prevent majority-class bias                          |

### 15.3 Potential Risks & Mitigations

| Risk                                | Mitigation                                                                 |
| ----------------------------------- | -------------------------------------------------------------------------- |
| False negatives (missing high-risk) | System is a supplement, never a replacement for professional judgment      |
| False positives (unnecessary alarm) | Probabilities and explanations allow users to contextualize predictions    |
| Misuse as diagnostic tool           | Prominent disclaimers on every page and in every API response              |
| Bias in training data               | Synthetic data with controlled, diverse templates reduces demographic bias |

---

## 16. Limitations & Future Work

### 16.1 Current Limitations

1. **English only** — the system does not support multilingual text.
2. **Text only** — no analysis of images, videos, or audio.
3. **Synthetic dataset** — while balanced and controlled, synthetic data may not fully capture the complexity and nuance of real-world social media language.
4. **No temporal analysis** — the system analyzes single texts in isolation, not patterns across a user's posting history.
5. **Perfect test accuracy caveat** — the SVM's 100% test accuracy reflects the distinguishability of the synthetic dataset; performance on real-world data would likely be lower.

### 16.2 Future Enhancements

1. **Real-world datasets** — integrate Twitter Depression Dataset, Reddit Mental Health Posts, and Kaggle mental health datasets for more realistic training.
2. **Transformer-based models** — fine-tune BERT, RoBERTa, or MentalBERT for richer contextual understanding.
3. **Multilingual support** — extend to Spanish, Hindi, and other high-prevalence languages.
4. **Temporal pattern analysis** — track sentiment trajectories over multiple posts.
5. **SHAP integration** — add SHAP (SHapley Additive exPlanations) for game-theory-based feature attribution.
6. **User feedback loop** — allow domain experts to flag and correct predictions, creating an active learning pipeline.
7. **Mobile-responsive UI** — optimize for mobile devices where most social media usage occurs.
8. **Docker deployment** — containerize the full stack for one-command deployment.

---

## 17. Conclusion

This project demonstrates a complete, end-to-end **Explainable AI framework** for mental health risk detection from social media text. By combining:

- A **rigorous NLP preprocessing pipeline** (cleaning, tokenization, stopword removal, lemmatization),
- **Multi-source feature engineering** (TF-IDF, VADER, TextBlob, emotion lexicon, n-grams),
- **Three supervised classifiers** (Logistic Regression, Random Forest, SVM) with automatic best-model selection,
- **LIME-based explainability** providing word-level contribution analysis,
- **LDA topic modeling** for thematic context, and
- A **modern full-stack web application** (Flask + React/Vite/Tailwind v4),

the system achieves high classification accuracy (up to 100% on the test set) while maintaining full transparency into its decision-making process. Every prediction is accompanied by human-readable explanations — which words contributed, what emotions were detected, what sentiment was measured, and what topics were identified.

The framework is designed as a **decision-support tool for researchers and educators**, with strong ethical safeguards including zero data retention, prominent disclaimers, and model-agnostic explainability that invites scrutiny rather than blind trust.

---
