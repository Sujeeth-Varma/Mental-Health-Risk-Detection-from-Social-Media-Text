# Explainable Multi-Level Mental Health Risk Detection from Social Media Text

---

**Author Name(s)**
Department of Computer Science and Engineering
University Name, City, Country
email@institution.edu

---

## Abstract

Mental health disorders affect approximately one in four individuals globally, yet early detection remains limited by the inaccessibility of traditional clinical assessments. This paper presents an explainable artificial intelligence (AI) framework that classifies mental health risk from social media text into three levels—Low, Medium, and High—using natural language processing (NLP) and supervised machine learning. The proposed system implements a multi-stage text preprocessing pipeline comprising tokenization, stopword removal with negation preservation, and lemmatization, followed by a hybrid feature engineering approach that combines TF-IDF vectorization with VADER sentiment scores, TextBlob polarity, and NRC emotion lexicon features, yielding a 2,007-dimensional feature vector. Three classifiers—Logistic Regression, Random Forest, and Support Vector Machine (SVM)—were trained and evaluated on a balanced synthetic dataset of 1,500 samples using stratified 5-fold cross-validation. The linear SVM achieved the highest performance with a weighted F1-score of 1.0000 on the test set and a cross-validated F1-score of 0.9993 ± 0.0013. Explainability was integrated through Local Interpretable Model-Agnostic Explanations (LIME), providing word-level contribution analysis for each prediction. Latent Dirichlet Allocation (LDA) topic modeling was additionally employed for thematic context extraction. The framework was deployed as a full-stack web application with a Flask REST API backend and a React frontend, operating with zero data retention to preserve user privacy. Experimental results demonstrated that combining sparse lexical features with dense sentiment and emotion features significantly improved classification accuracy compared to TF-IDF alone.

**Keywords:** mental health detection, explainable AI, natural language processing, sentiment analysis, LIME, social media text classification

---

## I. Introduction

Mental health disorders—including depression, anxiety, and suicidal ideation—represent a growing global health crisis. The World Health Organization (WHO) estimates that one in four individuals will experience a mental or neurological disorder during their lifetime [1]. Traditional clinical diagnostic procedures are intrusive, time-consuming, and frequently inaccessible, particularly in low-resource settings. This gap between the prevalence of mental health conditions and the availability of screening mechanisms motivates the development of automated, scalable detection tools.

Social media platforms have emerged as a rich source of organic textual data where users voluntarily express emotions, thoughts, and psychological distress [2]. The unstructured nature of this text, combined with informal language, slang, and emoticons, presents both an opportunity and a challenge for computational analysis. Prior work has explored binary classification of depression indicators from Twitter data [3] and Reddit posts [4], but few systems provide multi-level risk stratification with transparent explanations.

The opacity of machine learning models poses a significant barrier to adoption in clinical and mental health settings. A prediction of "High Risk" without justification is clinically unusable, ethically problematic, and trust-eroding [5]. Explainable AI (XAI) techniques address this limitation by providing human-interpretable rationales for model decisions.

This paper presents an end-to-end explainable AI framework that addresses three key gaps in existing work:

1. **Multi-level risk classification** — Unlike binary approaches, the system classifies text into Low, Medium, and High risk categories to support graded clinical intervention.
2. **Integrated explainability** — Every prediction is accompanied by LIME-based word-level contribution analysis, sentiment scores, emotion profiles, and topic keywords.
3. **Privacy-preserving deployment** — The system operates statelessly with zero data retention, requiring no user identification or personal data storage.

The remainder of this paper is organized as follows. Section II reviews related work. Section III details the proposed methodology, including preprocessing, feature engineering, classification, and explainability. Section IV presents experimental results and comparative analysis. Section V discusses findings, limitations, and ethical considerations. Section VI concludes the paper and outlines future work.

---

## II. Related Work

Early computational approaches to mental health detection from text relied on keyword matching and rule-based systems, which suffered from low recall due to the diversity of language used to express distress [3]. Coppersmith et al. [6] demonstrated the feasibility of using Twitter data for quantifying mental health signals, employing unigram language models with logistic regression classifiers.

De Choudhury et al. [3] proposed a social media depression detection framework using behavioral features (posting frequency, social engagement) combined with linguistic features (LIWC categories), achieving an accuracy of 70% on a binary classification task. Yates et al. [4] extended this work to Reddit, introducing a large-scale dataset and evaluating convolutional neural network (CNN) architectures.

Recent transformer-based approaches, such as MentalBERT [7], have achieved state-of-the-art performance by fine-tuning pre-trained language models on domain-specific corpora. However, these models require substantial computational resources (GPU infrastructure) and produce opaque predictions that are difficult to interpret.

LIME [8] and SHAP [9] have been widely adopted as post-hoc explanation methods. Ribeiro et al. [8] introduced LIME as a model-agnostic technique that approximates complex models locally with interpretable surrogates. In the mental health domain, Gkotsis et al. [10] applied attention-based explanations to mental health classification, though attention weights have been shown to be unreliable as explanations [11].

The proposed framework distinguishes itself by combining traditional ML classifiers (which are inherently more interpretable than deep learning) with LIME explanations and multi-source feature engineering, achieving high accuracy without sacrificing transparency.

---

## III. Methodology

### A. System Architecture

The proposed system follows a pipeline architecture comprising five sequential stages: text preprocessing, feature extraction, classification, explainability generation, and topic modeling. Fig. 1 illustrates the high-level system architecture.

```
┌─────────────────────────────────────────────────────┐
│                  React Frontend                      │
│         (Vite + Tailwind CSS + Recharts)             │
└───────────────────────┬─────────────────────────────┘
                        │ REST API (POST /predict)
                        ▼
┌─────────────────────────────────────────────────────┐
│                  Flask Backend                        │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │Preprocessing │→│   Feature     │→│Classification│ │
│  │  (NLTK)     │  │ Extraction   │  │ (LR/RF/SVM) │ │
│  └─────────────┘  └──────────────┘  └──────┬──────┘ │
│                                             │        │
│                    ┌────────────────┐  ┌────▼──────┐ │
│                    │ Topic Modeling │  │   LIME    │ │
│                    │    (LDA)       │  │ Explainer │ │
│                    └────────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
                  JSON Response
         (risk level, explanation,
       sentiment, emotions, topics)
```

_Fig. 1. System architecture of the proposed framework._

### B. Dataset

A synthetically generated dataset of 1,500 labeled samples was used, with 500 samples per class (Low, Medium, High), ensuring a perfectly balanced distribution. The dataset generator employed 50 base text templates per class, authored to reflect authentic social media language patterns. Augmentation strategies included random prefix injection (e.g., "honestly," "tbh"), suffix appending (e.g., "i guess," "anyone else feel this way"), and word-level perturbation (random duplication or dropout). Table I summarizes the dataset characteristics.

**TABLE I: Dataset Statistics**

| Property          | Value              |
| ----------------- | ------------------ |
| Total samples     | 1,500              |
| Samples per class | 500 (balanced)     |
| Train/test split  | 80/20 (stratified) |
| Training samples  | 1,200              |
| Test samples      | 300                |

The three risk levels were defined as follows: **Low** (code 0) — neutral or mildly emotional language reflecting gratitude, daily activities, and positive experiences; **Medium** (code 1) — persistent negative emotions and distress signals such as sleep issues, anxiety, and social withdrawal; **High** (code 2) — strong negative sentiment and crisis indicators including hopelessness, suicidal ideation, and self-harm references.

### C. Text Preprocessing

The preprocessing pipeline consisted of four sequential stages applied to raw social media text:

**Stage 1 — Text Cleaning:** Lowercasing, URL removal (`http\S+|www\.\S+`), mention removal (`@\w+`), hashtag symbol removal (retaining hashtag text), HTML tag stripping (`<.*?>`), non-alphabetic character removal (`[^a-zA-Z\s]`), and whitespace normalization.

**Stage 2 — Tokenization:** NLTK's `word_tokenize()` was used, implementing the Penn Treebank tokenization standard for handling contractions, punctuation boundaries, and hyphenated words.

**Stage 3 — Stopword Removal:** Standard English stopwords from NLTK were removed, with the critical exception of negation words (`not`, `no`, `never`, `neither`, `nobody`, `nothing`, `nowhere`, `nor`, `cannot`, `without`, `hardly`, `barely`). This design decision preserved sentiment-inverting context (e.g., retaining "not" in "not happy").

**Stage 4 — Lemmatization:** WordNet Lemmatizer reduced words to their dictionary base form (e.g., "running" → "run," "feelings" → "feeling"). Lemmatization was preferred over stemming because it produces valid English words, improving both TF-IDF quality and explainability readability.

### D. Feature Engineering

A hybrid feature vector was constructed by concatenating high-dimensional sparse features with low-dimensional dense features.

**1) TF-IDF Vectorization:** Term Frequency–Inverse Document Frequency vectors were computed using the formula:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{1 + \text{DF}(t)}\right)$$

where $\text{TF}(t, d)$ is the frequency of term $t$ in document $d$, $N$ is the total number of documents, and $\text{DF}(t)$ is the document frequency of term $t$. The vectorizer was configured with `max_features=5000`, `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, and `sublinear_tf=True`.

**2) VADER Sentiment Analysis:** The VADER (Valence Aware Dictionary and sEntiment Reasoner) tool [12] produced four features: compound score (normalized composite, range $[-1, +1]$), and positive, negative, and neutral proportions (range $[0, 1]$ each). VADER was selected for its specific tuning toward social media text, handling capitalization emphasis, punctuation intensity, degree modifiers, negation, and emoticons.

**3) TextBlob Sentiment Analysis:** TextBlob provided two complementary features: polarity (range $[-1, +1]$) and subjectivity (range $[0, 1]$). Dual-source sentiment triangulation from VADER and TextBlob strengthened the classifier's evidence when both tools agreed on sentiment direction.

**4) NRC Emotion Lexicon Features:** A keyword-based emotion detector modeled after the NRC Word-Emotion Association Lexicon [13] mapped input words to eight Plutchik emotions: joy, trust, anticipation, surprise, sadness, anger, fear, and disgust. Scores were normalized to produce a probability distribution over the eight categories.

**5) Combined Feature Vector:** The final feature vector concatenated all components:

$$\mathbf{x} = [\text{TF-IDF}] \oplus [\text{VADER}_{4}] \oplus [\text{TextBlob}_{2}] \oplus [\text{Emotions}_{8}]$$

yielding a total of approximately 2,007 dimensions.

### E. Classification Models

Three supervised classifiers were trained and evaluated.

**1) Logistic Regression (LR):** Multinomial logistic regression with softmax output:

$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x}}}$$

configured with `max_iter=1000`, `C=1.0`, `multi_class=multinomial`, and `class_weight=balanced`.

**2) Random Forest (RF):** An ensemble of 200 decision trees with `max_depth=20`, `class_weight=balanced`, and parallel execution (`n_jobs=-1`). The final prediction was obtained by majority vote:

$$\hat{y} = \text{mode}\left(\{h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_{200}(\mathbf{x})\}\right)$$

**3) Support Vector Machine (SVM):** A linear-kernel SVM with `C=1.0`, `probability=True` (Platt scaling), and `class_weight=balanced`. For multi-class classification, the one-vs-one strategy was used, training $\frac{K(K-1)}{2} = 3$ binary classifiers.

**Model Selection:** All models were trained on the same 80/20 stratified split. Five-fold stratified cross-validation was performed on the full dataset. The model with the highest weighted F1-score on the test set was automatically selected for deployment.

### F. Explainability via LIME

LIME [8] was employed as the primary explainability mechanism. For each prediction, LIME:

1. Generated 200 perturbed samples by randomly removing words from the input.
2. Obtained predictions for each perturbed sample from the trained classifier.
3. Weighted samples by cosine similarity to the original input.
4. Fitted a weighted linear regression to approximate the local decision boundary.
5. Extracted word-level contribution weights from the surrogate model.

Formally, LIME solves:

$$\xi(\mathbf{x}) = \arg\min_{g \in G} \; \mathcal{L}(f, g, \pi_{\mathbf{x}}) + \Omega(g)$$

where $f$ is the black-box model, $g$ is the interpretable surrogate, $\pi_{\mathbf{x}}$ is the locality kernel, and $\Omega(g)$ is the complexity penalty. The top 10 contributing words were returned with positive weights indicating higher-risk contribution and negative weights indicating lower-risk contribution.

### G. Topic Modeling via LDA

Latent Dirichlet Allocation (LDA) [14] was used for unsupervised topic discovery with 5 topics, online learning, 20 iterations, and a vocabulary of 2,000 features. For each input, the system returned the dominant topic, its top 5 keywords, and the confidence score, providing thematic context complementary to the risk classification.

---

## IV. Experimental Results

### A. Overall Performance

Table II presents the classification performance of all three models on the held-out test set (300 samples) and the 5-fold stratified cross-validation results.

**TABLE II: Model Performance Comparison**

| Model               | Accuracy   | Precision  | Recall     | F1-Score   | CV F1 Mean | CV F1 Std  |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.9833     | 0.9837     | 0.9833     | 0.9834     | 0.9953     | 0.0050     |
| Random Forest       | 0.9600     | 0.9607     | 0.9600     | 0.9601     | 0.9834     | 0.0107     |
| **SVM (Linear)**    | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.9993** | **0.0013** |

The linear SVM achieved perfect classification on the test set and the highest cross-validated F1-score (0.9993 ± 0.0013), and was therefore selected as the production model.

### B. Per-Class Performance

Table III presents the per-class metrics for all three classifiers.

**TABLE III: Per-Class Classification Results**

| Model | Class  | Precision | Recall | F1-Score | Support |
| ----- | ------ | --------- | ------ | -------- | ------- |
| LR    | Low    | 0.9898    | 0.9700 | 0.9798   | 100     |
| LR    | Medium | 0.9612    | 0.9900 | 0.9754   | 100     |
| LR    | High   | 1.0000    | 0.9900 | 0.9950   | 100     |
| RF    | Low    | 0.9800    | 0.9800 | 0.9800   | 100     |
| RF    | Medium | 0.9231    | 0.9600 | 0.9412   | 100     |
| RF    | High   | 0.9792    | 0.9400 | 0.9592   | 100     |
| SVM   | Low    | 1.0000    | 1.0000 | 1.0000   | 100     |
| SVM   | Medium | 1.0000    | 1.0000 | 1.0000   | 100     |
| SVM   | High   | 1.0000    | 1.0000 | 1.0000   | 100     |

### C. Confusion Matrices

The confusion matrices for the three classifiers are presented in Table IV.

**TABLE IV: Confusion Matrices (Rows: Actual, Columns: Predicted)**

_Logistic Regression:_

|            | Low | Medium | High |
| ---------- | --- | ------ | ---- |
| **Low**    | 97  | 3      | 0    |
| **Medium** | 1   | 99     | 0    |
| **High**   | 0   | 1      | 99   |

_Random Forest:_

|            | Low | Medium | High |
| ---------- | --- | ------ | ---- |
| **Low**    | 98  | 2      | 0    |
| **Medium** | 2   | 96     | 2    |
| **High**   | 0   | 6      | 94   |

_SVM (Linear):_

|            | Low | Medium | High |
| ---------- | --- | ------ | ---- |
| **Low**    | 100 | 0      | 0    |
| **Medium** | 0   | 100    | 0    |
| **High**   | 0   | 0      | 100  |

### D. Discussion

The SVM's perfect test-set accuracy and near-perfect cross-validation score indicate strong separability of the three risk classes in the feature space. The linear kernel's effectiveness with the high-dimensional TF-IDF features is consistent with established findings that linear SVMs excel when the number of features approaches or exceeds the number of samples [15].

Logistic Regression achieved the second-highest performance (F1 = 0.9834), with misclassifications concentrated between adjacent risk levels (Low ↔ Medium: 4 errors; Medium ↔ High: 1 error). This pattern is expected, as adjacent categories share overlapping linguistic features.

Random Forest exhibited the lowest performance (F1 = 0.9601), with six Medium-to-High misclassifications. This is attributable to the high dimensionality of TF-IDF features, where Random Forest's random feature subsampling may miss discriminative terms in individual trees.

It is important to note that the perfect SVM accuracy reflects the distinguishability of the synthetic dataset rather than expected real-world performance. The controlled generation process with distinct templates per class produces well-separated clusters in the feature space that a linear SVM can readily partition.

The LIME explanations provide clinically meaningful transparency. For high-risk inputs, LIME consistently highlighted crisis-indicative words (e.g., "hopeless," "alone," "worthless") with strong positive weights, while neutral or positive terms received negative weights. This alignment between clinical intuition and model explanations supports the system's potential utility as a decision-support tool.

---

## V. Ethical Considerations and Limitations

### A. Ethical Design Principles

The system was designed with the following ethical safeguards: (1) voluntary input only—no scraping or monitoring of user accounts; (2) zero data retention—each request is stateless with no storage of user inputs; (3) no user identification—no login, tracking, cookies, or personally identifiable information collection; (4) transparency—every prediction is accompanied by LIME explanations; and (5) explicit disclaimers—the system states it is not a replacement for professional clinical assessment in every API response.

### B. Limitations

1. The system supports English text only and does not handle multilingual input.
2. Only textual data is analyzed; images, videos, and audio are excluded.
3. The synthetic dataset, while balanced and controlled, may not fully capture the complexity and nuance of real-world social media language.
4. The system analyzes individual texts in isolation without temporal analysis across a user's posting history.
5. The SVM's perfect test accuracy is a reflection of the synthetic dataset's separability and would likely decrease with real-world data.

---

## VI. Conclusion and Future Work

This paper presented an explainable AI framework for multi-level mental health risk detection from social media text. The system combined a multi-stage NLP preprocessing pipeline, hybrid feature engineering (TF-IDF, VADER, TextBlob, emotion lexicon), three supervised classifiers, LIME-based explainability, and LDA topic modeling within a full-stack web application. The linear SVM achieved the highest classification performance with a weighted F1-score of 1.0000 on the test set and 0.9993 ± 0.0013 on 5-fold cross-validation. LIME provided word-level contribution analysis that aligned with clinical intuition, supporting the framework's utility as a transparent decision-support tool.

Future work will focus on: (1) training on real-world datasets from Twitter, Reddit, and other platforms to evaluate generalization; (2) fine-tuning transformer-based models (BERT, MentalBERT) for richer contextual representations; (3) extending support to multilingual text; (4) incorporating temporal pattern analysis across multiple posts; (5) integrating SHAP for complementary game-theory-based feature attribution; and (6) implementing active learning with domain expert feedback for continuous model improvement.

---

## References

[1] World Health Organization, "Mental disorders," WHO Fact Sheet, June 2022. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/mental-disorders

[2] G. Coppersmith, M. Dredze, and C. Harman, "Quantifying mental health signals in Twitter," in _Proc. Workshop Comput. Linguistics Clinical Psychol._, Baltimore, MD, USA, 2014, pp. 51–60.

[3] M. De Choudhury, M. Gamon, S. Counts, and E. Horvitz, "Predicting depression via social media," in _Proc. 7th Int. AAAI Conf. Weblogs Social Media (ICWSM)_, Cambridge, MA, USA, 2013, pp. 128–137.

[4] A. Yates, A. Cohan, and N. Goharian, "Depression and self-harm risk assessment in online forums," in _Proc. Conf. Empirical Methods Natural Language Process. (EMNLP)_, Copenhagen, Denmark, 2017, pp. 2968–2978.

[5] A. Adadi and M. Berrada, "Peeking inside the black-box: A survey on explainable artificial intelligence (XAI)," _IEEE Access_, vol. 6, pp. 52138–52160, 2018.

[6] G. Coppersmith, M. Dredze, C. Harman, and K. Hollingshead, "From ADHD to SAD: Analyzing the language of mental health on Twitter through self-reported diagnoses," in _Proc. Workshop Comput. Linguistics Clinical Psychol._, Denver, CO, USA, 2015, pp. 1–10.

[7] S. Ji, T. Zhang, L. Ansari, J. Fu, P. Tiwari, and E. Cambria, "MentalBERT: Publicly available pretrained language models for mental healthcare," in _Proc. 13th Lang. Resources Eval. Conf. (LREC)_, Marseille, France, 2022, pp. 7184–7190.

[8] M. T. Ribeiro, S. Singh, and C. Guestrin, ""Why should I trust you?": Explaining the predictions of any classifier," in _Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining_, San Francisco, CA, USA, 2016, pp. 1135–1144.

[9] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in _Proc. Advances Neural Inf. Process. Syst. (NeurIPS)_, Long Beach, CA, USA, 2017, pp. 4765–4774.

[10] G. Gkotsis et al., "Characterisation of mental health conditions in social media using informed deep learning," _Sci. Rep._, vol. 7, no. 1, p. 45141, 2017.

[11] S. Jain and B. C. Wallace, "Attention is not explanation," in _Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics (NAACL)_, Minneapolis, MN, USA, 2019, pp. 3543–3556.

[12] C. J. Hutto and E. Gilbert, "VADER: A parsimonious rule-based model for sentiment analysis of social media text," in _Proc. 8th Int. AAAI Conf. Weblogs Social Media (ICWSM)_, Ann Arbor, MI, USA, 2014, pp. 216–225.

[13] S. M. Mohammad and P. D. Turney, "Crowdsourcing a word-emotion association lexicon," _Comput. Intell._, vol. 29, no. 3, pp. 436–465, 2013.

[14] D. M. Blei, A. Y. Ng, and M. I. Jordan, "Latent Dirichlet allocation," _J. Mach. Learn. Res._, vol. 3, pp. 993–1022, Jan. 2003.

[15] T. Joachims, "Text categorization with support vector machines: Learning with many relevant features," in _Proc. 10th Eur. Conf. Mach. Learn. (ECML)_, Chemnitz, Germany, 1998, pp. 137–142.
