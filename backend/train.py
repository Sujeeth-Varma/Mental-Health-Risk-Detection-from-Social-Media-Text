"""Training script for the Mental Health Risk Detector."""

import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASET_PATH
from utils.dataset_generator import generate_dataset
from pipeline.model_trainer import ModelTrainer
from pipeline.topic_modeler import TopicModeler
from pipeline.preprocessor import TextPreprocessor


def main():
    print("=" * 60)
    print("  Mental Health Risk Detector - Model Training")
    print("=" * 60)

    # Step 1: Generate dataset
    print("\n📁 Step 1: Generating dataset...")
    if not os.path.exists(DATASET_PATH):
        generate_dataset(n_samples_per_class=500)
    else:
        print(f"   Dataset already exists at {DATASET_PATH}")

    # Step 2: Load dataset
    print("\n📂 Step 2: Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"   Total samples: {len(df)}")
    print(f"   Label distribution:\n{df['label'].value_counts().to_string()}")

    texts = df["text"].tolist()
    labels = df["risk_code"].tolist()

    # Step 3: Train models
    print("\n🤖 Step 3: Training models...")
    trainer = ModelTrainer()
    results = trainer.train(texts, labels)

    # Step 4: Save models
    print("\n💾 Step 4: Saving models...")
    trainer.save_models()
    trainer.save_evaluation(results)

    # Step 5: Train topic model
    print("\n📚 Step 5: Training topic model...")
    preprocessor = TextPreprocessor()
    processed_texts = preprocessor.preprocess_batch(texts)
    topic_modeler = TopicModeler(n_topics=5, method="lda")
    topic_modeler.fit(processed_texts)
    topic_modeler.save()

    topics = topic_modeler.get_topics()
    for topic in topics:
        print(f"   {topic['label']}: {', '.join(topic['keywords'][:5])}")

    # Step 6: Print summary
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n  {model_name}:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1_score']:.4f}")
        print(f"    CV F1:     {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")

    print("\n✅ Training complete! You can now start the Flask server with: python app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
