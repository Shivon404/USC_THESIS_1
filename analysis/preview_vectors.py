# analysis/preview_vectors.py

import pandas as pd
import joblib
import os

# === STEP 0: Ensure output folder exists ===
preview_dir = 'data/previews'
os.makedirs(preview_dir, exist_ok=True)

# === STEP 1: Load Vectorized Data ===
print("📥 Loading BoW and TF-IDF matrices...")
X_bow = joblib.load('data/processed/X_bow.pkl')
X_tfidf = joblib.load('data/processed/X_tfidf.pkl')

# === STEP 2: Load Vectorizers (to get feature names) ===
print("🔤 Loading vectorizers...")
bow_vectorizer = joblib.load('data/models/bow_vectorizer.pkl')
tfidf_vectorizer = joblib.load('data/models/tfidf_vectorizer.pkl')

# === STEP 3: Load Sentiment Labels (PICKLE FORMAT) ===
print("🏷️ Loading sentiment labels...")
y = joblib.load('data/processed/y_encoded.pkl')  # Not CSV!

# === STEP 4: Convert Sparse Matrices to DataFrames ===
print("🔁 Converting matrices to readable DataFrames...")
df_bow = pd.DataFrame(X_bow.toarray(), columns=bow_vectorizer.get_feature_names_out())
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
df_y = pd.DataFrame(y, columns=["encoded_label"])

# === STEP 5: Print Samples to Terminal ===
print("\n=== 🧾 Bag of Words Sample ===")
print(df_bow.head())

print("\n=== 🧾 TF-IDF Sample ===")
print(df_tfidf.head())

print("\n=== 🏷️ Encoded Labels Sample ===")
print(df_y.head())

# === STEP 6: Save 10-row Previews to CSV ===
print(f"\n💾 Saving preview files to: {preview_dir}")
df_bow.head(10).to_csv(f'{preview_dir}/preview_bow.csv', index=False)
df_tfidf.head(10).to_csv(f'{preview_dir}/preview_tfidf.csv', index=False)
df_y.head(10).to_csv(f'{preview_dir}/preview_labels.csv', index=False)

print("✅ All previews saved in data/previews/")
