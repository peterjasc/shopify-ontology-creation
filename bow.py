import pandas as pd
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys

XLSX_FILE_PATH = "Tooted.xlsx"
COLLECTIONS_FILE_PATH = "collections.txt"
TITLE_COLUMN_HEADER = "Long PD"
COLLECTION_COLUMN_HEADER = "Collection"
MODEL_DIR = "./bow_collection_model"
TARGET_EXAMPLES_PER_LABEL = 100

def load_data(xlsx_path, collections_path, title_col, collection_col, target_per_label_size):
    try:
        df = pd.read_excel(xlsx_path)
        df = df.dropna(subset=[title_col, collection_col])
        df[title_col] = df[title_col].astype(str)
        df[collection_col] = df[collection_col].astype(str)
        with open(collections_path, 'r', encoding='utf-8') as f:
            collections_list = [line.strip() for line in f if line.strip()]
        if not collections_list:
            raise ValueError(f"No collections found in {collections_path}. Cannot train model.")
        df = df[df[collection_col].isin(collections_list)]
        if df.empty:
            raise ValueError("No valid training data found after filtering by collections list.")
        print(f"Initial data size: {len(df)} examples across {df[collection_col].nunique()} unique collections.")
        augmented_dfs = []
        for collection_name, group in df.groupby(collection_col):
            current_count = len(group)
            if current_count < target_per_label_size:
                print(f"  Augmenting collection '{collection_name}': {current_count} -> {target_per_label_size} examples (by duplication).")
                num_duplicates_needed = (target_per_label_size // current_count) + 1
                augmented_group = pd.concat([group] * num_duplicates_needed, ignore_index=True)
                augmented_group = augmented_group.sample(n=target_per_label_size, replace=True, random_state=42).reset_index(drop=True)
                augmented_dfs.append(augmented_group)
            else:
                print(f"  Collection '{collection_name}' has {current_count} examples (sufficient). Sampling to {target_per_label_size}.")
                augmented_dfs.append(group.sample(n=target_per_label_size, random_state=42).reset_index(drop=True))
        df_final_augmented = pd.concat(augmented_dfs, ignore_index=True)
        print(f"Final augmented total data size: {len(df_final_augmented)} examples.")
        return df_final_augmented, collections_list
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def train_bow_model():
    print("--- Starting Bag-of-Words Model Training ---")
    print("Loading data...")
    df, collections_list = load_data(XLSX_FILE_PATH, COLLECTIONS_FILE_PATH, TITLE_COLUMN_HEADER, COLLECTION_COLUMN_HEADER, TARGET_EXAMPLES_PER_LABEL)
    if df is None or collections_list is None:
        print("Failed to load data. Exiting training.")
        return
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df[COLLECTION_COLUMN_HEADER])
    print(f"Detected {len(label_encoder.classes_)} unique collections for training: {label_encoder.classes_}")
    X_train, X_test, y_train, y_test = train_test_split(
        df[TITLE_COLUMN_HEADER], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"Training data size: {len(X_train)}")
    print(f"Validation data size: {len(X_test)}")
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("TF-IDF Vectorizer fitted.")
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    print("Logistic Regression model trained.")
    accuracy = model.score(X_test_vec, y_test)
    print(f"Model accuracy on validation set: {accuracy:.4f}")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    joblib.dump(model, os.path.join(MODEL_DIR, "logistic_regression_model.joblib"))
    print(f"\n--- Model components (vectorizer, encoder, model) saved to: {MODEL_DIR} ---")
    print("Training process complete.")

if __name__ == "__main__":
    train_bow_model()