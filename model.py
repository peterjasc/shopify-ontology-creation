import pandas as pd
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import os

XLSX_FILE_PATH = "Tooted.xlsx"
COLLECTIONS_FILE_PATH = "collections.txt"
TITLE_COLUMN_HEADER = "Title"
COLLECTION_COLUMN_HEADER = "Collection"

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./bert_collection_model"

def load_data(xlsx_path, collections_path, title_col, collection_col):
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
  return df, collections_list
 except FileNotFoundError as e:
  print(f"Error: File not found - {e}")
  return None, None
 except Exception as e:
  print(f"Error loading data: {e}")
  return None, None

def train_bert_model():
 print("--- Starting BERT Model Training (Conceptual) ---")
 print("Loading data...")
 df, collections_list = load_data(XLSX_FILE_PATH, COLLECTIONS_FILE_PATH, TITLE_COLUMN_HEADER,
          COLLECTION_COLUMN_HEADER)
 if df is None or collections_list is None:
  print("Failed to load data. Exiting training.")
  return
 label_encoder = LabelEncoder()
 df['label'] = label_encoder.fit_transform(df[COLLECTION_COLUMN_HEADER])
 id_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}
 print(f"Detected {len(label_encoder.classes_)} unique collections for training: {label_encoder.classes_}")
 tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
 def tokenize_function(examples):
  return tokenizer(examples[TITLE_COLUMN_HEADER], truncation=True, padding=True, max_length=128)
 from datasets import Dataset
 dataset = Dataset.from_pandas(df)
 tokenized_datasets = dataset.map(tokenize_function, batched=True,
          remove_columns=[TITLE_COLUMN_HEADER, COLLECTION_COLUMN_HEADER,
              '__index_level_0__'])
 tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
 train_test_split_ratio = 0.8
 train_dataset, eval_dataset = tokenized_datasets.train_test_split(test_size=1 - train_test_split_ratio).values()
 print(f"Training data size: {len(train_dataset)}")
 print(f"Validation data size: {len(eval_dataset)}")
 model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_encoder.classes_))
 training_args = TrainingArguments(
  output_dir=OUTPUT_DIR,
  num_train_epochs=3,
  per_device_train_batch_size=8,
  per_device_eval_batch_size=8,
  warmup_steps=500,
  weight_decay=0.01,
  logging_dir='./logs',
  logging_steps=100,
  eval_strategy="epoch",
  save_strategy="epoch",
  load_best_model_at_end=True,
  metric_for_best_model="loss",
  report_to="none"
 )
 trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_dataset,
  eval_dataset=eval_dataset,
  tokenizer=tokenizer,
 )
 print("\n--- Starting actual training (this will take time and resources) ---")
 trainer.train()
 if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)
 tokenizer.save_pretrained(OUTPUT_DIR)
 model.save_pretrained(OUTPUT_DIR)
 with open(os.path.join(OUTPUT_DIR, "label_classes.txt"), "w", encoding="utf-8") as f:
  for cls in label_encoder.classes_:
   f.write(f"{cls}\n")
 print(f"\n--- Model, tokenizer, and label classes saved to: {OUTPUT_DIR} ---")
 print("Training process conceptually complete.")
 print("To run actual training, uncomment 'trainer.train()' and ensure you have a suitable environment.")

if __name__ == "__main__":
 train_bert_model()