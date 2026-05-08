import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ============================================================
# 1. Load Dataset
# ============================================================
df = pd.read_csv('emails.csv')
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total emails: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nClass distribution:\n{df['spam'].value_counts().rename({0: 'Ham (0)', 1: 'Spam (1)'})}")

# ============================================================
# 2. Preprocessing
# ============================================================
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Lowercase
    text = str(text).lower()
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenize (simple whitespace split)
    tokens = text.split()
    # Remove stop words and short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    # Stemming
    tokens = [stemmer.stem(t) for t in tokens]
    # Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

print("\nPreprocessing emails...")
df['processed'] = df['text'].apply(preprocess)
print("Preprocessing complete.")
print(f"\nSample original:   {df['text'].iloc[0][:80]}...")
print(f"Sample processed:  {df['processed'].iloc[0][:80]}...")

# ============================================================
# 3. Build DTM using CountVectorizer
# ============================================================
# max_features limits vocabulary to top 5000 most frequent terms
vectorizer = CountVectorizer(max_features=5000, min_df=2)
DTM = vectorizer.fit_transform(df['processed'])  # shape: (n_docs, n_terms)
labels = df['spam'].values

print("\n" + "=" * 60)
print("DOCUMENT-TERM MATRIX (DTM)")
print("=" * 60)
print(f"DTM shape: {DTM.shape}  (rows=documents, cols=terms)")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Sample terms: {list(vectorizer.vocabulary_.keys())[:10]}")

# ============================================================
# 4. APPROACH 1 — Row Vectors (Document feature vectors)
#    Each row of DTM = one email encoded as bag-of-words
#    Task: classify each email as spam (1) or ham (0)
# ============================================================
print("\n" + "=" * 60)
print("APPROACH 1: ROW VECTORS (Documents as feature vectors)")
print("=" * 60)
print(f"  Input X shape: {DTM.shape}  →  each sample = one email")
print(f"  Labels y shape: {labels.shape}")

X_row = DTM
y_row = labels

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_row, y_row, test_size=0.2, random_state=42, stratify=y_row
)

# MLP Architecture:
#   Input layer  : 5000 neurons (one per vocabulary term)
#   Hidden layer 1: 256 neurons, ReLU activation
#   Hidden layer 2: 128 neurons, ReLU activation
#   Hidden layer 3: 64  neurons, ReLU activation
#   Output layer : 2 neurons (spam / ham)
# Optimizer: Adam  |  Learning rate: 0.001  |  Max iterations: 300
mlp_row = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15
)

print("\nTraining MLP on row vectors...")
mlp_row.fit(X_train_r, y_train_r)
y_pred_r = mlp_row.predict(X_test_r)
acc_row = accuracy_score(y_test_r, y_pred_r)

print("\n--- MLP Architecture (Row / Document Vectors) ---")
print("  Input(5000) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Output(2, Softmax)")
print("  Optimizer: Adam | Learning rate: 0.001 | Max iterations: 300")
print(f"  Early stopping: enabled (patience=15)")
print(f"\nAccuracy (Row vectors): {acc_row * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test_r, y_pred_r, target_names=['Ham', 'Spam']))

# ============================================================
# 5. APPROACH 2 — Column Vectors (Term feature vectors)
#    TDM = DTM.T  →  each ROW of TDM = one term's frequency
#    profile across all documents.
#    Task: classify each term as spam-associated (1) or ham-associated (0)
#    based on which class the term appears more frequently in.
# ============================================================
print("\n" + "=" * 60)
print("APPROACH 2: COLUMN VECTORS (Terms as feature vectors)")
print("=" * 60)

# Transpose to get TDM: shape (n_terms, n_docs)
TDM = DTM.T.toarray()  # dense array for MLP input
print(f"  TDM shape: {TDM.shape}  →  each sample = one term")

# Derive per-term labels:
#   For each term, sum its frequencies across spam emails and ham emails.
#   If spam_freq > ham_freq (proportionally) → label = 1 (spam-associated)
#   Otherwise → label = 0 (ham-associated)
n_spam = labels.sum()
n_ham  = len(labels) - n_spam

spam_mask = labels == 1
ham_mask  = labels == 0

# Proportional frequency per term in each class
spam_freq = TDM[:, spam_mask].sum(axis=1) / (n_spam + 1e-9)
ham_freq  = TDM[:, ham_mask].sum(axis=1)  / (n_ham  + 1e-9)
term_labels = (spam_freq > ham_freq).astype(int)

print(f"  Spam-associated terms: {term_labels.sum()} / {len(term_labels)}")
print(f"  Ham-associated  terms: {(term_labels == 0).sum()} / {len(term_labels)}")

X_col = TDM       # shape: (n_terms, n_docs)
y_col = term_labels

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_col, y_col, test_size=0.2, random_state=42, stratify=y_col
)

# MLP Architecture:
#   Input layer  : n_docs neurons (one per document in corpus)
#   Hidden layer 1: 256 neurons, ReLU activation
#   Hidden layer 2: 128 neurons, ReLU activation
#   Hidden layer 3: 64  neurons, ReLU activation
#   Output layer : 2 neurons (ham-associated / spam-associated)
# Optimizer: Adam  |  Learning rate: 0.001  |  Max iterations: 300
mlp_col = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15
)

print("\nTraining MLP on column vectors...")
mlp_col.fit(X_train_c, y_train_c)
y_pred_c = mlp_col.predict(X_test_c)
acc_col = accuracy_score(y_test_c, y_pred_c)

print(f"\n--- MLP Architecture (Column / Term Vectors) ---")
print(f"  Input({TDM.shape[1]}) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Output(2, Softmax)")
print("  Optimizer: Adam | Learning rate: 0.001 | Max iterations: 300")
print(f"  Early stopping: enabled (patience=15)")
print(f"\nAccuracy (Column vectors): {acc_col * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_c, target_names=['Ham-assoc', 'Spam-assoc']))

# ============================================================
# 6. Comparison & Discussion
# ============================================================
print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON")
print("=" * 60)
print(f"  Row vectors  (document-level spam detection): {acc_row * 100:.2f}%")
print(f"  Column vectors (term-level spam association): {acc_col * 100:.2f}%")

winner = "Row vectors" if acc_row >= acc_col else "Column vectors"
print(f"\n  Better performance: {winner}")

