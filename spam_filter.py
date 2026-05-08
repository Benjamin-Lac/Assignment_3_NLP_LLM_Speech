import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(stemmer.stem(t))
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return ' '.join(tokens)


# ── Load dataset ──────────────────────────────────────────────
df = pd.read_csv('emails.csv')
print(f"Dataset: {len(df)} emails — {df['spam'].sum()} spam, {(df['spam']==0).sum()} ham")

print("Preprocessing text (stemming + lemmatization)...")
df['processed'] = df['text'].apply(preprocess)

# ── Build Document-Term Matrix (DTM) ─────────────────────────
#   Rows = documents, Columns = unique terms
vectorizer = CountVectorizer(max_features=5000)
DTM = vectorizer.fit_transform(df['processed'])
DTM_dense = DTM.toarray()
print(f"DTM shape  (documents × terms): {DTM_dense.shape}")
print(f"TDM shape  (terms × documents): {DTM_dense.T.shape}")

y = df['spam'].values
indices = np.arange(len(y))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

# ─────────────────────────────────────────────────────────────
# Approach 1: Row Vectors — Document-based classification
#
# Each ROW of the DTM is a single email's term-frequency vector.
# The MLP takes a (1 × n_terms) vector and predicts spam/ham.
# This is the standard Bag-of-Words classification setup.
#
# MLP Architecture:
#   Input  : 5000 features (one per vocabulary term)
#   Hidden : 256 neurons (ReLU)  →  128 neurons (ReLU)
#   Output : 2 classes (ham / spam)
#   Solver : Adam, lr = 0.001, early stopping on 10% validation split
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Approach 1: Row Vectors  (Document-based Classification)")
print(f"  Feature matrix: {len(train_idx)} train / {len(test_idx)} test samples")
print(f"  Each sample = 1 document vector of length {DTM_dense.shape[1]}")
print("="*60)

X_train_row = DTM_dense[train_idx]
X_test_row  = DTM_dense[test_idx]
y_train     = y[train_idx]
y_test      = y[test_idx]

mlp_row = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
)
mlp_row.fit(X_train_row, y_train)
y_pred_row = mlp_row.predict(X_test_row)
acc_row = accuracy_score(y_test, y_pred_row)

print(f"Accuracy (row vectors): {acc_row*100:.2f}%")
print(classification_report(y_test, y_pred_row, target_names=['Ham', 'Spam']))

# ─────────────────────────────────────────────────────────────
# Approach 2: Column Vectors — Term-based classification
#
# Each COLUMN of the DTM (= each ROW of the TDM) is a single
# term's frequency profile across all documents.
# We assign a spam label to each term:
#   label = 1 if the term appears more in spam docs, else 0.
# An MLP is trained to classify terms as spam-associated or not.
#
# To produce a document-level spam score we then compute, for
# each document, the weighted average spam probability of the
# terms it contains (weighted by their term frequencies).
#
# MLP Architecture: identical to Approach 1 for fair comparison.
#   Input  : n_docs features (doc-frequency profile of the term)
#   Hidden : 256 neurons (ReLU)  →  128 neurons (ReLU)
#   Output : 2 classes (ham-term / spam-term)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Approach 2: Column Vectors  (Term-based Classification)")
TDM = DTM_dense.T          # shape: (n_terms, n_docs)
print(f"  TDM shape: {TDM.shape}")
print("="*60)

spam_mask = (y == 1)
ham_mask  = (y == 0)
spam_freq = TDM[:, spam_mask].sum(axis=1)
ham_freq  = TDM[:, ham_mask].sum(axis=1)
y_terms   = (spam_freq > ham_freq).astype(int)
print(f"  Spam-associated terms: {y_terms.sum()}, Ham-associated terms: {(y_terms==0).sum()}")

t_idx = np.arange(len(y_terms))
tt_train, tt_test = train_test_split(t_idx, test_size=0.2, random_state=42)

mlp_col = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
)
mlp_col.fit(TDM[tt_train], y_terms[tt_train])

term_acc = accuracy_score(y_terms[tt_test], mlp_col.predict(TDM[tt_test]))
print(f"  Term-level classifier accuracy: {term_acc*100:.2f}%")

# Infer document labels: weight each term's spam-probability by its
# normalized frequency in the document, then threshold at 0.5.
term_spam_prob = mlp_col.predict_proba(TDM)[:, 1]   # (n_terms,)
doc_lengths    = DTM_dense.sum(axis=1, keepdims=True) + 1e-10
DTM_norm       = DTM_dense / doc_lengths              # (n_docs, n_terms)
doc_spam_score = DTM_norm @ term_spam_prob            # (n_docs,)
y_pred_col     = (doc_spam_score > 0.5).astype(int)

acc_col = accuracy_score(y[test_idx], y_pred_col[test_idx])
print(f"\nDocument-level Accuracy (column vectors): {acc_col*100:.2f}%")
print(classification_report(y[test_idx], y_pred_col[test_idx], target_names=['Ham', 'Spam']))

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  Row vectors (document-based) accuracy : {acc_row*100:.2f}%")
print(f"  Column vectors (term-based) accuracy  : {acc_col*100:.2f}%")
winner = "Row" if acc_row >= acc_col else "Column"
print(f"  {winner} vectors perform better.")
