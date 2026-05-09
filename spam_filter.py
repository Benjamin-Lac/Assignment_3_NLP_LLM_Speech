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

# download nltk stuff
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# preprocess the text - lowercase, remove special chars, stem and lemmatize
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(stemmer.stem(t)) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

# load the dataset
df = pd.read_csv('emails.csv')
print(f"loaded {len(df)} emails, {df['spam'].sum()} spam and {(df['spam']==0).sum()} ham")

print("preprocessing emails...")
df['processed'] = df['text'].apply(preprocess)

# build the DTM using CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
DTM = vectorizer.fit_transform(df['processed'])
DTM_dense = DTM.toarray()
print("DTM shape (documents x terms):", DTM_dense.shape)
print("TDM shape (terms x documents):", DTM_dense.T.shape)

y = df['spam'].values
indices = np.arange(len(y))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

# -------------------------
# Approach 1 - Row Vectors (documents)
# each row in the DTM represents one email as a vector of term frequencies
# we train the MLP directly on these to classify spam vs ham
# -------------------------
print("\n--- Approach 1: Row Vectors (Document-based) ---")
print(f"training on {len(train_idx)} samples, testing on {len(test_idx)} samples")

X_train_row = DTM_dense[train_idx]
X_test_row = DTM_dense[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

mlp_row = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
mlp_row.fit(X_train_row, y_train)
y_pred_row = mlp_row.predict(X_test_row)
acc_row = accuracy_score(y_test, y_pred_row)

print(f"Accuracy (row vectors): {acc_row*100:.2f}%")
print(classification_report(y_test, y_pred_row, target_names=['Ham', 'Spam']))

# -------------------------
# Approach 2 - Column Vectors (terms)
# transpose the DTM to get the TDM
# each row now represents a term across all documents
# we label each term as spam or ham based on which type of doc it appears in more
# then use those predictions to score documents
# -------------------------
print("\n--- Approach 2: Column Vectors (Term-based) ---")
TDM = DTM_dense.T  # (n_terms, n_docs)
print("TDM shape:", TDM.shape)

spam_mask = (y == 1)
ham_mask = (y == 0)
spam_freq = TDM[:, spam_mask].sum(axis=1)
ham_freq = TDM[:, ham_mask].sum(axis=1)
y_terms = (spam_freq > ham_freq).astype(int)
print(f"spam terms: {y_terms.sum()}, ham terms: {(y_terms==0).sum()}")

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
    random_state=42
)
mlp_col.fit(TDM[tt_train], y_terms[tt_train])

term_acc = accuracy_score(y_terms[tt_test], mlp_col.predict(TDM[tt_test]))
print(f"term-level accuracy: {term_acc*100:.2f}%")

# use term spam probabilities to score each document
term_spam_prob = mlp_col.predict_proba(TDM)[:, 1]
doc_lengths = DTM_dense.sum(axis=1, keepdims=True) + 1e-10
DTM_norm = DTM_dense / doc_lengths
doc_spam_score = DTM_norm @ term_spam_prob
y_pred_col = (doc_spam_score > 0.5).astype(int)

acc_col = accuracy_score(y[test_idx], y_pred_col[test_idx])
print(f"\nAccuracy (column vectors): {acc_col*100:.2f}%")
print(classification_report(y[test_idx], y_pred_col[test_idx], target_names=['Ham', 'Spam']))

# summary
print("\n--- Summary ---")
print(f"Row vectors accuracy: {acc_row*100:.2f}%")
print(f"Column vectors accuracy: {acc_col*100:.2f}%")
winner = "Row" if acc_row >= acc_col else "Column"
print(f"{winner} vectors performed better")
