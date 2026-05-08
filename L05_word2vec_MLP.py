import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# Load dataset
docs_df = pd.read_csv('reviews.csv')

# Preprocess text
def preprocess(text):
    tokens = [t for t in simple_preprocess(text) if t not in STOPWORDS]
    return tokens

docs_df['tokens'] = docs_df['text'].apply(preprocess)

# Drop empty-token rows
tokenized_docs = [tokens for tokens in docs_df['tokens'] if len(tokens) > 0]
print("number of original documents:", len(docs_df))
print("number of non-empty documents:", len(tokenized_docs))
print("tokenized documents")
print(tokenized_docs)

# Train Word2Vec model
w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=20, window=5, min_count=1, workers=4)

def vectorize(tokens, model, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

# Vectorize documents
X = np.array([vectorize(tokens, w2v_model, 20) for tokens in docs_df['tokens']])
y = docs_df['label'].values

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Replace LogisticRegression with MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(128, 64), 
                    activation='relu', 
                    max_iter=1000, 
                    early_stopping=True,
                    random_state=42)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
