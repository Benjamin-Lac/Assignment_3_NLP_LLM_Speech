# Import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (replace or expand with your own reviews or read from a file)
reviews = [
    "I loved the movie! It was fantastic and thrilling.",
    "Absolutely terrible. The worst movie I’ve ever seen.",
    "The film was okay, but not great.",
    "An excellent movie with a wonderful story and characters.",
    "I hated it. So boring and predictable.",
    "Really good movie! The acting was amazing.",
    "Not bad, but it could have been better.",
    "Awful experience, waste of time.",
    "The plot was interesting and the cast did a great job.",
    "It was disappointing, I expected much more."
]

# Labels: 1 = Positive, 0 = Negative
labels = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]

# Step 1: Convert text data into a bag-of-words representation
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)
print(X)
print(X.shape)
print("dense representation:")
x_matrix = X.toarray()
print(x_matrix)
print((vectorizer.vocabulary_))
# Step 2: Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Step 3: Define and train the MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(50,),   # One hidden layer with 50 neurons
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

mlp.fit(X_train, y_train)

# Step 4: Evaluate model performance
y_pred = mlp.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 5: Try predicting new reviews
new_reviews = [
    "What an awesome film! I really enjoyed it.",
    "The movie was dull and a complete mess."
]
new_features = vectorizer.transform(new_reviews)
predictions = mlp.predict(new_features)

for review, label in zip(new_reviews, predictions):
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")
