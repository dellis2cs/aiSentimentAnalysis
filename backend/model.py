import pandas as pd
import nltk
import re
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Preprocessing function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Load the new dataset
new_data = pd.read_csv('fullDataset.csv', encoding='latin-1', header=None)

# Map polarity column: 0 -> negative (0), 4 -> positive (2)
new_data['sentiment'] = new_data[0].map({0: 0, 4: 2})

# Drop rows with missing sentiment values
new_data.dropna(subset=['sentiment'], inplace=True)

# Keep only relevant columns (tweet text and sentiment)
new_data = new_data[[5, 'sentiment']]  # Text is in column 5
new_data.columns = ['text', 'sentiment']  # Rename columns for clarity

# Preprocess text
new_data['text'] = new_data['text'].apply(clean_text)

# Combine datasets (if using an additional dataset for neutral sentiments)
data = pd.read_csv('train.csv', encoding='latin-1', header=None)
data.columns = ['ID', 'text', 'selected_text', 'sentiment', 'time', 'age', 'country', 'population', 'area', 'density']
data = data[['text', 'sentiment']]

# Drop missing values
data.dropna(inplace=True)

# Map sentiment labels in the existing dataset
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
data['sentiment'] = data['sentiment'].map(sentiment_mapping)

# Preprocess text in the existing dataset
data['text'] = data['text'].apply(clean_text)

# Combine datasets
combined_data = pd.concat([data, new_data], ignore_index=True)

# Drop rows with missing values in combined dataset
combined_data.dropna(subset=['text', 'sentiment'], inplace=True)

# Shuffle the combined data
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and labels
X = combined_data['text']
y = combined_data['sentiment']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# Evaluate Model
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Hyperparameter Tuning
params = {'C': [0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(max_iter=500), params, cv=5)
grid.fit(X_train_tfidf, y_train)

print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_

# Save the model and vectorizer
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Pipeline complete. Model and vectorizer saved.")
