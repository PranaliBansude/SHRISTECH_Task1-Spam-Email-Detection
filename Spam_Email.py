import pandas as pd
import numpy as np
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

nltk.download('stopwords')

data = pd.read_csv("spam.csv", encoding="latin-1")

if data.shape[1] > 2:
    data = data.iloc[:, :2]

data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

print("\nDataset Loaded Successfully\n")
print(data.head())

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

data['clean_message'] = data['message'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_message'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))

new_email = input("\nEnter an email message to check spam or not:\n")

clean_new = clean_text(new_email)
new_vec = vectorizer.transform([clean_new])

if model.predict(new_vec)[0] == 1:
    print("\nResult: Spam Email")
else:
    print("\nResult: Not Spam Email")
