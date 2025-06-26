<<<<<<< HEAD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score , precision_score , recall_score , f1_score
import joblib

# import libraries requires fro nlp
import nltk # nl toolkit
import re # regular expression

from nltk.corpus import stopwords # library for importing stopwords

#download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print(stop_words)

df = pd.read_csv("IMDB Dataset.csv")
df.head()
df.shape
df["review"].value_counts()
df["sentiment"].value_counts()
# Mapping the sentiments to some numerical value
df["sentiment"] = df["sentiment"].map({
    "positive" : 1,
    "negative" : 0
})

# clean the text
def clean_text(text) :
  text = re.sub(r"[^a-zA-Z]"," ",text).lower()
  tokens = text.split()
  tokens = [word for word in tokens if word not in stop_words]
  return " ".join(tokens)

# apply the clean text finction on review 
df["cleaned_review"] = df["review"].apply(clean_text)
df['cleaned_review']

# feature extraction
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]

# divide the dataset in to train-test-split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

# Train the model 
model = MultinomialNB()
model.fit(X_train,y_train)

# Make the prediction 
y_pred = model.predict(X_test)

#Calculate the performance metrics
accuracy = accuracy_score(y_pred,y_test)
precision = precision_score(y_pred,y_test)
recall = recall_score(y_pred,y_test)
f1 = f1_score(y_pred,y_test)
cm = confusion_matrix(y_pred,y_test)
cr = classification_report(y_pred,y_test)
print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)
print("Confusion Matrix :\n", cm)
print("Classification Report :\n", cr)

# save the model and vectorizer
joblib.dump(model,"sentiment_model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")
=======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score , precision_score , recall_score , f1_score
import joblib

# import libraries requires fro nlp
import nltk # nl toolkit
import re # regular expression

from nltk.corpus import stopwords # library for importing stopwords

#download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print(stop_words)

df = pd.read_csv("IMDB Dataset.csv")
df.head()
df.shape
df["review"].value_counts()
df["sentiment"].value_counts()
# Mapping the sentiments to some numerical value
df["sentiment"] = df["sentiment"].map({
    "positive" : 1,
    "negative" : 0
})

# clean the text
def clean_text(text) :
  text = re.sub(r"[^a-zA-Z]"," ",text).lower()
  tokens = text.split()
  tokens = [word for word in tokens if word not in stop_words]
  return " ".join(tokens)

# apply the clean text finction on review 
df["cleaned_review"] = df["review"].apply(clean_text)
df['cleaned_review']

# feature extraction
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]

# divide the dataset in to train-test-split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

# Train the model 
model = MultinomialNB()
model.fit(X_train,y_train)

# Make the prediction 
y_pred = model.predict(X_test)

#Calculate the performance metrics
accuracy = accuracy_score(y_pred,y_test)
precision = precision_score(y_pred,y_test)
recall = recall_score(y_pred,y_test)
f1 = f1_score(y_pred,y_test)
cm = confusion_matrix(y_pred,y_test)
cr = classification_report(y_pred,y_test)
print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)
print("Confusion Matrix :\n", cm)
print("Classification Report :\n", cr)

# save the model and vectorizer
joblib.dump(model,"sentiment_model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")
>>>>>>> 0667892726948feb7e3e737cb88168296c0364e0
print("Sentiment and vector model has been saved")