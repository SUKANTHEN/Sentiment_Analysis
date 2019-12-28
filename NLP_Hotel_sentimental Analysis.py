# Restaurant Reviews Sentimental Analysis

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#import dataset
data = pd.read_csv('Hotel_sentiments.tsv', delimiter = '\t')

# Data exploration
data.shape
data.size # No missing values were found
data.head(10)

#cleaning the text
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
corpus = []
for i in range(0,1001):
    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Try Fitting Naive Bayes to the Training data
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

