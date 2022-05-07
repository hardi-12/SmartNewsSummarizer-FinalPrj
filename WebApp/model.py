import tkinter as tk
import nltk
from textblob import TextBlob
from newspaper import Article
from gtts import gTTS 
from playsound import playsound
import pickle
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
import warnings
import string
from string import punctuation
from statistics import mean
from heapq import nlargest

stop_words = set(stopwords.words('english'))
punctuation = punctuation + '\n' + '—' + '“' + ',' + '”' + '‘' + '-' + '’'
warnings.filterwarnings('ignore')

dataset = pd.read_csv("BBC News Train.csv")
dataset.head()
dataset.info()
dataset['Category'].value_counts()
# Associate Category names with numerical index and save it in new column CategoryId
target_category = dataset['Category'].unique()
print(target_category)
dataset['CategoryId'] = dataset['Category'].factorize()[0]
dataset.head()
# Create a new pandas dataframe "category", which only has unique Categories, also sorting this list in order of CategoryId values
category = dataset[['Category', 'CategoryId']].drop_duplicates().sort_values('CategoryId')
category

x = dataset['Text']
y = dataset['CategoryId']

from sklearn.feature_extraction.text import CountVectorizer
x = np.array(dataset.iloc[:,0].values)
y = np.array(dataset.CategoryId.values)
cv = CountVectorizer(max_features = 5000)
x = cv.fit_transform(dataset.Text).toarray()
print("X.shape = ",x.shape)
print("y.shape = ",y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0, shuffle = True)

def run_model(model_name, est_c, est_pnlty):
    mdl=''

    if model_name == 'Logistic Regression':

        mdl = LogisticRegression()

    elif model_name == 'Random Forest':

        mdl = RandomForestClassifier(n_estimators=100 ,criterion='entropy' , random_state=0)

    elif model_name == 'Multinomial Naive Bayes':

        mdl = MultinomialNB(alpha=1.0,fit_prior=True)

    elif model_name == 'Support Vector Classifer':

        mdl = SVC()

    elif model_name == 'Decision Tree Classifier':

        mdl = DecisionTreeClassifier()

    elif model_name == 'K Nearest Neighbour':

        mdl = KNeighborsClassifier(n_neighbors=10 , metric= 'minkowski' , p = 4)

    elif model_name == 'Gaussian Naive Bayes':

        mdl = GaussianNB()

    oneVsRest = OneVsRestClassifier(mdl)

    oneVsRest.fit(x_train, y_train)

    y_pred = oneVsRest.predict(x_test)

    # Performance metrics

    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

    # Get precision, recall, f1 scores

    precision, recall, f1score, support = score(y_test, y_pred, average='micro')

    print(f'Test Accuracy Score of Basic {model_name}: % {accuracy}')

    print(f'Precision : {precision}')

    print(f'Recall : {recall}')

    print(f'F1-score : {f1score}')

    # Add performance parameters to list

    perform_list.append(dict([

    ('Model', model_name),

    ('Test Accuracy', round(accuracy, 2)),

    ('Precision', round(precision, 2)),

    ('Recall', round(recall, 2)),

    ('F1', round(f1score, 2))

    ]))

def summarize(url):
    article=Article(url)
    article.download()
    article.parse()
    article.nlp()
    analysis=TextBlob(article.text)
    if(analysis.polarity>0):
          sentiment="Positive"
    elif(analysis.polatity<0):
          sentiment="Negative"
    else:
          sentiment="Neutral"
    #sentiment='Polarity:{analysis.polarity},Sentiment:{"positive" if analysis.polarity>0 else "negative" if analysis.polarity<0 else "neutral" }'
    classifier = RandomForestClassifier(n_estimators=100 ,criterion='entropy' , random_state=0).fit(x_train, y_train)
    classifier
    y_pred = classifier.predict(x_test)
    y_pred1 = cv.transform([article.text])
    yy = classifier.predict(y_pred1)
    result = ""
    if yy == [0]:
      result = "Business News"
    elif yy == [1]:
      result = "Tech News"
    elif yy == [2]:
      result = "Politics News"
    elif yy == [3]:
      result = "Sports News"
    elif yy == [1]:
      result = "Entertainment News"
    speech=gTTS(text=article.summary,lang='en',slow="False")
    speech.save('static/summary.mp3')
    response={
        'title':article.title,
        'author':article.authors,
        'publication_date':article.publish_date,
        'summary':article.summary,
        'textClass':result,
        'sentiment':sentiment,
    }
    return response

    
    