
                                                                      AIT590-002|Term Project

                                           2020 U.S. Presidential Election Sentimental Analysis: Trump Vs. Biden Tweets

                                                Team9: Elham Jafarghomi, Saravana Vallaban, HaniehSadat Taghavi

                                                                        Instructor: Dr. Liao



########################################################################   Project Overview  ##############################################################################################################


For this Project, we applied Natural Language Processing (NLP) techniques to performed sentiment analysis on Donald Trump and Joe Biden Tweets during the election cycle using Machine Learning Algorithms.
We first optained a pre-labaled general tweets dataset, which was downloaded from Kaggle, as the training dataset. (https://www.kaggle.com/imrandude/twitter-sentiment-analysis) 
We performed a Vader sentiment analysis on our training dataset for the baseline solution.
We then trained Naive Bayes and Logistic Regression model and compared the accuracy of the results from the two models with the baseline solution.
Logistic Regression model was selected as our prefered model for predicting the sentiment of 2020 U.S. election candidates' tweets dataset, 
which was scraped from Twitter directly from Donald Trump and Joe Biden official accounts from May 1st until November 2nd. 
Using NLP tools and techniques we tried to capture insight into how their attitudes were evolving as we got closer to the election day




########################################### Description of the Files in Folder : AIT590-002-Team9-Term Project - Project ################################################################################



- AIT590-002-Team9-Term Project Code.ipynb : This file contains our project code as the working system, which was done in Jupyter NoteBook. 

- AIT590-002-Team9-Term Project Code.html : This file is the HTML format of our project code as working system, which was done in Jupyter NoteBook, with all the outputs. 

- tweet_scaping-code.ipynb :This file contains all the coding done for scraping data from Twitter.

- train.csv : This CSV file is the training set downloaded from Kaggle, which we used for training our Machine Learning Classifiers.

- Trump.csv : This CSV file contains Donald Trump's tweets scraped from Twitter, which was used as the test dataset

- Biden.csv : This CSV file contains Joe Biden's tweets scraped from Twitter, which was used as the test dataset

- trump wordcloud.png : This PNG file is a gray scale picture, used as a mask for generating WordCloud of the most frequent words used in Trump's tweets. 
(This file should be kept in the same folder with the code when running) 

- biden wordcloud.png : This PNG file is a gray scale picture, used as a mask for generating WordCloud of the most frequent words used in Biden's tweets.
(This file should be kept in the same folder with the code when running)




#######################################################################    Getting Started    ###########################################################################################################


### Environment:

Language : Python (Jupyter NoteBook)


### Installations Required:

pip install spacy
pip install twitterscraper
pip install nltk
pip install numpy
pip install matplotlib
pip install vadersentiment
pip install plotly


### Packages and Libraries Required:


import string               
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import spacy
nltk.download("vader_lexicon")
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import WordNetLemmatizer   
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer         
from sklearn.naive_bayes import MultinomialNB                     
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB              
from sklearn import linear_model as lm
from PIL import Image



####################################################################   Working System Overview    ##############################################################################################



###### Our sentiment analysis will involve folowing steps:

1. Train dataset explaratory analysis

2. Train dataset preprocessing

3. Performing Baseline Solution using Vader on train dataset

4. Model building and evaluating models performances

5. Test datasets explaratory analysis

6. Test datasets preprocessing

7. Predidicting the sentiment on test datasets using the best performing model

6. Analysis of results and visualizations




##### Section 1| Train Dataset 


- After installing and importing the rquired packages, the train dataset is imported as 'traindf', and explanatory analysis on has been done on this pre-labeled general tweets dataset.



## Data Preprocessing 


- for preprocessing, we performed the followings:

      Eliminate and handle URLs.
      Eliminate and handle @mentions.
      Tokenize the string into words.
      Remove stop words like “and, is, a, on, etc.”
      Lematizing- or convert every word to its stem. Like a dancer, dancing, danced, becomes ‘dance’.
      Convert all your words to lower case.


- In our data cleaning process we found that it works much better to first tokenize the data before text cleaning in our case.


- A function to tokenize the text column is defined:


def tokenize_text(textdata):
    textdata= textdata.lower()
    textdata = word_tokenize(textdata)
    return textdata


- A function to lemmatize the words and get the root of words  is defined:


lemmatizer = WordNetLemmatizer()

def lemma(textdata):
    textdata = [lemmatizer.lemmatize(token, pos="v") for token in textdata]
    return textdata


- Afterwards stopwords are removed using spacy; stopwords are customized such that some words that are not written correctly such as mi instead of my and etc, 
are added to stopwords for rwmoval and some words are eliminated since the number of spacy stopwordfs are almost twice of the nltk stopwords


STOP_WORDS |= {"mi", "gon", "na", "cause", "atleast", "wiiit", "doin"}
STOP_WORDS -= {"less", "serious", "against", "enough", "never", "few", "more", "most", "mostly", "together", "very", 
              "many", "former", "say", "behind", "again", "beyond", "least"}
def remove_stopwords(textdata):
    textdata = [token for token in textdata if token not in STOP_WORDS]
    return textdata



- Next we remove the punctuations by applying a defined function to not only remove punctuations from “string.punctuations”, but also remove the digits, re-tweets (RT), URLs and  @mentions. 
- We then removed all words with less than two characters that were remaining in our datasets.
 
punctuations = string.punctuation     
 
def remove_punctuation(textdata):
   
    textdata = re.sub("(@[A-Za-z0-9]+)"," ", textdata)
    textdata = re.sub("([^0-9A-Za-z \t])"," ", textdata)
    textdata =re.sub(r'’https?:\/\/(www\.)?[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)', " ", textdata)
    textdata = re.sub('[0-9]+', '', textdata)
    textdata = re.sub(r'^rt[\s]+','', textdata)  # removes RT
    textdata = textdata.lower()
    textdata = "".join([char for char in textdata if char not in punctuations])

    return textdata





- we also Got the length of tokens in each tokenized tweet (after removing stopwords) and Checked to make sure we have at least one token left after preprocessing and only keeped those rows



- Word Cloud of positive and negative sentiments are build after preprocessing




### Sentiment Analysis Using Vader

- We tried a baseline solution accuray on the training dataset 

- Then we Decide sentiment as positive, negative and neutral base on the compound score from Vader sentiment

- We Checked the vaderSentiment prediction with sentiment of the data (pre-labeled); the level of accuracy



### Model training Process

## TF-IDF Vectorization

- Because our traindata and test data are two different datasets, and number of features are not the same, we use maximum number of features that we can use 
in order to be able to run our model on our test datasets

tfidfvec = TfidfVectorizer(max_features = 2000, ngram_range=(1,2)).fit(traindf.cleanTweets)




## Train the Logistic Regression Model


y = d.Sentiment_binary
X = d.drop('Sentiment_binary', axis=1)

- split the data to train and test for predicting sentiment of text:

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=123)

- Call the logistic regression model for training:

log_reg = LogisticRegression(max_iter=4000).fit(X_train, y_train)

- predict the sentiment on the text data:

y_pred = log_reg.predict(X_test)



## Train the Multinomial Naive Bayes Model


- split the data to train and test for predicting sentiment of text:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.20, random_state =123)

- Defining the model; will filt Multinominal NB model:

model = MultinomialNB()
model.fit(X_train, y_train)

model.score(X_test, y_test)

- predict on test data:

y_predict = model.predict(X_test)



## Train the Bernoulli Naive Bayes Model


- split the data to train and test for predicting sentiment of text:

X_trainb, X_testb, y_trainb, y_testb = train_test_split(X, y, test_size =0.20, random_state =123)

- Call the bernoulli model:

Bmodel = BernoulliNB().fit(X_trainb,y_trainb)

- Predict on test data:


############################################################################################################################################################################################################

#### Section 2| Test Datasets 

- In this section, we read Donald Trump and Joe Biden tweets dataset that were scrapped from their twitter accounts
- we do all cleaning steps that we did for the general train data above.
- We then do the sentiment analysis using our best performing model that we trained on a general pre-labeled tweet dataset which was logistic regression.


## Donald Trump Tweets data

- For building a wordcloud of the most frequent words used in Trump's tweets, first import the "trump wordcloud.png".
- "trump wordcloud.png" is used as a mask for the wordcloud.



# Predicting Trump tweet's polarity using the Logistic regression model

- Convert the cleanTweet tokens to clean text for being used in vectorization

- TF-IDF Vectorization is done on trump dataset:

tfidfvec1 = TfidfVectorizer(max_features = 2000, ngram_range=(1,2)).fit(trumpdf.cleanTweets)

V1 = tfidfvec1.transform(trumpdf.cleanTweets)

transformed_df1 = pd.DataFrame(V1.toarray(), columns=tfidfvec1.get_feature_names())
d1 = trumpdf.join(transformed_df1)




- Call the logistic regression model on trump data to predict the sentimnets of his tweets and store them on the predicted sentiment column:

- After predicting the sentiments of Donald Trump tweets, visualizations are presented for better underestanding and analyzing the results




## Joe Biden Tweets Data


- For building a wordcloud of the most frequent words used in Biden's tweets, first import the "Biden wordcloud.png".
- "Biden wordcloud.png" is used as a mask for the wordcloud.



# predicting Biden tweet's polarity using the Logistic regression model


- Convert the cleanTweet tokens to clean text for being used in vectorization

- TF-IDF Vectorization is done on biden dataset:

tfidfvec2 = TfidfVectorizer(max_features = 2000, ngram_range=(1,2)).fit(bidendf.cleanTweets)

V2 = tfidfvec2.transform(bidendf.cleanTweets)

transformed_df2 = pd.DataFrame(V2.toarray(), columns=tfidfvec2.get_feature_names())
d2 = bidendf.join(transformed_df2)


- Call the logistic regression model on Biden data to predict the sentimnets of his tweets and store them on the predicted sentiment column:

- After predicting the sentiments of Joe Biden tweets, visualizations are presented for better underestanding and analyzing the results



### Visualizations

- Biden Tweets Sentiment Ratio Per each Month is ploted

- Trump Tweets Sentiment Ratio Per each Month is plotted

- Both candidates Overal tweets Sentiment comparison is plotted

- Presidential Candidates Positive Sentiment Changes Up to the Election Day is plotted


