#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Data Pre-processing

# In[1]:


pip install nltk


# In[2]:


import nltk
from nltk import FreqDist

f = open('TheSocialDilemmaDATASETTEXTONLY.csv', 'rb')
raw = f.read()
raw = raw.decode('utf8')


# In[3]:


type (raw)


# In[4]:


raw = raw.replace('\n','')


# # Tokenization

# In[5]:


nltk.download('punkt')


# In[6]:


tokens = nltk.word_tokenize(raw)
type(tokens)


# In[7]:


#change all tokens into lower case 
words1 = [w.lower() for w in tokens]   #list comprehension 

#only keep text words, no numbers 
words2 = [w for w in words1 if w.isalpha()]


# In[8]:


#generate a frequency dictionary for all tokens 
freq = FreqDist(words2)

#sort the frequency list in descending order
sorted_freq = sorted(freq.items(),key = lambda k:k[1], reverse = True)
sorted_freq


# # Remove Stop Words

# In[9]:


nltk.download('stopwords')


# In[10]:


from nltk.corpus import stopwords

stopwords = stopwords.words('english')


# In[11]:


words_nostopwords = [w for w in words2 if w not in stopwords]


# In[12]:


#generate a frequency dictionary for all tokens 
freq_nostw = FreqDist(words_nostopwords)

#sort the frequency list in decending order
sorted_freq_nostw = sorted(freq_nostw.items(),key = lambda k:k[1], reverse = True)
sorted_freq_nostw


# # Stemming

# In[13]:


# Using Porter Stemmer 
porter = nltk.PorterStemmer()
stem1 = [porter.stem(w) for w in words_nostopwords]

# The frequency distribution 
freq1 = FreqDist(stem1)
#Sort the result
sorted_freq1 = sorted(freq1.items(),key = lambda k: k[1], reverse = True)


# In[14]:


sorted_freq1


# # POS Tagging for all nouns

# In[15]:


nltk.download('averaged_perceptron_tagger')


# In[16]:


POS_tags = nltk.pos_tag(tokens) #use unprocessed 'tokens', not 'words'

#Generate a list of POS tags
POS_tag_list = [(word,tag) for (word,tag) in POS_tags if tag.startswith('N')]


# In[17]:


#Generate a frequency distribution of all the POS tags
tag_freq = nltk.FreqDist(POS_tag_list)
#Sort the result 
sorted_tag_freq = sorted(tag_freq.items(), key = lambda k:k[1], reverse = True)


# In[18]:


POS_tag_list


# In[19]:


sorted_tag_freq


# In[20]:


tag_freq.plot(30)


# ## Step 2: Sentiment Analysis

# In[21]:


# Load Dataset
import pandas as pd
# read file
df = pd.read_csv("TheSocialDilemmaDATASETTEXTONLY.csv", header = 0, names = ['Tweet'])
df.head()


# In[22]:


data = df.Tweet.str.lower()


# ## Dictionary approach

# In[23]:


df = df.assign(binary = 0)


# In[24]:


def count_pos_neg(data, positive_dict, negative_dict, binary): #lists of key words - positive_dict & negative_dict
# count of positive and negative words that appeared in each message
# net count which is calculated by positive count subtracting negative count.
# same approach can be used in different dictionaries
    poscnt = [] #the three variables, list of variables
    negcnt = []
    netcnt = []
    binary = []

    for nrow in range(0,len(data)):
        text = data[nrow] #loop through each message / paragraph / row of data 
        
        qa = 0 # total number of positive words in the message 
        qb = 0 # total number of negative key words in the message 

        for word in positive_dict : # if any words in the dictionary appear in your text 
            if (word in text) :
                qa = qa + 1 # if it appears in your data, then positive is +1

        for word in negative_dict :
            if (word in text) :
                qb = qb + 1 # if word from dictionary appears in your data, then negative is +1

        qc = qa - qb #net count of positive and negative words = count of words that appear in row
        qd = 0
        if qc > 0:
            qd = 1
        if qc == 0:
            qd = 2

        poscnt.append(qa) 
        negcnt.append(qb)
        netcnt.append(qc)
        binary.append(qd)# append list of message words

    return (poscnt, negcnt, netcnt, binary) #total number of positive, negative, and positve/negative 


# Bing Liu's Dictionary

# In[25]:


#run this line if you never download it before
import nltk
nltk.download("opinion_lexicon")


# In[26]:


#import Bing Liu's dictionary
from nltk.corpus import opinion_lexicon


# In[27]:


df_BL = df


# In[28]:


pos_list_BL=set(opinion_lexicon.positive())
neg_list_BL=set(opinion_lexicon.negative())
binary = df_BL.binary

# after this step, created positive and negative lists
# this then becomes the input of the function 


# In[29]:


df_BL['poscnt_BL'], df_BL['negcnt_BL'], df_BL['netcnt_BL'], df_BL['binary']= count_pos_neg(data, pos_list_BL, neg_list_BL, binary)

# can put the data and two lists into the created function -- now have three variables attached to the view
# number of positive key words, negative words, and net count (negative number == negative review)


# In[30]:


for word in pos_list_BL:
    if (word is data[1]) :
        print(word)


# In[31]:


df_BL['binary'].value_counts()


# In[32]:


import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Positive', 'Negative', 'Neutral'
sizes = [2507, 14460, 3101]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# Vader

# In[33]:


#run this line if you never download it before
get_ipython().system('pip install vadersentiment')


# In[34]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[35]:


df_vader = df


# In[36]:


analyzer = SentimentIntensityAnalyzer()
scores = [analyzer.polarity_scores(sentence) for sentence in data]


# In[37]:


# scores

# first three results - positive, negative and neutral results and they will also add up to 1 
# compound score = polarity on a scale of -1 to 1 


# In[38]:


neg_s = [i["neg"] for i in scores]
neu_s = [i["neu"] for i in scores]
pos_s = [i["pos"] for i in scores]
compound_s = [i["compound"] for i in scores]

# pulling the items from list and putting them into a dataframe
# Vader also gives a neutral score, which is not included into other methods


# In[39]:


df_vader['negscore_Vader'], df_vader['neuscore_Vader'], df_vader['posscore_Vader'], df_vader['compound_Vader'] = neg_s, neu_s, pos_s, compound_s


# In[40]:


df_vader[['Tweet','negscore_Vader','neuscore_Vader','posscore_Vader','compound_Vader']].head(5)


# In[41]:


# df.loc[df['c1'] >= 5, 'c2'] = 1
df_vader.loc[df_vader['compound_Vader'] > 0, 'binary'] = 1
df_vader.loc[df_vader['compound_Vader'] == 0, 'binary'] = 2


# In[42]:


df_vader[['Tweet','negscore_Vader','neuscore_Vader','posscore_Vader','compound_Vader', 'binary']].head(5)


# In[43]:


df_vader['binary'].value_counts()


# In[44]:


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Positive', 'Negative', 'Neutral'
sizes = [8102, 4773, 7193]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[45]:


# NLTK
nltk.download('vader_lexicon')

# Vader is really good on social media analysis, including emotion/emoji


# In[46]:


from nltk.sentiment import SentimentIntensityAnalyzer


# In[47]:


df_nltk = df


# In[48]:


sia = SentimentIntensityAnalyzer()
scores = [sia.polarity_scores(sentence) for sentence in data]


# In[49]:


# scores


# In[50]:


neg_s_n = [i["neg"] for i in scores]
neu_s_n = [i["neu"] for i in scores]
pos_s_n = [i["pos"] for i in scores]
compound_s_n = [i["compound"] for i in scores]


# In[51]:


df_nltk['negscore_NLTK'], df_nltk['neuscore_NLTK'], df_nltk['posscore_NLTK'], df_nltk['compound_NLTK'] = neg_s_n, neu_s_n, pos_s_n, compound_s_n


# In[52]:


df_nltk[['Tweet','negscore_NLTK','neuscore_NLTK','posscore_NLTK','compound_NLTK']].head(5)


# In[53]:


df_nltk.loc[df_nltk['compound_NLTK'] > 0, 'binary'] = 1
df_vader.loc[df_vader['compound_Vader'] == 0, 'binary'] = 2


# In[54]:


df_nltk[['Tweet','negscore_NLTK','neuscore_NLTK','posscore_NLTK','compound_NLTK', 'binary']].head(5)


# In[55]:


df_nltk['binary'].value_counts()


# In[56]:


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Positive', 'Negative', 'Neutral'
sizes = [8192, 4706, 7170]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# # Emotional analysis
# # Using Text2emotion dictionary: Happy, Angry, Sad, Surprise, and Fear; also include emojis

# In[57]:


#Install package using pip
get_ipython().system('pip install text2emotion')


# In[58]:


#Import the modules
import text2emotion as te


# In[60]:


# pull the text from text column and convert to text file
text_col = list(df['Tweet'])

# converting list into string and then joining it with space
text_data = ' '.join(str(e) for e in text_col)
  
# printing result
print(text_data)


# In[ ]:


#Call to the function
te.get_emotion(text_data)


# In[61]:


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Happy', 'Angry', 'Surprise', 'Sad', 'Fear'
sizes = [0.09, 0.04, 0.37, 0.18, 0.32]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[62]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('Happy', 'Angry', 'Surprise', 'Sad', 'Fear')
y_pos = np.arange(len(objects))
performance = [0.09, 0.04, 0.37, 0.18, 0.32]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Emotional Response')
plt.title('Detected Emotion of Tweet')

plt.show()


# # Feature extraction
# # Countvectorizer
# # TF-IDF vectorizer and n-grams

# In[63]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics


# In[64]:


df_nltk


# # Split the Data into Train-Test Sets

# In[65]:


# split to 20 percent test data and 80 percent train data
# labels can be seen as y, an dependent variable
train_corpus, test_corpus, train_labels, test_labels = train_test_split(df["Tweet"],
                                                                        df["binary"],
                                                                        test_size=0.2)


# # Construct Features for Machine Learning

# # Bag of Words

# In[66]:


# build bag of words features' vectorizer and get features
bow_vectorizer=CountVectorizer(min_df=1, ngram_range=(1,1))
bow_train_features = bow_vectorizer.fit_transform(train_corpus)
bow_test_features = bow_vectorizer.transform(test_corpus)


# In[67]:


count_array = bow_train_features.toarray()
df = pd.DataFrame(data=count_array,columns = bow_vectorizer.get_feature_names())
print(df.shape)
df.head(20)


# # TF-IDF

# In[68]:


# build tfidf features' vectorizer and get features
tfidf_vectorizer=TfidfVectorizer(min_df=1, 
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=(1,1))
tfidf_train_features = tfidf_vectorizer.fit_transform(train_corpus)  
tfidf_test_features = tfidf_vectorizer.transform(test_corpus)   


# In[69]:


count_array1 = tfidf_test_features.toarray()
df = pd.DataFrame(data=count_array1,columns = tfidf_vectorizer.get_feature_names())
print(df.shape)
df.head(20)


# In[70]:


def train_predict_evaluate_model(classifier, 
                                 train_features, train_labels, 
                                 test_features, test_labels):
    # build model. fit(x,y)    
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features) 
    # evaluate model prediction performance   
    '''get_metrics(true_labels=test_labels, 
                predicted_labels=predictions)'''
    print(metrics.classification_report(test_labels,predictions))
    print(metrics.confusion_matrix(test_labels,predictions))
    return predictions, metrics.accuracy_score(test_labels,predictions)  


# # Import Classifiers

# In[71]:


from sklearn.naive_bayes import MultinomialNB # import naive bayes
from sklearn.tree import DecisionTreeClassifier # import Decision Tree
from sklearn.ensemble import RandomForestClassifier # import random forest
from sklearn.metrics import confusion_matrix


# # Train and Test on Bag of Words Feature

# In[72]:


# assign naive bayes function to an object
mnb = MultinomialNB()
#The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
#It is primarily used to compare the performance of two classifiers. 
#Suppose that classifier A has a higher recall, and classifier B has higher precision. 
#In this case, the F1-scores for both the classifiers can be used to determine which one produces better results

#Support is the number of actual occurrences of the class in the specified dataset
# predict and evaluate naive bayes
mnb_bow_predictions, mnb_bow_accuracy = train_predict_evaluate_model(classifier=mnb,
                                           train_features=bow_train_features,
                                           train_labels=train_labels,
                                           test_features=bow_test_features,
                                           test_labels=test_labels)


# In[73]:


# assign decision tree function to an object
dt = DecisionTreeClassifier()

# predict and evaluate decision tree
dt_bow_predictions, dt_bow_accuracy = train_predict_evaluate_model(classifier=dt,
                                                               train_features=bow_train_features,
                                                               train_labels=train_labels,
                                                               test_features=bow_test_features,
                                                               test_labels=test_labels)


# In[74]:


# assign random forest function to an object
rf = RandomForestClassifier(criterion="entropy")

# predict and evaluate random forest
rf_bow_predictions, rf_bow_accuracy = train_predict_evaluate_model(classifier=rf,
                                           train_features=bow_train_features,
                                           train_labels=train_labels,
                                           test_features=bow_test_features,
                                           test_labels=test_labels)


# # Train and Test on TFIDF features

# In[75]:


# predict and evaluate naive bayes
mnb_tfidf_predictions, mnb_tfidf_accuracy = train_predict_evaluate_model(classifier=mnb,
                                           train_features=tfidf_train_features,
                                           train_labels=train_labels,
                                           test_features=tfidf_test_features,
                                           test_labels=test_labels)


# In[76]:


# predict and evaluate decision tree
dt_tfidf_predictions, dt_tfidf_accuracy = train_predict_evaluate_model(classifier=dt,
                                           train_features=tfidf_train_features,
                                           train_labels=train_labels,
                                           test_features=tfidf_test_features,
                                           test_labels=test_labels)


# In[77]:


# predict and evaluate random forest
rf_tfidf_predictions, rf_tfidf_accuracy = train_predict_evaluate_model(classifier=rf,
                                           train_features=tfidf_train_features,
                                           train_labels=train_labels,
                                           test_features=tfidf_test_features,
                                           test_labels=test_labels)


# In[78]:


rf_tfidf_accuracy


# In[79]:


dt_tfidf_accuracy


# In[80]:


# create a dictionary that stores all the accuracy information
accuracy_dict = {}
for m in ["mnb","dt","rf"]:
    accuracy_dict[m] = {}
    for f in ["bow","tfidf"]:
        exec('accuracy_dict["{}"]["{}"] = {}_{}_accuracy'.format(m, f, m, f))
        
#Accuracy Matrix
models_score = pd.DataFrame(accuracy_dict).rename(columns={"mnb":"Naive Bayes", 
                                            "dt":"Decision Tree", 
                                            "rf":"Random Forest"}, 
                                   index={"bow":"Bag-of-words", 
                                          "tfidf":"TFIDF", 
                                          "avgwv":"Word2Vec"})

models_score['Best Score'] = models_score.idxmax(axis=1)


# In[81]:


models_score


# # WordCloud

# In[82]:


from wordcloud import WordCloud, STOPWORDS
import sys, os
os.chdir(sys.path[0])


# In[83]:


wc = WordCloud(
        background_color='white',
        height = 600,
        width=400
)


# In[84]:


wc.generate(text_data)


# In[85]:


wc.to_file('wordcloud_output.png')


# In[86]:


from IPython.display import Image


# In[87]:


Image('wordcloud_output.png')

