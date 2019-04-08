#!/usr/bin/env python
# coding: utf-8

# __Graduate Presentation__<br>
# CSCI 5448 (Object-Oriented Analysis & Design)<br>
# __Maram Kurdi__ 

# ## Natural Language Toolkit (NLTK) 

# This code will ask the user to enter a text in order to preprocess it for nay NLP tasks<br>
# **Text preprocessing include:**
# * Sentence Tokenization
# * Word Tokenization
# * Removing Stopwords
# * Removing Punctuation
# * Stemming
# * Lemmatization
# * Part of Speech Tagging
# * Frequency Distribution
# * Chunking
# * Sentiment Analysis
# 

# In[1]:


import nltk


# In[2]:


# get user text input
user_text = input("Enter text to process:\n")


# ### 1. Sentence Tokenization
# Sentence tokenizer breaks text into sentences

# In[3]:


from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(user_text)
print(sentences)
print("Number of sentences: ",len(sentences))


# ### 2. Word Tokenization
# Word tokenizer breaks sentences into words

# In[4]:


from nltk.tokenize import word_tokenize
words = word_tokenize(user_text)
print(words)
print("Number of words: ",len(words))


# ### 3. Removing Stopwords and Punctuation
# Removing Stopwords from text that don't give meaning such as (the, a, this, etc)<br>
# Removing punctuation from text such as ( .  ,  ?)

# In[5]:


from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
tokenizer = RegexpTokenizer(r'\w+')
filtered_panct = tokenizer.tokenize(user_text) # remove punctuation
stop_words = set(stopwords.words("english"))
filtered_text = [i for i in filtered_panct if not i in stop_words] # remove stopwords
print (filtered_text)


# ### 4. Stemming using PorterStemmer
# Stemming means reducing inflected words to their word stem, base or root form

# In[6]:


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed_text = [stemmer.stem(word) for word in filtered_text] # stemming words
print (stemmed_text)


# ### 5. Lemmatization
# Lemmatization is the process of removing inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma

# In[7]:


from nltk.stem.wordnet import WordNetLemmatizer
Lemmatizer = WordNetLemmatizer()
lemmatized_text = [Lemmatizer.lemmatize(word,"v") for word in filtered_text] # Lemmatizing words
print (lemmatized_text)


# ### 6. Part of Speech Tagging
# is the process of marking up a word in a text as corresponding to a particular part of speech, based on both its definition and its context

# In[8]:


Part_of_Speech = nltk.pos_tag(filtered_text)
print (Part_of_Speech)


# ### 7. Chunking 
# Chunking is a task that follows Part-Of-Speech Tagging and that adds more structure to the sentence. The result is a grouping of the words in “chunks”.

# In[9]:


reg_exp = "NP: {<DT>?<JJ>*<NN>}"
rp = nltk.RegexpParser(reg_exp)
chunked = rp.parse(Part_of_Speech)
print(chunked)
chunked.draw() 


# ### 8. Frequency Distribution
# Counting the occurrence for each word for giving text

# In[10]:


from nltk.probability import FreqDist
word_freq = FreqDist(filtered_text)
print(word_freq.most_common())


# ### 9. Sentiment Analysis
# Sentiment Analysis is the process of computationally identifying and categorizing opinions expressed in a text

# In[11]:


#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
sentiment = SentimentIntensityAnalyzer()
for sentence in sentences:
     print(sentence)
     ss = sentiment.polarity_scores(sentence)
     for k in ss:
         print("{0}: {1}, ".format(k, ss[k]), end="")
     print()

