#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing using TextBlob

# TextBlob is a python library and offers a simple API to access its methods and perform basic NLP tasks

# #Setting up system
# Anaconda prompt:
# pip install -U textblob
# python -m textblob.download_corpora
1. Tokenization:
Refers to dividing text or sentence into a sequence of tokens, which roughly correspond to words.
Steps:
a. Create a textblob object and pass a string with it
b. Call functions of textblob in order to do a specific task
# In[4]:


from textblob import TextBlob
blob = TextBlob("It is a sunny day today. \n It is the month of August.")

#tokenization into sentences
blob.sentences

#extracting only first sentence
blob.sentences[0]


# In[5]:


#printing words of first sentence
for words in blob.sentences[0].words:
    print(words)


# 2. Noun Phrase Extraction
# Extracting only noun phrases.

# In[7]:


blob = TextBlob("It is a sunny day today.")
for np in blob.noun_phrases:
    print(np)
    
blob = TextBlob("Machine Learning is a fantastic way to predict future of business operations.")
for np in blob.noun_phrases:
    print(np)


# 3. Part of Speech Tagging 
# Part of speech tagging or grammatical tagging - method to mark words present in text on basis of its definition and context.
# It tells us whether a word is a noun or an adjective or a verb. 

# In[8]:


for words, tag in blob.tags:
    print(words, tag)


# 4. Words Inflection and Lemmatization 
# Inflection is a process of word formation in which characters are added to the base form of a word to express grammatical meanings. 
# Word inflection in TextBlob : Words tokenized from textblob can be easily changed into singular or plural.

# In[12]:


blob = TextBlob("Machine Learning is a fantastic way to predict future of business operations. \n It is important for engineers pursuing CS to learn the subject.")
print(blob.sentences[1].words[2])
print(blob.sentences[1].words[2].singularize())


# In[13]:


#TextBlob library also offers an in-build object known as Word. We create a word object and apply afunction directly to it.

from textblob import Word
w = Word('Platform')

w.pluralize()


# In[15]:


#We can also use tags to inflect a particular type of word

for word,pos in blob.tags:
    if pos == 'NN':
        print(word.pluralize())
        
#Words can be lemmatized using the  lemmatize function

w = Word('reaching')
w.lemmatize("v")  ## v here represents verb


# 5. N- grams
# A combination of multiple words together are called N-Grams. N-grams (N > 1) are generally more informative as compared to words, and can be used as features for langauge modeling. 
# N- grams can be easily accessed in TextBlob using the ngrams function, which returns a tuple of n successive words. 

# In[17]:


for ngram in blob.ngrams(3):
    print(ngram)


# 6. Sentiment Analysis 
# Determining the attitude or emotion of writer, i.e whether it is positive or negative or neutral. 
# The sentiment function of textblob returns two properties, polarity and subjectivity.
# 
# Polarity is a float which lies in range of [-1,1] where 1 means positive statement and -1 means a negative statement. Subjective sentences generally refer to personal opinion, emotion or judgment whereas objective refers to factual information. Subjectivity is also a float which lies in the range of of [0,1].

# In[22]:


blob = TextBlob("Google's ML crash course is a great course to learn data science ")
print(blob)
blob.sentiment

#We can see that polarity is 0.8, which means that the statement is positive and 0.75 subjectivity refers that mostly it is a public opinion and not a factual information.


# 7. Spelling Correction :
# Spelling correction is a feature which can be accessed using the correct function

# In[23]:


blob  = TextBlob("Machine Lerning is a dta science topic")
blob.correct()


# In[24]:


#we can also check list of suggested word and its confidence using the spellcheck function 
blob.words[4].spellcheck()


# 8. Creating a short summary of a text: 

# In[26]:


import random
blob = TextBlob("Analytics Vidhya is a thriving community for data driven industry. This platform allows people to know more about analytics from its articles, Q&A forum and learning paths. Also professionals and amateurs are able to sharpen their skillsets by providing a platform to participate in hackathons.")

nouns = list()

for word, tag in blob.tags:
    if tag == 'NN':
        nouns.append(word.lemmatize())

print("This text is about...")
for item in random.sample(nouns, 5):
    word = Word(item)
    print(word.pluralize())
    
    
#What we did above that we extracted out a list of nouns from the text to give a general idea to the reader about the things the text is related to.


# 9. Translation and Language Detection 

# In[30]:


blob = TextBlob("صباح الخير. ")
blob.detect_language()

#Arabic

#Translating it to English
blob.translate(from_lang = 'ar', to = 'en')

#Even if you don’t explicitly define the source language, TextBlob will automatically detect the language and translate into the desired language.
blob.translate(to = 'en')


# 10. Text Classification using TextBlob
# 

# In[38]:


training = [ 
('Tom Holland is a terrible spiderman.', 'pos'),
('a terrible Javert (Russell Crowe) ruined Les Miserables for me...','pos'),
('The Dark Knight Rises is the greatest superhero movie ever!','neg'),
('Fantastic Four should have never been made.','pos'),
('Wes Anderson is my favorite director!','neg'),
('Captain America 2 is pretty awesome.','neg'),
('Let\s pretend "Batman and Robin" never happened..','pos'),
]

testing = [
    ('Superman was never an interesting character.','pos'),
('Fantastic Mr Fox is an awesome film!','neg'),
('Dragonball Evolution is simply terrible!!','pos')
]


#Textblob provides in-build classifier module to create a custom classifier.

from textblob import classifiers

classifier = classifiers.NaiveBayesClassifier(training)

#Textblob also offers Decision Tree Classifier along with Naive Bayes Classifier

dt_classifier = classifiers.DecisionTreeClassifier(training)

#Checking the accuracy of classifier on testing dataset

print(classifier.accuracy(testing))

#Textblob also provides most informative features
classifier.show_informative_features(5)

# We can see that if the text contains “is”, then there is a high probability that the statement will be negative.

#Checking classifier on a random text

blob = TextBlob('the weather is terrible!', classifier=classifier)
print (blob.classify())


# Pros and Cons of TextBlob :
# Pros:
# Since, it is built on the shoulders of NLTK and Pattern, therefore making it simple for beginners by providing an intuitive interface to NLTK.
# It provides language translation and detection which is powered by Google Translate ( not provided with Spacy).
# Cons:
# It is little slower in the comparison to spacy but faster than NLTK. (Spacy > TextBlob > NLTK)
# It does not provide features like dependency parsing, word vectors etc. which is provided by spacy.

# In[16]:


string1 = TextBlob("Machine Learning")
string1[1:5] ##extracting 1 to 5 letters

