import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
punct = string.punctuation
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(data):

    #tokenize the sentences
    words = word_tokenize(data.lower())
    
    #lemmatize each word to its lemma
    lemma_words = [lemmatizer.lemmatize(i) for i in words if i not in punct] #generally got better accuracy with stopwords
    

    return(lemma_words)