from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('stopwords')
import re
import string

class tokenizer_class(object):
    def __init__(self):
        self.snowball_stemmer = SnowballStemmer("english")
    
    def __call__(self, doc):
        doc = re.sub('[,.-:/()?{}*$#&]', ' ', doc)
        doc = ''.join(ch for ch in doc if ch not in string.punctuation)
        doc = ''.join(ch for ch in doc if ord(ch) < 128)
        doc = doc.lower()
        words = doc.split()
        words = [word for word in words if word not in text.ENGLISH_STOP_WORDS]
        
        return [
                self.snowball_stemmer.stem(word) for word in words
                ]

if __name__ == "__main__":
    
    categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
    train_data = fetch_20newsgroups(subset='train', categories=categories,shuffle=True, random_state=42)
    stop_words = text.ENGLISH_STOP_WORDS

    count_vect = CountVectorizer(tokenizer=tokenizer_class(), stop_words = stop_words, lowercase= True, min_df = 2)
    tf_transformer = TfidfTransformer(use_idf=False)
    pipeline = Pipeline ([('vectorize', count_vect), ('tf-idf', tf_transformer)])
    tfidf_transformer = pipeline.fit_transform(train_data.data)

    print ("for min_df = 2:")
    print ("number of terms: ", tfidf_transformer.shape[1])

    count_vect = CountVectorizer(tokenizer=tokenizer_class(), stop_words = stop_words, lowercase= True, min_df = 5)
    tf_transformer = TfidfTransformer(use_idf=False)
    pipeline = Pipeline ([('vectorize', count_vect), ('tf-idf', tf_transformer)])
    tfidf_transformer = pipeline.fit_transform(train_data.data)

    print ("for min_df = 5:")
    print ("number of terms: ", tfidf_transformer.shape[1])

