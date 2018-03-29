from sklearn.feature_extraction.text import TfidfTransformer
import project1b

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import math


#def calculate_tcicf(freq,categories, categories_per_term):
#    result = freq* math.log10(categories/)                    
#                (categories/float(1+categories_per_term)))
 #   return result

tfidf_transformer = TfidfTransformer()

stop_words = text.ENGLISH_STOP_WORDS

vectorizer = CountVectorizer(analyzer='word',stop_words=stop_words,ngram_range=(1, 1), tokenizer=project1b.tokenizer_class(),
                             lowercase=True,max_df=0.99, min_df=5) #min_df = 5


all_categories=['comp.graphics',
                'comp.os.ms-windows.misc',
                'comp.sys.ibm.pc.hardware',
                'comp.sys.mac.hardware',
                'comp.windows.x',
                'rec.autos',
                'rec.motorcycles',
                'rec.sport.baseball',
                'rec.sport.hockey',
                'alt.atheism',
                'sci.crypt',
                'sci.electronics',
                'sci.med',
                'sci.space',
                'soc.religion.christian',
                'misc.forsale',
                'talk.politics.guns',
                'talk.politics.mideast',
                'talk.politics.misc',
                'talk.religion.misc'
                ]

all_docs_per_category=[]

for cat in all_categories:
    categories=[cat]
    all_data = fetch_20newsgroups(subset='train',categories=categories).data
    temp = ""
    for doc in all_data:
        temp= temp + " "+doc
    all_docs_per_category.append(temp)

icf = vectorizer.fit_transform(all_docs_per_category)
tf_icf = tfidf_transformer.fit_transform(icf).toarray()[:]
print(tf_icf.shape)
print(len(vectorizer.get_feature_names()))

for category in [2,3,14,15]:
    tficf={}
    term_index=0;
    for term in vectorizer.get_feature_names():
        tficf[term]=tf_icf[category][term_index]
        term_index+=1
    significant_terms = dict(sorted(tficf.items(), key=lambda x:x[1],reverse=True) [:10]) #get 10 significant terms
    print (significant_terms.keys())
    print ('-' *30 )