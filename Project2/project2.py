from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix 
from sklearn.metrics.cluster import homogeneity_score 
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from scipy.sparse import linalg 
from sklearn.decomposition import PCA
from numpy.linalg import svd
import matplotlib.pyplot as plt 
from sklearn.decomposition import NMF
import pylab
from sklearn.preprocessing import normalize
from sklearn.preprocessing import FunctionTransformer

import nltk
nltk.download('stopwords')
import re
import string


def build_tfidf(all_data, train_data, test_data, count_vect, tf_transformer): 
    pipeline = Pipeline ([
                            ('vectorize', count_vect), 
                            ('tf-idf', tf_transformer)
                        ])
    tfidf_transformer = pipeline.fit_transform(all_data.data)
    tfidf_test = pipeline.transform(test_data.data)
    print ("dimension of tfidf: ", tfidf_transformer.shape)
    return tfidf_transformer, tfidf_test

def perform_kmeans(tfidf_transformer, k):
    km = KMeans(n_clusters=k)#, init='k-means++', max_iter=100, n_init=1)
    km = km.fit(tfidf_transformer)
    return km

def plot_percent_variance(tfidf_transformer):
    svd = TruncatedSVD(n_components=1000) #change to 1000
    svd.fit_transform(tfidf_transformer)
    ratio = svd.explained_variance_ratio_
    singular_values = svd.singular_values_            #where is this used?

    #retained value 
    sum_arr = [] 
    for k in range(1,1001): #change to 1001
        sum = 0.0
        for i in range(k):
            sum = sum + ratio[i]
        sum_arr.append(sum) 

    x_values = range(1,1001)

    plt.plot(x_values, sum_arr)
    plt.xlabel('principal component r')
    plt.ylabel('percent variance')
    plt.show()

def append_measures (actual, predicted, hom, com, vmeas, rand, mut):
    hom.append(homogeneity_score(actual, predicted))
    com.append(completeness_score(actual, predicted))
    vmeas.append(v_measure_score(actual, predicted))
    rand.append(adjusted_rand_score(actual, predicted))
    mut.append(adjusted_mutual_info_score(actual, predicted))

def plot_measures_with_r_svd(tfidf_transformer, all_target_group):

    hom = []
    com = []
    vmeas = []
    rand = []
    mut = []
    r_arr = [1,2,3,5,10,20,50,100,300]
    svd = TruncatedSVD(n_components=1000)
    svd_data = svd.fit_transform(tfidf_transformer)

    for r in r_arr:
        km = perform_kmeans(svd_data[:,:r], 2)        #?
        print ('confusion matrix for r as', r)              
        print (confusion_matrix(all_target_group, km.labels_))
        append_measures(all_target_group, km.labels_, hom, com, vmeas, rand, mut)
    plt.plot(r_arr, hom, 'r', label='homogeneity')
    plt.plot(r_arr, com, 'g', label='completeness')
    plt.plot(r_arr, vmeas, 'b', label='v_measure')
    plt.plot(r_arr, rand, 'y', label='adjusted_rand')
    plt.plot(r_arr, mut, 'k', label='adjusted_mutual_info')
    plt.legend();
    plt.xlabel('principal component r with svd')
    plt.ylabel('measures')
    plt.title('SVD')
    plt.show()

def plot_measures_with_r_nmf(tfidf_transformer, all_target_group):
    hom = []
    com = []
    vmeas = []
    rand = []
    mut = []
    r_arr = [1,2,3,5,10,20,50,100,300]

   # nmf = NMF(n_components=1000) #how to do svd once and disregard unimportant components?
   # nmf_data = nmf.fit_transform(tfidf_transformer)

    for r in r_arr:
        nmf = NMF(n_components=r, max_iter = (50 if r == 300 else 200)) #how to do svd once and disregard unimportant components?
        nmf_data = nmf.fit_transform(tfidf_transformer)

        km = perform_kmeans(nmf_data[:,:r], 2)
        print ('confusion matrix for r as', r)
        print (confusion_matrix(all_target_group, km.labels_))
        append_measures(all_target_group, km.labels_, hom, com, vmeas, rand, mut)
    plt.plot(r_arr, hom, 'r', label='homogeneity')
    plt.plot(r_arr, com, 'g', label='completeness')
    plt.plot(r_arr, vmeas, 'b', label='v_measure')
    plt.plot(r_arr, rand, 'y', label='adjusted_rand')
    plt.plot(r_arr, mut, 'k', label='adjusted_mutual_info')
    plt.xlabel('principal component r with nmf')
    plt.ylabel('measures')   
    plt.title('NMF') 
    plt.legend()
    plt.show()

def all_categories_svd(tfidf_transformer, all_target_group):
    
    hom = []
    com = []
    vmeas = []
    rand = []
    mut = []
    r_arr = [1,2,3,5,10,20,50,100,300]
    svd = TruncatedSVD(n_components=1000, algorithm = 'arpack', random_state =42)
    svd_data = svd.fit_transform(tfidf_transformer)

    for r in r_arr:
        km = perform_kmeans(svd_data[:,:r], 20)        #?
        print ('confusion matrix for r as', r)              
        print (confusion_matrix(all_target_group, km.labels_))
        append_measures(all_target_group, km.labels_, hom, com, vmeas, rand, mut)
    plt.plot(r_arr, hom, 'r', label='homogeneity')
    plt.plot(r_arr, com, 'g', label='completeness')
    plt.plot(r_arr, vmeas, 'b', label='v_measure')
    plt.plot(r_arr, rand, 'y', label='adjusted_rand')
    plt.plot(r_arr, mut, 'k', label='adjusted_mutual_info')
    plt.legend()
    plt.xlabel('principal component r with svd')
    plt.ylabel('measures')
    plt.title('SVD')
    plt.show()

    #find the optimal dimension, and try different k(number of clusters)

    #try different transformations of the obtained features 

def all_categories_nmf(tfidf_transformer, all_target_group):
    hom = []
    com = []
    vmeas = []
    rand = []
    mut = []
    r_arr = [1,2,3,5,10,20,50,100,300]

   # nmf = NMF(n_components=1000) #how to do svd once and disregard unimportant components?
   # nmf_data = nmf.fit_transform(tfidf_transformer)

    for r in r_arr:
        nmf = NMF(n_components=r, max_iter = (50 if r == 300 else 200)) #how to do svd once and disregard unimportant components?
        nmf_data = nmf.fit_transform(tfidf_transformer)

        km = perform_kmeans(nmf_data[:,:r],20)
        print ('confusion matrix for r as', r)
        print (confusion_matrix(all_target_group, km.labels_))
        append_measures(all_target_group, km.labels_, hom, com, vmeas, rand, mut)
    plt.plot(r_arr, hom, 'r', label='homogeneity')
    plt.plot(r_arr, com, 'g', label='completeness')
    plt.plot(r_arr, vmeas, 'b', label='v_measure')
    plt.plot(r_arr, rand, 'y', label='adjusted_rand')
    plt.plot(r_arr, mut, 'k', label='adjusted_mutual_info')
    plt.xlabel('principal component r with nmf')
    plt.ylabel('measures')  
    plt.title('NMF')  
    plt.legend()
    plt.show()

    #find the optimal dimension, and try different k(number of clusters)

     #try different transformations of the obtained features 



def print_five_measures (target, predicted):
    print ('homogeneity score:')
    print (homogeneity_score(target, predicted))

    print ('completeness score:')
    print (completeness_score(target, predicted))

    print ('V-measure:')
    print (v_measure_score(target, predicted))

    print ('adjusted rand score:')
    print (adjusted_rand_score(target, predicted))

    print ('adjuted mutual info score:')
    print (adjusted_mutual_info_score(target, predicted))


def print_five_measures (target, predicted):
    print ('homogeneity score:')
    print (homogeneity_score(target, predicted))

    print ('completeness score:')
    print (completeness_score(target, predicted))

    print ('V-measure:')
    print (v_measure_score(target, predicted))

    print ('adjusted rand score:')
    print (adjusted_rand_score(target, predicted))

    print ('adjuted mutual info score:')
    print (adjusted_mutual_info_score(target, predicted))

def visualizePerformance(tfidf_transformer, target, k, n): #target, num_cluster, n_component
    print("svd at its best score r =",n)
    svd = TruncatedSVD(n_components=n) #best score is 2
    svd_data = svd.fit_transform(tfidf_transformer)
    km = KMeans(n_clusters=k, n_init = 30, random_state=42).fit_predict(svd_data)
    pca = PCA(n_components=n).fit_transform(svd_data)
    plt.scatter(pca[:,0], pca[:,1], c = km)
    plt.show()

    print("nmf at its best score r = ",n)
    nmf = NMF(n_components = n, init = 'random', random_state = 42) #best score is 3 from the contingency matrix [1][1]
    nmf_data = nmf.fit_transform(tfidf_transformer)
    km = KMeans(n_clusters=k, n_init = 30, random_state=42).fit_predict(nmf_data)
    plt.scatter(nmf_data[:,0], nmf_data[:,1], c = km)
    plt.show()

    print ('part b --------')
    # svd normalzed
    print('svd normalized')

    svd_norm = normalize(svd_data) 
    kmeans = KMeans(n_clusters=k, n_init=30)
    km = kmeans.fit_predict(svd_norm)
    pca = PCA(n_components=n).fit_transform(svd_norm)
    plt.scatter(pca[:,0], pca[:,1], c = km)
    plt.show()
    print('contingency matrix: ')
    print(confusion_matrix(target, kmeans.labels_))
    print_five_measures(target, kmeans.labels_)

    print('nmf normalzed')

    nmf_norm = normalize(nmf_data) 
    kmeans = KMeans(n_clusters=k, n_init=30)
    km = kmeans.fit_predict(nmf_norm)
    plt.scatter(nmf_norm[:,0], nmf_norm[:,1], c = km)
    plt.show()
    print('contingency matrix: ')
    print(confusion_matrix(target, kmeans.labels_))
    print_five_measures(target, kmeans.labels_)


#2nd bullet
    print('applying log transformation after NMF hear:')

    logTransform = FunctionTransformer(np.log1p) #(log10, log2, log1p, emath.log) => only log1p works lol
    nmf_log = logTransform.transform(nmf_data)
    kmeans = KMeans(n_clusters=k, n_init=30)
    km = kmeans.fit_predict(nmf_log)
    plt.scatter(nmf_log[:,0], nmf_log[:,1], c = km)
    plt.show()   
    print('contingency matrix: ')
    print(confusion_matrix(target, kmeans.labels_))
    print_five_measures(target, kmeans.labels_)


# 3rd bullet 
    print('log then norm')

    nmf_log = logTransform.transform(nmf_data)
    nmf_log_norm = normalize(nmf_log)
    kmeans = KMeans(n_clusters=k, n_init=30)
    km = kmeans.fit_predict(nmf_norm)
    plt.scatter(nmf_norm[:,0], nmf_norm[:,1], c = km)
    plt.show()
    print('contingency matrix: ')
    print(confusion_matrix(target, kmeans.labels_))
    print_five_measures(target, kmeans.labels_)

    print('norm then log')

    nmf_norm = normalize(nmf_data)
    nmf_norm_log = logTransform.transform(nmf_norm)
    kmeans = KMeans(n_clusters=k, n_init=30)
    km = kmeans.fit_predict(nmf_norm)
    plt.scatter(nmf_norm[:,0], nmf_norm[:,1], c = km)
    plt.show()
    print('contingency matrix: ')
    print(confusion_matrix(target, kmeans.labels_))
    print_five_measures(target, kmeans.labels_)


if __name__ == "__main__":


    categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
    train_data = fetch_20newsgroups(subset='train', categories=categories,shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=categories,shuffle=True, random_state=42)
    all_data = fetch_20newsgroups(subset='all', categories=categories,shuffle=True, random_state=42)
    stop_words = text.ENGLISH_STOP_WORDS
    count_vect = CountVectorizer(stop_words = stop_words, lowercase= True, min_df = 3)
    tf_transformer = TfidfTransformer(use_idf=False)

    #group the subclasses into 2 superclasses
    all_target_group = [ int(x / 4) for x in all_data.target]

    print ('question 1: =============')
    
    vectorizer = TfidfVectorizer(
                                 min_df=3, stop_words=stop_words,
                                 )
    tfidf_transformer = vectorizer.fit_transform(all_data.data)
    print ("dimension of tfidf: ", tfidf_transformer.shape)

    print ('question 2: =============')

    km = KMeans(n_clusters=2, max_iter = 1000, random_state=42).fit(tfidf_transformer)
    
    print ('contingency matrix:')
    print (confusion_matrix(all_target_group, km.labels_))
    print_five_measures(all_target_group, km.labels_)

    print ('question 3: ============')

    print ('part i: ')

    plot_percent_variance(tfidf_transformer)

    print ('part ii:')

    print ('for SVD')
    plot_measures_with_r_svd(tfidf_transformer, all_target_group)

    print ('for NMF')
    plot_measures_with_r_nmf(tfidf_transformer, all_target_group)

    print ('question 4: ============')
    print ('part a --------')
    visualizePerformance(tfidf_transformer, all_target_group, 2,2) #n_cluster, n_component 

    print ('question 5: ============')

    all_data_all_cat = fetch_20newsgroups(subset='all',shuffle=True, random_state=42)

    all_target_all_cat = all_data_all_cat.target #should i group them first? 

    vectorizer_all_cat = TfidfVectorizer(
                                 min_df=3, stop_words=stop_words
                                 )
    tfidf_transformer_all = vectorizer_all_cat.fit_transform(all_data_all_cat.data)

    all_categories_svd(tfidf_transformer_all, all_target_all_cat)

    all_categories_nmf(tfidf_transformer_all, all_target_all_cat)

    #transformations for 20 clusters 

    visualizePerformance(tfidf_transformer_all, all_target_all_cat, 20,10) #target, n_cluster, n_component 


