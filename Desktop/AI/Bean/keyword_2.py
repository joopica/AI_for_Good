import pandas as pd
import numpy as np
import re
import nltk
import os

import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from pandas import DataFrame 

#nltk.download('stopwords')
#nltk.download('wordnet')

from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import RegexpTokenizer 
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from scipy.sparse import coo_matrix

## INITIAL PROCESSING
def load_file(file_name):
	dataset = pd.read_csv(file_name)
	return dataset

def word_count(dataset,content_header,describe=False):
	dataset['word_count'] = dataset[content_header].apply(lambda x: len(str(x).split(" ")))
	dataset = dataset[[content_header,'word_count']]
	if describe is True:
		print(dataset.word_count.describe())
	return dataset

def to_corpus(data,content_header,stopwords):
	
	data.head()
	corpus = []
	for i in range(data.shape[0]):
		# remove punctuation
		text = re.sub('[^a-zA-Z]',' ',data[content_header][i])
		# convert to lowercase 
		text = text.lower()
		# remove tags 
		text = re.sub("&lk;/?&gt;", " &lt;&gt; ",text)
		#remove special characters and digits
		text = re.sub("(\\d|\\W)+"," ",text)
		#remove space after non and not
		text = text.replace('not ', 'not')
		text = text.replace('non ', 'non')
		text = text.replace('non-', 'non')
		text = text.replace('well ', 'well')
		# convert to list from string
		text = text.split()
		# stemming
		ps = PorterStemmer()
		# lemmatizing
		lem = WordNetLemmatizer()

		text = [lem.lemmatize(word) for word in text if not word in stopwords]
		text = " ".join(text)

		corpus.append(text)

	return corpus

## MOST AND LEAST COMMON
def common_n_words(dataset,content_header,n):
	freq = pd.Series(' '.join(dataset[content_header]).split()).value_counts()[:n]
	return freq

def uncommon_n_words(dataset,content_header,n):
	freq = pd.Series(' '.join(dataset[content_header]).split()).value_counts()[n:]
	return freq

## UPDATING STOPWORDS
def add_stop_words(curr_stop,new_stop):
	stopwords = curr_stop.union(new_stop)
	return stopwords

def reset_stopwords():
	stop_words = set(stopwords.words("english"))
	return stop_words


## VECTORIZING AND N TOP MGRAMS
def get_top_nwords(corpus,n=None):
	vec = CountVectorizer().fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word,sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
	words_freq = sorted(words_freq, key= lambda x: x[1], reverse=True)

	return words_freq[:n]

def get_top_n2words(corpus,n=None):
	vec = CountVectorizer(ngram_range=(2,2),max_features=2000).fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word,sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
	words_freq = sorted(words_freq, key= lambda x: x[1], reverse=True)

	return words_freq[:n]

def get_top_n3words(corpus,n=None):
	vec = CountVectorizer(ngram_range=(3,3),max_features=2000).fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word,sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
	words_freq = sorted(words_freq, key= lambda x: x[1], reverse=True)

	return words_freq[:n]

def get_top_nmwords(corpus,m=1,n=None):
	vec = CountVectorizer(ngram_range=(m,m),max_features=2000).fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word,sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
	words_freq = sorted(words_freq, key= lambda x: x[1], reverse=True)

	return words_freq[:n]


## TFIDF FEATURE EXTRACTION
def topn_from_vector(corpus, stop_words, i=1):
	cv = CountVectorizer(max_df=1,stop_words=stop_words,
		max_features=10000,ngram_range=(1,4))
	X = cv.fit_transform(corpus)

	tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
	tfidf_transformer.fit(X)

	feature_names = cv.get_feature_names()
	tfidf_vector = tfidf_transformer.transform(cv.transform([corpus[i]]))

	return tfidf_vector, feature_names


def sort_coo(coo_matrix):
	tuples = zip(coo_matrix.col, coo_matrix.data)
	return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_vector(feature_names, sorted_items, n=10):
	sorted_items = sorted_items[:n]
	score_vals = []
	feature_vals = []

	# word index of feature name and tf-idf score 
	for idx, score in sorted_items:
		score_vals.append(round(score,3))
		feature_vals.append(feature_names[idx])

	#create tuples of feature,score
	results = {}
	for idx in range(len(feature_vals)):
		results[feature_vals[idx]]=score_vals[idx]

	return results


## EXTRACTING FINAL 
def extract_n_keywords(file_name, content_header, show=False, n=10):
	data = load_file(file_name)
	data = word_count(data, content_header)
	stopwords = reset_stopwords()

	corpus = to_corpus(data,content_header,stopwords)

	keywords = []

	for idx in range(0,len(corpus)):
		corpus_cv = topn_from_vector(corpus,stopwords,idx)
		sorted_items = sort_coo(corpus_cv[0].tocoo())
		curr_keywords = extract_topn_vector(corpus_cv[1],sorted_items,n)

		if show is True:
			print("\nComment:")
			print(corpus[idx])
			print("\nKeywords:")
			for k in curr_keywords:
			    print(k,curr_keywords[k])

		keywords.append(curr_keywords)
		print(type(curr_keywords))

		# for k in curr_keywords:
		# 	keywords.append([k,curr_keywords[k]])

	return keywords

## testing
# data = load_file('services.csv')
# data = word_count(data, 'full_desc')
# common = uncommon_n_words(data,'full_desc',40)
# stopwords = reset_stopwords()
# corpus = to_corpus(data,'full_desc',stopwords)
# topn = get_top_nwords(corpus,5)
# topn2 = get_top_n2words(corpus,5)
# topn3 = get_top_nmwords(corpus,5,5)
# corpus_cv = topn_from_vector(corpus,stopwords,4)
# sorted_items = sort_coo(corpus_cv[0].tocoo())
# keywords = extract_topn_vector(corpus_cv[1],sorted_items,20)
# print(keywords)
print(extract_n_keywords('services.csv','full_desc'))

