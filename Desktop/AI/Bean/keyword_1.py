import pandas 
import csv
import re
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

import os
from os import path 
from PIL import Image 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
from pandas import DataFrame

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from scipy.sparse import coo_matrix 

# inputt = open('reddit_comment_scrape.csv', 'rb')
# output = open('reddit_comment_scrape.csv', 'wb')

# writer = csv.writer(output)

dataset = pandas.read_csv('services.csv')
dataset.head()

dataset['word_count'] = dataset['full_desc'].apply(lambda x: len(str(x).split(" ")))
dataset[['full_desc','word_count']].head()

# for row in csv.reader(inputt):
# 	if row['word_count'] < 25:
# 		writer.writerow(row)

# inputt.close()
# output.close()

dataset.word_count.describe()

#identify common words 
freq = pandas.Series(' '.join(dataset['full_desc']).split()).value_counts()[:20]
# print("frequent words: ", freq)

#identify uncommon words 
freq1 = pandas.Series(' '.join(dataset['full_desc']).split()).value_counts()[-30:]
# print("uncommon words: ", freq1)

lem = WordNetLemmatizer()

#creating stop words 
stop_words = set(stopwords.words("english"))

#creating list of custom stopwords
new_words = ["people", "day",
	"would","know","say","one","got",
	"think","like","also","make",
	"first","thing","take","go","really","could","give"]
stop_words = stop_words.union(new_words)

## processes comments, each put in corpus
corpus = []
for i in range(0, 12):

	#remove punctuation 
	text = re.sub('[^a-zA-Z]',' ',dataset['full_desc'][i])

	#convert to lower case
	text = text.lower()

	#remove tags 
	text = re.sub("&lk;/?&gt;", " &lt;&gt; ",text)

	#remove special characters and digits
	text = re.sub("(\\d|\\W)+"," ",text)

	#remove space after not 
	text = text.replace('not ', 'not')
	text = text.replace('non ', 'non')

	#convert to list from string
	text = text.split()

	#stemming
	ps = PorterStemmer()

	#lemmatizatino 
	text = [lem.lemmatize(word) for word in text if not word in stop_words]
	text = " ".join(text)
	corpus.append(text)
	# print(text)

# print(corpus)
# print(corpus[3])

# wordcloud = WordCloud(background_color='white',
# 	stopwords=stop_words,
# 	max_words=100,
# 	max_font_size=50,
# 	random_state=42).generate(str(corpus))

# print(wordcloud)
# fig = plt.figure(1)
# plt.imshow(wordcloud)
# plt.axis('off')
# plt.show()

cv = CountVectorizer(max_df=1,stop_words=stop_words,
	max_features=10000, ngram_range=(1,3))
X = cv.fit_transform(corpus)

# print(list(cv.vocabulary_.keys())[:10])

#Most frequently occuring words 
def get_top_n_words(corpus, n=None):
	vec = CountVectorizer().fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
	words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

	return words_freq[:n]

def get_top_n2_words(corpus, n=None):
	vec = CountVectorizer(ngram_range=(2,2),max_features=2000).fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
	words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

	return words_freq[:n]

def get_top_n3_words(corpus, n=None):
	vec = CountVectorizer(ngram_range(3,3),max_features=2000).fit(corpus)
	bag_of_words = vec.transform(corpus)
	sum_words = bag_of_words.sum(axis=0)
	words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
	words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

	return words_freq[:n]

# top2_words = get_top_n2_words(corpus, n=20)
# top2_df = pandas.DataFrame(top2_words)
# top2_df.columns=["Bi-gram", "Freq"]
# print(top2_df)

# sns.set(rc={'figure.figsize':(13,8)})
# h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
# h.set_xticklabels(h.get_xticklabels(), rotation=45)
# plt.show()

#declare tfidf transformer object 
tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X)

#get feature names 
feature_names = cv.get_feature_names()

# print(feature_names)

#fetch document for which keywords needs to be extracted
doc = corpus[3]

#generate tf idf for given document
tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,20)
 

## export to csv
# df1 = pd.DataFrame(posts, columns=['title','id','num_comments','body','comments'])
# file_name = "/Users/jessicachan/Desktop/AI/Bean/reddit_scrape.csv"
# export_csv = df1.to_csv(file_name, index=True, header=True)

#now print the results
print("\nComment:")
print(doc)
print("\nKeywords:")
for k in keywords:
    print(k,keywords[k])


