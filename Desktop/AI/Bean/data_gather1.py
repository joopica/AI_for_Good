import pandas as pd
from rake_nltk import Rake 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer 

df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')

df = df[['Title','Genre','Actors','Plot']]
df.head()

df['Key_words'] = ""

for index, row in df.iterrows():
	plot = row['Plot']

	# instantiating Rake by default it uses english stopwords
	# from nltk and discards all punctuation characters as well 
	r = Rake()

	# extracting the words by passing the text
	r.extract_keywords_from_text(plot)

	# getting the dictionary with key words as keys scores as values
	key_words_dict_scores = r.get_word_degrees()

	#assigning the key words to the new column for corr movie
	row['Key_words'] = list(key_words_dict_scores.keys())

# dropping the Plot column 
df.drop(columns = ['Plot'], inplace = True)

count = CountVectorizer()
count_matrix = count.fit_transform(df['Key_words'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df.index)

def recs(title, cosine_sim = cosine_sim):
	recommended_movies = []
	idx = indices[indices == title].index[0]
	score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

	top10 = list(score_series.iloc[1:11].index)

	for i in top10:
		recommended_movies.append(list(df.index)[i])

	return recommended_movies