import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("/Users/jessicachan/Desktop/AI/Bean/movie_dataset_28.csv")

features = ['keywords', 'cast', 'genres', 'director']

def combine_features(row):
	return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']

for feature in features: 
	df[feature] = df[feature].fillna(' ')
	# filling all NaNs with a blank string 
	df["combined_features"] = df.apply(combine_features, axis=1)
	#applying combined_features() method over each rows of dataframe
	#and storing the combined string in "combined_features column

cv = CountVectorizer() 
#creating new CV() object 
count_matrix = cv.fit_transform(df["combined_features"]) 
#feeding combined strings(movie contents) to CountVectorizer() object

print(df["combined_features"])

cosine_sim = cosine_similarity(count_matrix)


## helper function to get title from movie index and VV
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title] ["index"].values[0]

movie_user_likes = "Avatar"
movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(cosine_sim[movie_index]))
#accessing the row corresponding to given movie to finr all the similarity scores
#for that movie and then enumerating over it 

sorted_similar_movies = sorted(similar_movies, key=lambda
	x:x[1], reverse=True)[1:]

i=0
print("top 5 sim movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
	print(get_title_from_index(element[0]))
	i=i+1
	if i>5:
		break
