import praw
import os 
from praw.models import MoreComments
import pandas as pd
from pandas import DataFrame

# server requests
reddit = praw.Reddit(client_id='U6e7Akdybwf94w',
                     client_secret='CYjRrJkldCdgdm1N_G8Nywo4s1g', 
                     password='72186996',
                     user_agent='joop', 
                     username='joopica')

# scraping with query 
posts = []
threads = []

McGill_sr = reddit.subreddit('McGill')
search = McGill_sr.search('mental health')

for post in search:
	comments = []
	thread = reddit.submission(id=post.id).comments
	thread.replace_more(limit=None)

	for top_comment in thread.list(): #reddit.submission(id=post.id).comments.list():
		comments.append(top_comment.body)
		threads.append([post.id, top_comment.body])
		# print(top_comment.body)
	posts.append([post.title, post.id, post.num_comments, post.selftext, comments])

df1 = pd.DataFrame(posts, columns=['title','id','num_comments','body','comments'])
df2 = pd.DataFrame(threads, columns=['id','comments'])

# save to CSV, change your path before saving 
# path_name "/Users/jessicachan/Desktop/AI/Bean/"
file_name = "/Users/jessicachan/Desktop/AI/Bean/reddit_scrape.csv"
file_name_comm = "/Users/jessicachan/Desktop/AI/Bean/reddit_comment_scrape.csv"
export_csv = df1.to_csv(file_name, index=True, header=True)
export_comments = df2.to_csv(file_name_comm, index=True, header=True)

# THINGS TO DO WITH THE TEXT DATA 
# lemmatization 
# convert to lower case 
# getting rid of the and to etc. #stop words in nltk, create your own stop word list!!
# word embedding? sentence embedding? BERT
