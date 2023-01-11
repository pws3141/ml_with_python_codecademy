### Viral Tweets project
# use the K-Nearest Neighbor algorithm to predict whether a tweet will go viral

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


all_tweets = pd.read_json("random_tweets.json", lines=True)

print(len(all_tweets))
print(all_tweets.columns)
print(all_tweets.loc[0]['text'])

#Print the user here and the user's location here.
print(all_tweets.loc[0]["user"].keys())
all_tweets.loc[0]["user"]["location"]

# insert new column
# label tweet as viral if more than median retweets
median_retweets = np.median(all_tweets["retweet_count"])
all_tweets["is_viral"] = np.where(all_tweets['retweet_count'] >= median_retweets, 1, 0)

# print number of viral and non-viral tweets
all_tweets['is_viral'].value_counts()

# create new column for tweet length
all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']),
                                              axis=1)
# create new column 'followers_count'
all_tweets["followers_count"] = all_tweets.apply(lambda tweet:
                                                 tweet['user']['followers_count'],
                                                 axis = 1)
# create new column 'friends_count'
all_tweets["friends_count"] = all_tweets.apply(lambda tweet:
                                               tweet['user']['friends_count'],
                                               axis = 1)


# split data into labels and features
# just considering three features
labels = all_tweets["is_viral"]
data = all_tweets[["tweet_length", "followers_count", "friends_count"]]
# normalise the columns of the data
# 'axis = 0' scales columns
scaled_data = scale(data, axis = 0)

# creating train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(scaled_data, labels,
                                                                    train_size = 0.8,
                                                                    random_state = 1)

# create KNN model with k = 5
classifier = KNeighborsClassifier(n_neighbors = 5)
# train on our data
classifier.fit(train_data, train_labels)
# lets look at the score
classifier.score(test_data, test_labels)

# try k = 1, ..., 200
scores = []
k = list(range(1, 201))

for i in k:
    classifier = KNeighborsClassifier(n_neighbors = i)
    classifier.fit(train_data, train_labels)
    scores.append(classifier.score(test_data, test_labels))

%matplotlib

plt.plot(k, scores, "g-")
plt.xlabel("Number of neighbours, k")
plt.ylabel("Score")
plt.show()
