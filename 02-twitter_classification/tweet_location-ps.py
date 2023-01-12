### Tweet classifying:
#  use a Naive Bayes Classifier to find patterns in real tweets
# predict whether a sentence came from New York, London, or Paris.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion matrix


## examine the data
# new york
new_york_tweets = pd.read_json("new_york.json", lines=True)
print(len(new_york_tweets))
print(new_york_tweets.columns)
print(new_york_tweets.loc[12]["text"])
# london
london_tweets = pd.read_json("london.json", lines=True)
print(len(london_tweets))
print(london_tweets.columns)
print(london_tweets.loc[12]["text"])
# paris
paris_tweets = pd.read_json("paris.json", lines=True)
print(len(paris_tweets))
print(paris_tweets.columns)
print(paris_tweets.loc[12]["text"])

# create list of all text of tweets
# and label new_york = 0, london = 1, paris = 2
new_york_text = new_york_tweets["text"].tolist()
london_text = london_tweets["text"].tolist()
paris_text = paris_tweets["text"].tolist()

all_tweets = new_york_text + london_text + paris_text
labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)

# create training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(all_tweets,
                                                                    labels,
                                                                    train_size = 0.8,
                                                                    random_state = 1)

# To use a Naive Bayes Classifier, we need to transform our lists of words into
# count vectors

counter = CountVectorizer()
counter.fit(train_data)
train_count = counter.transform(train_data)
test_count = counter.transform(test_data)

print("train_count[3]")
print(train_count[3])
print("test_count[3]")
print(test_count[3])

# use the CountVectors to train and test the Naive Bayes Classifier

classifier = MultinomialNB()
classifier.fit(train_count, train_labels)
predictions = classifier.predict(test_count)

## check quality of our model
# using 'accuracy_score'
# i.e. percentage of tweets in the test set that the classifier correctly classified
accuracy_score(test_labels, predictions)
# using confusion matrix
confusion_matrix(test_labels, predictions)


## test my own tweet
tweet = "I can't believe I missed the Queen's speach this Christmas! I was playing the piano instead!"
tweet_counts = counter.transform([tweet])
tweet_predict = classifier.predict(tweet_counts)
print(tweet_predict) # predicts incorrectly!
