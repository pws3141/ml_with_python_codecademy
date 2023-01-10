import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load files
businesses = pd.read_json('yelp_business.json', lines=True)
reviews = pd.read_json('yelp_review.json', lines=True)
users = pd.read_json('yelp_user.json', lines=True)
checkins = pd.read_json('yelp_checkin.json', lines=True)
tips = pd.read_json('yelp_tip.json', lines=True)
photos = pd.read_json('yelp_photo.json', lines=True)

# adjust num of cols and chars for ease of visual
pd.options.display.max_columns = 60
pd.options.display.max_colwidth = 500

businesses.head()
reviews.head()
users.head()
checkins.head()
tips.head()
photos.head()

# examining the 'businesses' dataframe
businesses.shape
businesses.name.head()
print("There are {:d} businesses in the dataset".format(len(businesses.name)))

# examining the 'users' dataframe
users.columns

# find rating of business with id '5EvUIR4IzCWUOm0PsUZXjA'
businesses[businesses["business_id"] == "5EvUIR4IzCWUOm0PsUZXjA"]["stars"]

# merge dataframes together
# note take all dataframes have 'business_id' col, so merge on this
df = pd.merge(businesses, reviews, how='left', on='business_id')
df = pd.merge(df, users, how='left', on='business_id')
df = pd.merge(df, checkins, how='left', on='business_id')
df = pd.merge(df, tips, how='left', on='business_id')
df = pd.merge(df, photos, how='left', on='business_id')

df.columns

# drop features that are strings
# leaving continuous and binary data
features_to_remove = ['address', 'attributes', 'business_id', 'categories','city',
                      'hours', 'is_open', 'latitude', 'longitude', 'name',
                      'neighborhood', 'postal_code', 'state', 'time']
df.drop(features_to_remove, axis = 1, inplace = True)

# check to see if any cols contain NAs
df.isna().any()

# they do, so need to change this
# change all NaNs to 0
# assuming that NaNs means that Yelp had no data, e.g. no pictures
df.fillna({'weekday_checkins':0,
           'weekend_checkins':0,
           'average_tip_length':0,
           'number_tips':0,
           'average_caption_length':0,
           'number_pics':0},
          inplace=True)

# check we have removed all NaNs
df.isna().any()

# look at correlation between variables in the dataframe
df_corrs = df.corr()
df_corrs['stars']

# plotting some of the highly correlated features to 'stars'
%matplotlib
# 'average_review_sentiment'
plt.scatter(df['average_review_sentiment'], df['stars'], alpha = 0.1)
plt.xlabel("Average Review Sentiment")
plt.ylabel("Star Rating")
plt.show()

# 'average_review_length'
plt.scatter(df['average_review_length'], df['stars'], alpha = 0.1)
plt.xlabel("Average Review Length")
plt.ylabel("Star Rating")
plt.show()

# 'average_review_age'
plt.scatter(df['average_review_age'], df['stars'], alpha = 0.1)
plt.xlabel("Average Review Age")
plt.ylabel("Star Rating")
plt.show()

# and one variable that has very low corr to 'stars'
# 'number_funny_votes'
plt.scatter(df['number_funny_votes'], df['stars'], alpha = 0.1)
plt.xlabel("Number Funny Votess")
plt.ylabel("Star Rating")
plt.show()

#### Linear Regression

# on features: 'average_review_length' and 'average_review_age'
features = df[['average_review_length', 'average_review_age']]
ratings = df['stars']

# creating training and test samples
X_train, X_test, y_train, y_test = train_test_split(features, ratings, 
                                                    train_size = 0.8,
                                                    random_state = 1)
                                                 
# create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# look at R2 of this model on test set
model.score(X_test, y_test) # 0.08083081210060561
# not a good score...
# these two features alone are not able to effectively predict stars

# lets predict x_test labels using our model
y_predicted = model.predict(X_test)
# look at plot of y_predict against y_test
plt.scatter(y_predicted, y_test, alpha = 0.2)
plt.xlabel("Predicted Stars")
plt.ylabel("True Stars")
plt.show()
# plots not along y=x line: so model heteroscedastic



#### Linear Regression on some different subsets of the data

# subset of only average review sentiment
sentiment = ['average_review_sentiment']
# subset of all features that have a response range [0,1]
binary_features = ['alcohol?','has_bike_parking','takes_credit_cards',
                   'good_for_kids','take_reservations','has_wifi']
# subset of all features that vary on a greater range than [0,1]
numeric_features = ['review_count','price_range','average_caption_length',
                    'number_pics','average_review_age','average_review_length',
                    'average_review_sentiment','number_funny_votes',
                    'number_cool_votes','number_useful_votes','average_tip_length',
                    'number_tips','average_number_friends','average_days_on_yelp',
                    'average_number_fans','average_review_count',
                    'average_number_years_elite','weekday_checkins','weekend_checkins']
# all features
all_features = binary_features + numeric_features
# add your own feature subset here
feature_subset = ['number_cool_votes', 'average_number_years_elite',
                  'average_tip_length', 'number_pics']


# take a list of features to model as a parameter
def model_these_features(feature_list):# {{{

    #
    ratings = df.loc[:,'stars']
    features = df.loc[:,feature_list]

    #
    X_train, X_test, y_train, y_test = train_test_split(features, ratings,
                                                        test_size = 0.2, random_state = 1)

    # if one feature, then reshape to one column np.array
    if len(X_train.shape) < 2:
        X_train = np.array(X_train).reshape(-1,1)
        X_test = np.array(X_test).reshape(-1,1)

    #
    model = LinearRegression()
    model.fit(X_train,y_train)

    #
    print('Train Score:', model.score(X_train,y_train))
    print('Test Score:', model.score(X_test,y_test))

    # print the model features and their corresponding coefficients, from most
    # predictive to least predictive
    print(sorted(list(zip(feature_list,model.coef_)),key = lambda x: abs(x[1]),reverse=True))

    #
    y_predicted = model.predict(X_test)

    #
    plt.scatter(y_test,y_predicted, alpha = 0.1)
    plt.xlabel('Yelp Rating')
    plt.ylabel('Predicted Yelp Rating')
    plt.ylim(1,5)
    plt.show()
# }}}

# create a model on sentiment here
model_these_features(sentiment)
plt.clf()


# create a model on all binary features here
model_these_features(binary_features)
plt.clf()


# create a model on all numeric features here
model_these_features(numeric_features)
plt.clf()


# create a model on all features here
model_these_features(all_features)
plt.clf()


# create a model on your feature subset here
model_these_features(feature_subset)
plt.clf()


### Prediction
# predict rating of 'Danielle's Delicious Delicacies'
# using 'all_features' model
all_features

# retrain our model using 'all_features'
features = df.loc[:,all_features]
ratings = df.loc[:,'stars']
X_train, X_test, y_train, y_test = train_test_split(features, ratings,
                                                    test_size = 0.2, random_state = 1)
model = LinearRegression()
model.fit(X_train,y_train)

# calculate mean, minimum, and maximum values for each feature
pd.DataFrame(list(zip(features.columns,features.describe().loc['mean'], 
                      features.describe().loc['min'], 
                      features.describe().loc['max'])), 
            columns=['Feature','Mean','Min','Max'])


# lets see how our cafe does...
danielles_delicious_delicacies = np.array([1, 1, 0, 1, 0, 1, 20, 2, 5, 100, 10, 600,
                                           0.84, 30000, 3000, 20000, 70, 1000, 3000,
                                           3000, 460, 3000, 2, 6000, 10000]).reshape(1,-1)

model.predict(danielles_delicious_delicacies)
