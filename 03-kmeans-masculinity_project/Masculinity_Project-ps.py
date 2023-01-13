### Masculinity Project
# In this project, we will be investigating the way people think about masculinity
# by applying the KMeans algorithm to data from FiveThirtyEight

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


# load and examine the data
survey = pd.read_csv("masculinity.csv")
survey.shape
survey.columns
sum(survey[["q0007_0001"]]['q0007_0001'].str.count("Often"))
survey.head()

## mapping the data from str to int
# "Often" -> 4
# "Sometimes" -> 3
# "Rarely" -> 2
# "Never, but open to it" -> 1
# "Never, and not open to it" -> 0

cols_to_map = ["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004",
       "q0007_0005", "q0007_0006", "q0007_0007", "q0007_0008", "q0007_0009",
       "q0007_0010", "q0007_0011"]
for col in cols_to_map:
    survey[col] = survey[col].map({"Often":4, "Sometimes":3, "Rarely":2,
                                               "Never, but open to it":1,
                                               "Never, and not open to it":0})

print(survey["q0007_0011"].value_counts())

## plotting some of the features
# survey["q0007_0001"] and survey["q0007_0002"]

%matplotlib 

plt.scatter(survey["q0007_0001"], survey["q0007_0002"], alpha = 0.1)
plt.xlabel("Ask friend profession advice")
plt.ylabel("Ask friend personal advice")
plt.show

## Building KMeans model

# we are interested in answers to Q7 a, b, c, d, e, h, i
# first, get rid of NaNs in our data
features = ["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004",
                    "q0007_0005", "q0007_0008", "q0007_0009"]
rows_to_cluster = survey.dropna(subset = features)

# create KMeans model
classifier = KMeans(n_clusters = 2)
classifier.fit(rows_to_cluster[features])
print(classifier.cluster_centers_)

print(classifier.labels_)

# find the indices for the points, split between clusters
cluster_zero_indices = []
cluster_one_indices = []
for i in range(len(classifier.labels_)):
    if classifier.labels_[i] == 0:
        cluster_zero_indices.append(i)
    else:
        cluster_one_indices.append(i)

print(cluster_zero_indices)

# seperate data into the two clusters
cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]
cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]

# consider how education varies across clusters
#print(cluster_zero_df["educ4"].value_counts())
#print(cluster_one_df["educ4"].value_counts())
print(cluster_zero_df["educ4"].value_counts() / len(cluster_zero_df))
print(cluster_one_df["educ4"].value_counts() / len(cluster_one_df))

print(cluster_zero_df["educ3"].value_counts() / len(cluster_zero_df))
print(cluster_one_df["educ3"].value_counts() / len(cluster_one_df))

# consider how age varies across clusters
#print(cluster_zero_df["age3"].value_counts())
#print(cluster_one_df["age3"].value_counts())
print(cluster_zero_df["age3"].value_counts() / len(cluster_zero_df))
print(cluster_one_df["age3"].value_counts() / len(cluster_one_df))

# consider how kids varies across clusters
print(cluster_zero_df["kids"].value_counts() / len(cluster_zero_df))
print(cluster_one_df["kids"].value_counts() / len(cluster_one_df))
