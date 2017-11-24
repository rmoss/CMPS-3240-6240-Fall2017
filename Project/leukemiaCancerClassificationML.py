import numpy as np
import pandas as pd
import re

#***DATA READER*** Takes 3 CSV files and cleans data so that the data items are in the form of learning examples(a two dimensional array of samples, each row corresponds to a sample (patient) and each column corresponds to a feature (gene expression).
testfile='data_set_ALL_AML_independent.csv'
trainfile='data_set_ALL_AML_train.csv'
ct_target='actual.csv'

#import data sets (train and test samples with gene expression values + cancer type labels)
raw_train = pd.read_csv(trainfile)
raw_test = pd.read_csv(testfile)
ct_target = pd.read_csv(ct_target)

# Remove "call" columns not needed from training and test dataframes
train_keep = [col for col in raw_train.columns if "call" not in col]
test_keep = [col for col in raw_test.columns if "call" not in col]

train = raw_train[train_keep].set_index("Gene Accession Number")
test = raw_test[test_keep].set_index("Gene Accession Number")

# Transpose the columns and rows so that genes become features and rows become observations
train = train.T
test = test.T

# removing chip endogenous controls (not informative for cancer classification)
train_keep = [col for col in train if not re.match("^AFFX", col)]
test_keep = [col for col in test if not re.match("^AFFX", col)]

train = train[train_keep]
test = test[test_keep]

# clean the column and index names
train = train.drop(["Gene Description"])
test = test.drop(["Gene Description"])

train.index.names = ['Patient Samples']
train.columns.names = ['Gene Accession Number']
test.index.names = ['Patient Samples']
test.columns.names = ['Gene Accession Number']

# Print head to show data reader works
print(train.head())
print(ct_target.head())


# ***TRAIN AND TEST***
# Apply a machine learning algorithm on the gene expression data and evaluate its performance

# Make a test/train/evaluation split of the data.  First combine train and test data into one dataframe
train = train.append(test)

# For initial test, just using first 50 columns of gene expression data as selected features (X).  Target labelled file is assigned to y, then scikit train_test_split used to manage data 
X = train#.iloc[:,:50]
y = ct_target['cancer']
from sklearn.model_selection import train_test_split
# to make the experiment replicable we use a constant random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=30, random_state = 253)

# Use a standard ML algorithm on the training set. Using k nearest neighbors from sci-kit learn 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Evaluate the model with the test set.  Using an initial evaluation measure of accuracy. Print accuracy to the screen.
p = knn.predict(X_test)
accuracy = sum(p == y_test)/len(y_test)
print("Accuracy: " + str(accuracy))

