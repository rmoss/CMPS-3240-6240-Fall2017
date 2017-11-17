import numpy as np
import pandas as pd
import re

#***DATA READER*** Takes 3 CSV files and cleans data so that the data items are in the form of learning examples(a two dimensional array of samples, each row corresponds to a sample (patient) and each column corresponds to a feature (gene expression).
testfile='data_set_ALL_AML_independent.csv'
trainfile='data_set_ALL_AML_train.csv'
patient_cancer='actual.csv'

#import data sets (train and test samples with gene expression values + cancer type labels)
raw_train = pd.read_csv(trainfile)
raw_test = pd.read_csv(testfile)
patient_cancer = pd.read_csv(patient_cancer)

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


