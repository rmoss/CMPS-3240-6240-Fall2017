import numpy as np
import pandas as pd
import re
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, precision_score
import matplotlib.pyplot as plt
from ggplot import *
import itertools

# FUNCTION plot_confusion_matrix: for plotting the confusion matrices
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """
    Plots the confusion matrix. Modified verison from 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    Inputs: 
        cm: confusion matrix
        title: Title of plot
    """
    classes=["AML", "ALL"]    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.bone)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    thresh = cm.mean()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]), 
                 horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black")

# FUNCTION basic_metrics: to calculate and print basic metrics for each model
def basic_metrics(preds, y_test_labels):
  metrics.accuracy_score(preds, y_test_labels)
  accu_dtc = accuracy_score(y_test_labels, preds, normalize=True)
  prec_dtc_all = precision_score(y_test_labels, preds, pos_label="ALL")
  prec_dtc_aml = precision_score(y_test_labels, preds, pos_label="AML")
  print("Accuracy:%.2f" %accu_dtc)
  print("Precision ALL: %.2f" %prec_dtc_all)
  print("Precision AML: %.2f \n" %prec_dtc_aml)
  cfmatrix = confusion_matrix(y_true=y_test_labels, y_pred=preds)
  plt.subplot(121)
  plot_confusion_matrix(cfmatrix, title="Confusion Matrix")
  plt.show(block=True)

################################################################################
#***DATA READER*** Takes 3 CSV files and cleans data so that the data items are
# in the form of learning examples(a two dimensional array of samples, each row
# corresponds to a sample (patient) and each column corresponds to a feature
# (gene expression).
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

################################################################################
# ***TRAIN AND TEST***
# Apply a machine learning algorithm on the gene expression data and evaluate
# its performance.

# Makes a test/train/evaluation split of the data.  First combine train and test
# data into one dataframe
train = train.append(test)

# For initial test, just used first 50 columns of gene expression data as
# selected features (X).  Target labeled file is assigned to y, then scikit
# train_test_split used to manage data 
#X = train#.iloc[:,:50]
#y = ct_target['cancer']

#Normalize the data
train_norm = (train - train.min()) / (train.max() - train.min())

# Tested a threshold for var, not as good as PCA
# normalized = train_norm.loc[:, train_norm.var()>.065]

# Normalized combined train and test data (train_norm) assigned to X. 'Cancer'
# column of target labeled file (ct_target) assigned to y.  
X = train_norm
y = ct_target['cancer']

#Reduce dimensionality of data to 50 with PCA Analysis
sklearn_pca = sklearnPCA(n_components=50)
X_sklearn = sklearn_pca.fit_transform(X)

# Split the test/train data using 50 (~70% for training). Use a constant random
# state to make the experiment replicable.
X_train, X_test, y_train, y_test = train_test_split(X_sklearn, y, train_size=50, random_state = 253)

# ***Evaluation with different models****
# Using standard ML algorithms on the training set. 
# Evaluate with the test set using basic_metric() function above for basic
# accuracy and precision score for each class as initial evaluation.

# ***kNN k=1*** Using k nearest neighbors from sci-kit learn 
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
p_1nn = knn.predict(X_test)
print("Metrics for kNN, k = 1:")
basic_metrics(p_1nn, y_test)

# ***kNN k=3*** Using k nearest neighbors from sci-kit learn 
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
p_3nn = knn.predict(X_test)
print("Metrics for kNN, k = 3:")
basic_metrics(p_3nn, y_test)

# ***SVM*** 
svm = LinearSVC()
svm.fit(X_train, y_train)
p_svm = svm.predict(X_test)
print("Metrics for SVM: ")
basic_metrics(p_svm, y_test)

# ***Random Forest***
# check to see what is best estimator
rf_param = {
    "n_estimators": [1,10,50,100,500,1000],
    "criterion": ["gini","entropy"],
    "max_features": ["auto"],
    "max_depth": [None,1,5,10],
    "max_leaf_nodes": [None],
    "oob_score": [False],
    "n_jobs": [-1],
    "warm_start": [False],
    "random_state": [1]
}
rf_model_r = RandomForestClassifier()
rf_r = GridSearchCV(estimator=rf_model_r, param_grid=rf_param, 
                              scoring=None,
                              n_jobs=-1, 
                              cv=10, 
                              verbose=1,
                              return_train_score=True)

rf_r.fit(X_train, y_train)

print("Best score:", rf_r.best_score_)
print(rf_r.best_estimator_)
rf_model = rf_r.best_estimator_
y_pred_rf = rf_model.predict(X_test)
print("Metrics for SVM: ")
basic_metrics(y_pred_rf, y_test)

print("End of Program")
