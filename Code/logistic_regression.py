### This code was meant to simulate A. Gaydhani, and is therefore inspired by his approach
from useful_functions import pre_process_data, split_data, str2bool, print_results
import numpy as np
import pandas as pd
import argparse
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

preprocess = True

parser = argparse.ArgumentParser(description='Input for the model.')

parser.add_argument('-preprocess', type=str2bool, default='False')
args = parser.parse_args()
if preprocess == False:
    data = pd.read_csv("data/data_processed.csv", sep = "\t")
else:
    data, max_len_padding = pre_process_data(data = pd.read_csv("data/data_merged.csv", sep = "\t"))

# Now use the data for a TF-IDF-based approach and n-gram approach.

train, test = split_data(data)
count_vect = CountVectorizer(ngram_range=(1, 3), lowercase=True)
train_counts = count_vect.fit_transform(train['tweets'])

tf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True, norm='l2', sublinear_tf=False)
train_feats = tf_transformer.fit_transform(train_counts)

from sklearn.linear_model import LogisticRegression

# Use the exact same logistic regression model as in previous works.
log_reg = LogisticRegression(class_weight='balanced', solver='liblinear', C=100, penalty='l2', max_iter=100, multi_class='ovr')
log_reg_fit = log_reg.fit(train_feats, train['class'])

counts = count_vect.transform(test['tweets'])
tfidf_test = tf_transformer.transform(counts)
Y_preds_test = log_reg_fit.predict(tfidf_test)

print_results(test['class'], Y_preds_test, "results/log_reg_results.txt")
