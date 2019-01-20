from useful_functions import pre_process_data, split_data, str2bool, print_results
import numpy as np
import pandas as pd
import argparse
from xgboost import XGBClassifier
preprocess = True
parser = argparse.ArgumentParser(description='Input for the model.')
parser.add_argument('-preprocess', type=str2bool, default='False')
args = parser.parse_args()
if preprocess == False:
    data = pd.read_csv("data/data_processed.csv", sep = "\t")
else:
    data, max_len_padding = pre_process_data(data = pd.read_csv("data/data_merged.csv", sep = "\t"))
# Now use the data for a TF-IDF-based approach and n-gram approach.

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial

train, test = split_data(data)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer(ngram_range=(1, 3), lowercase=True)
Xtrain_counts = count_vect.fit_transform(train['tweets'])

tf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True, norm='l2', sublinear_tf=False)
train_feats_tf_idf = tf_transformer.fit_transform(Xtrain_counts)

model = XGBClassifier()
model.fit(train_feats_tf_idf, train['class'])

counts = count_vect.transform(test['tweets'])
test_feats_tf_idf = tf_transformer.transform(counts)

y_pred = model.predict(test_feats_tf_idf)
preds = [round(value) for value in y_pred]
true = test['class']
true = pd.Series(true)
print_results(true, preds, "results/results_XGboost_TFIDF.txt")
