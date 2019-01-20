from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from useful_functions import *
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from useful_functions import *
import sklearn as sk
import pandas as pd
import numpy as np
import nltk
import keras
from pandas import Series
from keras import regularizers
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Dense, Activation, Embedding, Input, Reshape as Rshape, Flatten as Flatten
from sklearn.metrics import confusion_matrix

import re
import argparse

parser = argparse.ArgumentParser(description='Input for the model.')
parser.add_argument('-seed', type=int, default=123)
parser.add_argument('-use_pretrained_vecs', type=str2bool, default='False')
parser.add_argument('-emb_dim', type=int, default=100)
args = parser.parse_args()

##### Set params #####


reprocess_data = True
validation_fraction = 0.2

seed = args.seed
use_pretrained_vecs = args.use_pretrained_vecs
if use_pretrained_vecs == True:
    print("Using preprocessed vectors")
    remove_stop_words = True
    stem = False
else:
    remove_stop_words = True
    stem = True

#### Model params ####

# Embedding dimension. Should be 25, 50, 100 or 200.
emb_dim = args.emb_dim

if reprocess_data == True:
    print("Reprocessing data!")
    raw_data = pd.read_csv("data/data_merged.csv", sep = "\t")
    data, max_len_padding = pre_process_data(data = raw_data, remove_stop_words = remove_stop_words, stem = stem)
else:
    print("Using preprocessed data!")
    data = pd.read_csv("data/data_processed.csv", sep = "\t")

# get the w2vec model

w2vec_model, w_index, tokenizer = get_w2vec_model(data, use_pretrained_vecs = use_pretrained_vecs, emb_dim = emb_dim)
train, test = train_test_split(data)
train_raw, test_raw = train_test_split(raw_data, random_state = seed)

train.to_csv("data/train.csv")
test.to_csv("data/test.csv")

# Convert to input to CNN model
train_feats = pad_sequences(tokenizer.texts_to_sequences(train['tweets']), maxlen = max_len_padding)
test_feats = pad_sequences(tokenizer.texts_to_sequences(test['tweets']), maxlen = max_len_padding)
labels_train = np.asarray(train['class'])
labels_test = np.asarray(test['class'])
vocab_size = len(w_index)+1
emb_matrix = np.zeros((vocab_size,emb_dim))
for w, i in w_index.items():
    try:
        emb_vec = w2vec_model.wv[w]
        emb_matrix[i] = emb_vec
    except KeyError:
        emb_matrix[i] = np.zeros(emb_dim) # If

model = XGBClassifier()
model.fit(train_feats, labels_train)

y_pred = model.predict(test_feats)
preds = [round(value) for value in y_pred]
true = labels_test
true = pd.Series(true)
print_results(true, preds, "results/results_XGboost.txt")
