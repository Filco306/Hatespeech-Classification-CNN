from useful_functions import pre_process_data, split_data, get_w2vec_model

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
import nltk # Prepare stopwords

import re
import argparse

parser = argparse.ArgumentParser(description='Input for the model.')
parser.add_argument('-regval', type=float)
parser.add_argument('-activationfunction', type=str, default='relu')
args = parser.parse_args()
##### Set params #####

max_len_padding = 24

# If we do not want to run the model again.
reuse_old_runs = False
reprocess_data = True

use_validation = True
n_epochs = 10
train_fraction = 0.6
validation_fraction = 0.2
test_fraction = 1 - train_fraction - validation_fraction



#### Model params ####
activation_function = args.activationfunction

filters = 50

if args.regval != None:
    l2_reg = args.regval
else:
    l2_reg = 0.01



if reprocess_data == True:
    print("Reprocessing data!")
    raw_data = pd.read_csv("data/data_merged.csv", sep = "\t")
    data = pre_process_data(raw_data)
else:
    print("Using preprocessed data!")
    data = pd.read_csv("data/data_processed.csv", sep = "\t")



# get the w2vec model
w2vec_model, w_index, tokenizer = get_w2vec_model(data)

emb_dim = 150

# Get the embedding layer
if use_validation == False:
    train, test = split_data(data)
else:
    train, validation, test = split_data(data, train_test = False, partitioning = (0.6,0.2))
    validation_feats = pad_sequences(tokenizer.texts_to_sequences(validation['tweets']), maxlen = max_len_padding)
    validation_labels = keras.utils.to_categorical(np.asarray(validation['class']))

# Convert to input to CNN model
train_feats = pad_sequences(tokenizer.texts_to_sequences(train['tweets']), maxlen = max_len_padding)
test_feats = pad_sequences(tokenizer.texts_to_sequences(test['tweets']), maxlen = max_len_padding)
labels_train = keras.utils.to_categorical(np.asarray(train['class']))
labels_test = keras.utils.to_categorical(np.asarray(test['class']))

vocab_size = len(w_index)+1
emb_matrix = np.zeros((vocab_size,emb_dim))
for w, i in w_index.items():
    try:
        emb_vec = w2vec_model.wv[w]
        emb_matrix[i] = emb_vec
    except KeyError:
        emb_matrix[i] = np.zeros(emb_dim) # If
emb_layer = Embedding(vocab_size, emb_dim, weights=[emb_matrix], trainable=True)

inputs = Input(shape=(train_feats.shape[1],))
emb = emb_layer(inputs)
rshape = Rshape((train_feats.shape[1],emb_dim,1))(emb)

conv0 = Conv2D(filters, (3, emb_dim), activation = 'relu',kernel_regularizer=regularizers.l2(l2_reg))(rshape)
conv1 = Conv2D(filters, (4, emb_dim), activation = 'relu',kernel_regularizer=regularizers.l2(l2_reg))(rshape)
conv2 = Conv2D(filters, (5, emb_dim), activation = 'relu',kernel_regularizer=regularizers.l2(l2_reg))(rshape)

m_pool0 = MaxPooling2D((train_feats.shape[1] - 3 + 1,1), strides=(1,1))(conv0)
m_pool1 = MaxPooling2D((train_feats.shape[1] - 4 + 1,1), strides=(1,1))(conv1)
m_pool2 = MaxPooling2D((train_feats.shape[1] - 5 + 1,1), strides=(1,1))(conv2)

tensor_merged = keras.layers.concatenate([m_pool0, m_pool1, m_pool2], axis=1)
flat = Flatten()(tensor_merged)
rshape = Rshape((150,))(flat)
d_out = Dropout(0.5)(flat)
output = Dense(units=3, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(d_out)
model = Model(inputs, output)
model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
print(test_feats.shape)
print(labels_test.shape)

if reuse_old_runs == False:
    # TO RUN A NEW, USE THE TWO LINES BELOW. OTHERWISE COMMENT IT
    if use_validation == False:
        model.fit(train_feats,labels_train, epochs = n_epochs)
    else:
        model.fit(train_feats,labels_train, epochs = n_epochs, validation_data=(validation_feats, validation_labels))
    preds = model.predict(test_feats)
    np.savetxt("predictions_test.csv", preds, delimiter=",")
else:
    # TO REUSE OLD RUNS, USE THE LINE BELOW. OTHERWISE COMMENT IT
    preds = pd.read_csv("predictions_test.csv", sep=",", header=None)

preds = np.apply_along_axis(np.argmax,1, preds)
true = np.apply_along_axis(np.argmax,1,np.asarray(labels_test))
conf_m = confusion_matrix(true, preds)
print("confusion_matrix is")
print(conf_m)
print("Overall accuracy is")
print(np.divide(np.sum(np.diagonal(conf_m)),np.sum(conf_m)))

def normalize_conf_m(col):
    return np.divide(col, np.sum(col))

# Calculate precision
# precisions:
precisions = np.array([])
recalls = np.array([])
F1s = np.array([])
for i in range(conf_m.shape[1]):
    p = np.divide(conf_m[i,i],np.sum(conf_m[:,i]))
    r = np.divide(conf_m[i,i],np.sum(conf_m[i,:]))
    precisions = np.append(precisions,p)
    recalls = np.append(recalls,r)
    F1s = np.append(F1s, np.divide(np.multiply(np.multiply(p, r),2), np.add(p, r)))
print("precisions are ")
print(precisions)
print("Recalls are")
print(recalls)
print("Precision is")
print(np.mean(precisions))
print("Recall is")
print(np.mean(recalls))

# Get the normalized confusion matrx
print(np.apply_along_axis(normalize_conf_m, 0, conf_m))


class_weights = np.array(np.divide(test['class'].value_counts(),np.sum(test['class'].value_counts())))
precision_overall = np.sum(np.multiply(precisions, class_weights))
recall_overall = np.sum(np.multiply(recalls, class_weights))
F1_overall = np.divide(np.multiply(np.multiply(precision_overall, recall_overall),2), np.add(precision_overall, recall_overall))


import datetime
with open("results/results.txt", "a") as res:
    res.write("Result obtained on " + str(datetime.datetime.now()) + "\n \n")
    res.write("PARAMETERS SET TO \n")
    res.write("Padding: \t" + str(max_len_padding) + "\n ")
    res.write("Activation function: \t"+ str(activation_function)+"\n ")
    res.write("use_validation: \t" + str(use_validation) + "\n ")
    res.write("n_epochs: \t" + str(n_epochs)+"\n")
    res.write("train_fraction \t"+ str(train_fraction)+"\n")
    res.write("validation_fraction: \t" + str(validation_fraction)+"\n")
    res.write("l2_reg: \t" + str(l2_reg)+"\n")
    res.write("filters: \t" + str(filters)+"\n \n")
    res.write("------RESULTS------"+"\n \n")
    res.write("Confusion matrix : \n"+str(conf_m)+"\n")
    res.write("Precisions : \n" + str(precisions) + "\n")
    res.write("Accuracy : \n" + str(np.divide(np.sum(np.diagonal(conf_m)),np.sum(conf_m)))+ "\n")
    res.write("Recalls : \n" + str(recalls)+ "\n")
    res.write("F1s : \n" + str(F1s) + "\n")
    res.write("Precision overall: \n" + str(precision_overall) + "\n")
    res.write("Recall : \n" + str(recall_overall)+"\n")
    res.write("F1 overall : \n" + str(F1_overall))
    res.write("\n\n\n")
