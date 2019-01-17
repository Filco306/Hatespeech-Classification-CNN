### This code was meant to simulate A. Gaydhani, and is therefore inspired by his code (although not an exact copy. Modifications were made to suit my purpose. )

from useful_functions import pre_process_data, split_data, pre_process_data_gaydhani, str2bool
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from gaydhani_preprocessing import preprocess_gaydhani
import argparse
preprocess = True


parser = argparse.ArgumentParser(description='Input for the model.')

parser.add_argument('-preprocess', type=str2bool, default='False')
args = parser.parse_args()
preprocess_as_gaydhani = args.preprocess

if preprocess == False:
    data = pd.read_csv("data/data_processed.csv", sep = "\t")
elif preprocess_as_gaydhani == False:
    data, max_len_padding = pre_process_data(data = pd.read_csv("data/data_merged.csv", sep = "\t"))
else:
    train = pd.read_csv("../Data/Gaydhani/train.csv", sep = ",")


    test = pd.read_csv("../Data/Gaydhani/test.csv", sep = ",")
    test = test.drop_duplicates()

    # Make a left join to get everything that is in training and test set
    test = test.merge(train.drop_duplicates(),how='left',left_on='text',right_on='text')


    # All the ones with NaN values are the only ones existing ONLY in the test set. Drop the rest.
    test = test[test.output_class_y.notnull() == False]
    test = test.drop(columns=['output_class_y'])
    test.columns = ['class', 'tweet_text', 'bad_column']
    train.columns = ['bad_column','class', 'tweet_text']
    train, max_len_padding = pre_process_data_gaydhani(data = train)
    test, max_len_padding = pre_process_data_gaydhani(data = test)
    print(train)
    #train.columns = ['bad_column', 'class', 'tweets']
    test.columns = ['class', 'bad_column', 'tweets']
    print(test.head())
    print("preprocessing as gaydhani")


# Now use the data for a TF-IDF-based approach and n-gram approach.


from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial



if preprocess_as_gaydhani == False:
    train, test = split_data(data)
print(train)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer(ngram_range=(1, 3), lowercase=True)
Xtrain_counts = count_vect.fit_transform(train['tweets'])

tf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
Xtrain_tfidf = tf_transformer.fit_transform(Xtrain_counts)


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(C=100, class_weight='balanced', solver='liblinear', penalty='l2', max_iter=100, multi_class='ovr')
print("Fitting model")
log_reg_fit = log_reg.fit(Xtrain_tfidf, train['class'])


counts = count_vect.transform(test['tweets'])
tfidf_test = tf_transformer.transform(counts)
Y_preds_test = log_reg_fit.predict(tfidf_test)

conf_m = confusion_matrix(test['class'], Y_preds_test)


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
with open("results/log_reg_results.txt", "a") as res:
    res.write("Result obtained on " + str(datetime.datetime.now()) + "\n \n")
    res.write("Preprocess as Gaydhani: "+str(pre_process_data_gaydhani + "\n \n"))
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
