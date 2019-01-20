##### Defining useful functions for the processing and such #####
import numpy as np
import pandas as pd
import nltk # Prepare stopwords
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Preprocess the data.
def pre_process_data(data, remove_stop_words = True, stem = True, include_mentions = False):
    print("remove_stop_words: " + str(remove_stop_words))
    print("stem: "+str(stem))
    data['tweet_text'] = data['tweet_text'].str.lower() # Make to lowercase.
    # Remove all usernames!

    if include_mentions == True:
        data.tweet_text.replace(r'@(\S+)\s?', ' U ', regex=True, inplace=True)
    else:
        data.tweet_text.replace(r'@(\S+)\s?', '', regex=True, inplace=True)
    data.tweet_text.replace(r'[\\x]{2}[a-f0-9]{2}',' ',regex=True, inplace=True) # Remove all ascii and non-ascii characters
    data.tweet_text.replace(r'&(\S+)\s?', ' ', regex=True, inplace=True)
    data.tweet_text.replace(r'[&#]{2}[0-9]{4,6}', ' ', regex=True, inplace=True) # Replace all emojis
    data.tweet_text.replace(r'http\S+', '', regex=True, inplace=True) # Remove all urls
    data.tweet_text.replace(r'www.\S+', '', regex=True, inplace=True) # Remove all urls

    max_amt_words = 0

    if remove_stop_words == True and stem == True:
        stop_words = set(stopwords.words('english'))
        tweets = []
        ps = nltk.stem.PorterStemmer()
        for tweet in data['tweet_text']:
            words = tweet.split(" ")
            if max_amt_words < len(words):
                max_amt_words = len(words)
            words = list(filter(None, words))
            words = [ps.stem(word) for word in words if word not in stop_words]
            words = " ".join(words)
            tweets.append(words)
        data['tweets'] = tweets
    elif remove_stop_words == False and stem == True:

        ps = nltk.stem.PorterStemmer()
        for tweet in data['tweet_text']:
            words = tweet.split(" ")
            if max_amt_words < len(words):
                max_amt_words = len(words)
            words = list(filter(None, words))
            words = [ps.stem(word) for word in words]
            words = " ".join(words)
            tweets.append(words)
        data['tweets'] = tweets
    elif remove_stop_words == True and stem == False:
        stop_words = set(stopwords.words('english'))
        tweets = []

        for tweet in data['tweet_text']:
            words = tweet.split(" ")
            if max_amt_words < len(words):
                max_amt_words = len(words)
            words = list(filter(None, words))
            words = [word for word in words if word not in stop_words]
            words = " ".join(words)
            tweets.append(words)
        data['tweets'] = tweets
    else:
        for tweet in data['tweet_text']:
            nr_words = len(tweet.split(" "))
            if max_amt_words < nr_words:
                max_amt_words = nr_words
        data['tweets'] = data['tweet_text']
    data = data.drop('tweet_text', axis=1)
    if remove_stop_words == True:
        data.tweets.replace(r'\b(?:{})\b'.format('|'.join(stop_words)), '', regex=True, inplace=True) # Removing stop_words
    data.tweets.replace(r'[^A-Za-z0-9, '' '']', '', regex=True, inplace=True) # Remove non-alphanumeric characters
    data.tweets.replace(r',', '', regex=True, inplace=True)
    data.tweets.replace(r' +', ' ', regex=True, inplace=True)
    data['class'] = pd.Series(data['class'], dtype = "category")
    data.to_csv("data/data_processed.csv", sep='\t', encoding='utf-8')
    return data, max_amt_words

# If train_on_data == False, we will use pretrained vector from glove
def get_w2vec_model(data, use_pretrained_vecs = False, emb_dim = 200):
    tweets = data['tweets']
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    w_index = tokenizer.word_index
    vocab_size = len(w_index)+1
    if use_pretrained_vecs == False:
        # Inspiration taken from tutorial: http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.XCAQp89Ki1s
        w2vec_model = Word2Vec(tweets, size=emb_dim, window=10,
            min_count=2, # We will only include them if they occur twice or more.
            workers=10)
        w2vec_model.train(tweets, total_examples=len(tweets), epochs = 10)

    else: # TODO: Use glove tweet training. Then we must assume no stemming
        print("Using glove vectors")
        w2vec_model = KeyedVectors.load_word2vec_format("glove.twitter.27B/glove-twitter-"+str(emb_dim)+".txt", binary=False)
        print("glove vectors loaded")
    return w2vec_model, w_index, tokenizer

def split_data(data, train_test = True, partitioning = 0.8, seed = 123):
    np.random.seed(seed) # Split in training and test set
    if train_test == True:
        indices = np.random.rand(len(data)) < partitioning
        train = data[indices]
        test = data[~indices]
        return train, test
    else: # Else partition into training, validation and test into 60, 20, 20
        train, validation, test = np.split(data.sample(frac=1), [int(partitioning[0]*len(data)), int((partitioning[0] + partitioning[1])*len(data))])
        return train, validation, test

# Created by Maxim@StackOverflow. All credits to him.
# Can be found at https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_results(true,preds, file_name):
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


    class_weights = np.array(np.divide(true.value_counts(),np.sum(true.value_counts())))
    precision_overall = np.sum(np.multiply(precisions, class_weights))
    recall_overall = np.sum(np.multiply(recalls, class_weights))
    F1_overall = np.divide(np.multiply(np.multiply(precision_overall, recall_overall),2), np.add(precision_overall, recall_overall))


    import datetime
    with open(file_name, "a") as res:
        res.write("Result obtained on " + str(datetime.datetime.now()) + "\n \n")
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
