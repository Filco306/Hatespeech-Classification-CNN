##### Define functions #####
import numpy as np
import pandas as pd
import nltk # Prepare stopwords
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec #from word_2_vec_modelling import w_2_vec

def pre_process_data(data, remove_stop_words = True, stem = True):
    data['tweet_text'] = data['tweet_text'].str.lower() # Make to lowercase.
    # Remove all usernames!
    data.tweet_text.replace(r'@(\S+)\s?', ' U ', regex=True, inplace=True)
    data.tweet_text.replace(r'[\\x]{2}[a-f0-9]{2}',' ',regex=True, inplace=True) # Remove all ascii and non-ascii characters
    data.tweet_text.replace(r'&(\S+)\s?', ' ', regex=True, inplace=True)
    data.tweet_text.replace(r'[&#]{2}[0-9]{4,6}', ' ', regex=True, inplace=True) # Replace all emojis
    data.tweet_text.replace(r'http\S+', '', regex=True, inplace=True) # Remove all urls
    data.tweet_text.replace(r'www.\S+', '', regex=True, inplace=True) # Remove all urls
    data.tweet_text.replace(r'.*\b(im|i''m)\b','i am',regex=True, inplace=True) # Change words and slangs
    data.tweet_text.replace(r'.*\b(lil)\b','little',regex=True, inplace=True)
    data.tweet_text.replace(r'.*\b(af)\b','as fuck',regex=True, inplace=True)
    if remove_stop_words == True and stem == True:
        stop_words = set(stopwords.words('english'))
        additional_stop_words = ["you're", "i'm", "u're","youu", "u", "r", "ya"]
        """with open("stoplist_en.txt","r") as stop_word_file:
            for line in stop_word_file:
                aString = line
                stop_words.add(aString.strip())"""
        for a in additional_stop_words:
            stop_words.add(a)
        tweets = []
        ps = nltk.stem.PorterStemmer()
        for tweet in data['tweet_text']:
            words = tweet.split(" ")
            words = list(filter(None, words))
            words = [ps.stem(word) for word in words if word not in stop_words]
            words = " ".join(words)
            tweets.append(words)
        data['tweets'] = tweets
    elif stem == True and remove_stop_words == False:
        stop_words = set(stopwords.words('english'))
        additional_stop_words = ["you're", "i'm", "u're","youu", "u", "r", "ya"]

        for a in additional_stop_words:
            stop_words.add(a)
        tweets = []
        ps = nltk.stem.PorterStemmer()
        for tweet in data['tweet_text']:
            words = tweet.split(" ")
            words = list(filter(None, words))
            words = [ps.stem(word) for word in words]
            words = " ".join(words)
            tweets.append(words)
        data['tweets'] = tweets
    elif stem == False and remove_stop_words == True:
        stop_words = set(stopwords.words('english'))
        additional_stop_words = ["you're", "i'm", "u're","youu", "u", "r", "ya"]

        for a in additional_stop_words:
            stop_words.add(a)
        tweets = []
        ps = nltk.stem.PorterStemmer()
        for tweet in data['tweet_text']:
            words = tweet.split(" ")
            words = list(filter(None, words))
            words = [word for word in words if word not in stop_words]
            words = " ".join(words)
            tweets.append(words)
        data['tweets'] = tweets
    else:
        data['tweets'] = data['tweet_text']
    data = data.drop('tweet_text', axis=1)
    data.tweets.replace(r'\b(?:{})\b'.format('|'.join(stop_words)), '', regex=True, inplace=True) # Removing stop_words
    data.tweets.replace(r'[^A-Za-z0-9, '' '']', '', regex=True, inplace=True) # Remove non-alphanumeric characters
    data.tweets.replace(r',', '', regex=True, inplace=True)
    data.tweets.replace(r' +', ' ', regex=True, inplace=True)
    data['class'] = pd.Series(data['class'], dtype = "category")
    data.to_csv("data/data_processed.csv", sep='\t', encoding='utf-8')
    return data

def get_w2vec_model(data, train_on_data = True):
    if train_on_data == True:
        tweets = data['tweets']
        #print(tweets)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(tweets)
        w_index = tokenizer.word_index
        vocab_size = len(w_index)+1
        w2vec_model = Word2Vec( # Inspiration taken from tutorial: http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.XCAQp89Ki1s
            tweets,
            size=150,
            window=10,
            min_count=2, # We will only include them if they occur twice or more.
            workers=10
        )
        w2vec_model.train(tweets, total_examples=len(tweets), epochs = 10)
        return w2vec_model, w_index, tokenizer
    else: # TODO: Use glove tweet training. Assumes no stemming.
        return None

def split_data(data, train_test = True, partitioning = 0.8, seed = 123):
    np.random.seed(seed) # Split in training and test set
    if train_test == True:
        train_partition = partitioning
        indices = np.random.rand(len(data)) < train_partition
        train = data[indices]
        test = data[~indices]
        return train, test
    else: # Else partition into training, validation and test into 60, 20, 20
        train, validation, test = np.split(data.sample(frac=1), [int(partitioning[0]*len(data)), int((partitioning[0] + partitioning[1])*len(data))])
        return train, validation, test
