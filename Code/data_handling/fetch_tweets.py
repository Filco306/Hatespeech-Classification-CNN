# Fetches the tweets from the third dataset and puts them into a file.
# standard

from __future__ import print_function
import getopt
import logging
#import os
import sys
from subprocess import call
import pandas as pd

global Logger
Logger = logging.getLogger('get_tweets_by_id')

batch_size = 100

tweets = pd.read_csv("../data/NAACL_SRW_2016.csv", header=None, sep=",")
print(tweets[1].value_counts())


print("starting")
ids = list(tweets[0])
call("./set_up_twitter_env_vars.sh")
import os
consumer_key = os.environ.get('TWITTER_CONSUMER_KEY')
consumer_secret = os.environ.get("TWITTER_CONSUMER_SECRET")
access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
access_token_secret = os.environ.get("TWITTER_ACCESS_SECRET")

print(consumer_key)
print(consumer_secret)
print(access_token)
print(access_token_secret)
import tweepy

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Cred: chrisinmtown@StackOverflow. Copied from https://stackoverflow.com/questions/28384588/twitter-api-get-tweets-with-specific-id
def get_tweet_list(twapi, idlist):
    '''
    Invokes bulk lookup method.
    Raises an exception if rate limit is exceeded.
    '''
    # fetch as little metadata as possible
    tweets = twapi.statuses_lookup(id_=idlist, include_entities=False, trim_user=True)
    if len(idlist) != len(tweets):
        Logger.warn('get_tweet_list: unexpected response size %d, expected %d', len(tweets), len(idlist))
    for tweet in tweets:
        print(tweet)
        if type(tweet) != "str":
            print('%s,%s' % (tweet.id, tweet.text.encode('UTF-8')))
            yield tweet.id, tweet.text.encode('UTF-8')

# Cred: chrisinmtown@StackOverflow. Copied from https://stackoverflow.com/questions/28384588/twitter-api-get-tweets-with-specific-id
# Slightly modified to suit my own purpose
def get_tweets_bulk(twapi, id_list):
    '''
    Fetches content for tweet IDs in a file using bulk request method,
    which vastly reduces number of HTTPS requests compared to above;
    however, it does not warn about IDs that yield no tweet.

    `twapi`: Initialized, authorized API object from Tweepy
    `idfilepath`: Path to file containing IDs
    '''
    list_of_tweets = []
    # process IDs from the file
    tweet_ids = list()

    for id in id_list:
        tweet_id = id
        Logger.debug('Enqueing tweet ID %s', tweet_id)
        tweet_ids.append(tweet_id)
        # API limits batch size
        if len(tweet_ids) == batch_size:
            Logger.debug('get_tweets_bulk: fetching batch of size %d', batch_size)
            tw = get_tweet_list(twapi, tweet_ids)
            for t in tw:
                list_of_tweets.append(t)
            tweet_ids = list()
    # process remainder
    if len(tweet_ids) > 0:
        Logger.debug('get_tweets_bulk: fetching last batch of size %d', len(tweet_ids))
        tw = get_tweet_list(twapi, tweet_ids)
        for t in tw:
            list_of_tweets.append(t)
    return list_of_tweets

# Tuple with tweet id and tweet text is return
tweets_to_fetch = get_tweets_bulk(api, ids[:5])

tweets_w_text_and_id = pd.DataFrame(tweets_to_fetch, columns=[0,1])
tweets.columns = ['id', 'class']
tweets_w_text_and_id.columns = ['id','tweet_text']
data = pd.merge(tweets, tweets_w_text_and_id, on='id')


data.to_csv("../data/tweets_fetched_sample.csv", sep="\t", encoding='utf-8')
