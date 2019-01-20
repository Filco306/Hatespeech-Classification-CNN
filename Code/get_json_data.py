import json

def get_tweets(clas, filename):
    tweets = []
    for line in open(filename, "r"):
        tweets.append(json.loads(line))
    innocent_tweets = []
    for tweet in tweets:
        innocent_tweets.append((int(tweet['id_str']), clas, str(tweet['text'])))
    return innocent_tweets
innocent_tweets = get_tweets(2, "../datashare/neither.json")
hate_speech_tweets = get_tweets(0, "../datashare/racism.json")
print("Number of racist tweets ")
print(len(hate_speech_tweets))
hate_speech_tweets_2 = get_tweets(0, "../datashare/sexism.json")
all_data = innocent_tweets + hate_speech_tweets + hate_speech_tweets_2

# Now we have all tweets. Now put them into a dataframe and write it to a csv.
import pandas as pd

data = pd.DataFrame(all_data, columns = ["id", "class", "tweet_text"])
data.to_csv("../datatweets_third_dataset.csv", sep='\t', encoding='utf-8')
