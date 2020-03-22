# This file creates the data_merged.csv file, which is then used as the raw data.
import numpy as np
import pandas as pd

# To change the data files used, change the paths of these.
filename_data1 = "data_used/twitter_hate_speech.csv"
filename_data2 = "data_used/twitter_hate_speech2.csv"
filename_data3 = "data_used/tweets_third_dataset.csv"

pd.set_option("display.expand_frame_repr", False)
data1 = pd.read_csv(filename_data1)
data2 = pd.read_csv(filename_data2)
data3 = pd.read_csv(filename_data3, sep="\t")
print(
    data3[
        data3["id"].isin(
            data3["id"].value_counts()[data3["id"].value_counts() > 2].index
        )
    ]
)
data3 = data3.drop_duplicates(subset="id", keep=False)
print(data3["class"].value_counts())
print(data1.columns.values)
print(data2.columns.values)
# Drop unnecessary columns
data1 = data1.drop(
    [
        "unit_id",
        "golden",
        "unit_state",
        "trusted_judgments",
        "last_judgment_at",
        "created_at",
        "orig_golden",
        "orig_last_judgment_at",
        "orig_trusted_judgments",
        "orig_unit_id",
        "orig_unit_state",
        "updated_at",
        "orig_does_this_tweet_contain_hate_speech",
        "does_this_tweet_contain_hate_speech_gold",
    ],
    axis=1,
)
print(data1.head())
print(data2.shape)
data2 = data2.drop(
    ["Unnamed: 0", "count", "hate_speech", "offensive_language", "neither"], axis=1
)
print(data2.shape)
print(data2.head())
data1Classes = [
    "The tweet contains hate speech",
    "The tweet uses offensive language but not hate speech",
    "The tweet is not offensive",
]
# 0 - hate speech 1 - offensive language 2 - neither
# Put data1 into classes 0, 1 and 2
labels = []
for a in data1["does_this_tweet_contain_hate_speech"]:
    labels.append(data1Classes.index(a))
labels = np.asarray(labels)
data1["class"] = labels
print(data1.head())
col_names = ["class", "tweet_text"]
data = pd.DataFrame(columns=col_names)
data1_nrows = data1.shape[0]
data2_nrows = data2.shape[0]
data3_nrows = data3.shape[0]
print(data1_nrows)
print(data2_nrows)
print(data3_nrows)
print(data1["class"].value_counts())
print(data2["class"].value_counts())
print(data3["class"].value_counts())


def append_data(data, data_to_append, nrows, is_last_set=False):
    for i in range(nrows):
        if is_last_set == False:
            data = data.append(
                {
                    "class": data_to_append.iloc[[i]]["class"].values[0],
                    "tweet_text": data_to_append.iloc[[i]]["tweet_text"].values[0],
                },
                ignore_index=True,
            )
        elif data_to_append.iloc[[i]]["class"].values[0] == 0.0:
            data = data.append(
                {
                    "class": 0,
                    "tweet_text": data_to_append.iloc[[i]]["tweet_text"].values[0],
                },
                ignore_index=True,
            )
        else:
            data = data.append(
                {
                    "class": 2,
                    "tweet_text": data_to_append.iloc[[i]]["tweet_text"].values[0],
                },
                ignore_index=True,
            )
        if i % 100 == 0:
            print(i)
    return data


# Appending all data from the 3 different datasets in the same format.
data = append_data(data, data1, data1_nrows)
data = append_data(data, data2, data2_nrows)
data = append_data(data, data3, data3_nrows, is_last_set=True)
print(data.head())
print(data.shape)
data.to_csv("data/data_merged.csv", sep="\t", encoding="utf-8")
# When data is merged to 1 frame, start preprocessing!
print(data1["class"].value_counts())
print(data2["class"].value_counts())
print(data3["class"].value_counts())
