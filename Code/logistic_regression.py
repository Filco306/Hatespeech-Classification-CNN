from useful_functions import pre_process_data, split_data
import numpy as np
import pandas as pd

preprocess = True

if preprocess == False:
    data = pd.read_csv("data/data_processed.csv", sep = "\t")
else:
    data = pre_process_data(data = pd.read_csv("data/data_merged.csv", sep = "\t"))
print(data)

# Now use the data for a TF-IDF-based approach.
