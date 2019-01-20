# Was made to import and process A. Gaydhani's data. Available at his repo:
# https://github.com/adityagaydhani14/Toxic-Language-Detection-in-Online-Content/
# Observe that the training and test data is mixed up, and it was therefore later not used in my work
import pandas as pd

train = pd.read_csv("../../Data/Gaydhani/train.csv", sep = ",")
train['output_class'] = train['output_class'].astype(dtype = "str", errors="ignore")
train = train[(train.output_class == "0") | (train.output_class == "1") | (train.output_class == "2")]
train['output_class'] = train['output_class'].astype(dtype = "int64", errors="ignore")
train['output_class'] = pd.Series(train['output_class'], dtype = "category")
train.to_csv("../../Data/Gaydhani/train.csv", sep = ",")
test = pd.read_csv("../../Data/Gaydhani/test.csv", sep = ",")

data = pd.concat([train, test], ignore_index=True, sort=True)
data.columns = ['class', 'tweets']
data.to_csv("correct_data.csv", sep = "\t")

print(data)
print(data['class'].value_counts())
