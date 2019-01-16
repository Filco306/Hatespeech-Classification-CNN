import pandas as pd

train = pd.read_csv("../../Data/Gaydhani/train.csv", sep = ",")
train['output_class'] = train['output_class'].astype(dtype = "str", errors="ignore")
train = train[(train.output_class == "0") | (train.output_class == "1") | (train.output_class == "2")]
train['output_class'] = train['output_class'].astype(dtype = "int64", errors="ignore")
#train = train[(train.output_class == 0) | (train.output_class == 1) | (train.output_class == 2)]
train['output_class'] = pd.Series(train['output_class'], dtype = "category")
train.to_csv("../../Data/Gaydhani/train.csv", sep = ",")
test = pd.read_csv("../../Data/Gaydhani/test.csv", sep = ",")
#test['output_class'] = pd.Series(test['output_class'], dtype = "category")

data = pd.concat([train, test], ignore_index=True, sort=True)
data.columns = ['class', 'tweets']
data.to_csv("correct_data.csv", sep = "\t")

print(data)
print(data['class'].value_counts())
