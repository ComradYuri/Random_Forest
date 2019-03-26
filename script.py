import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# header argument ignores reading first row in .csv
income_data = pd.read_csv("income.csv", header=0, delimiter=', ', engine='python')
income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)
income_data["country-int"] = income_data["native-country"].apply(lambda row: 0 if row == "United-States" else 1)
income_data["education-int"] = income_data["education"].apply(
    lambda row: {"Doctorate": 3, "Masters": 2, "Bachelors": 1}.get(row, 0))
# print(income_data.iloc[0])

# select labels and data
labels = income_data[["income"]]
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "country-int", "education-int"]]

# split labels and data
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

# create forest
# .values.ravel() is used to prevent an error
forest = RandomForestClassifier(random_state=1, n_estimators=10)
forest.fit(train_data, train_labels.values.ravel())

# accuracy of the forest on test data
print(forest.score(test_data, test_labels))
