import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
df = pd.read_csv('heart_disease.csv.txt')

# print(df.head(10))
model = sm.GLM.from_formula("AHD ~ Age + Sex + Chol + RestBP + Fbs + RestECG + Slope + Oldpeak + Ca + ExAng + ChestPain + Thal",
family = sm.families.Binomial(), data = df)
result = model.fit()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenconder = LabelEncoder()
X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values
X[:,2] = labelenconder.fit_transform(X[:,2])
X[:,12] =labelenconder.fit_transform(X[:,12])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(x_train, y_train)
predictions = reg.predict(x_test)
print(predictions)

Accuracy = reg.score(x_test, y_test)
print(Accuracy)