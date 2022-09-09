import numpy as np
import matplotlib.pyplot as plt
import csv
import sklearn
from sklearn import datasets, svm, metrics, neighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

df = pd.read_csv(r'C:\Users\Welcome\PycharmProjects\CompetitiveProgramming\house-prices-advanced-regression-techniques\train.csv')
test = pd.read_csv(r'C:\Users\Welcome\PycharmProjects\CompetitiveProgramming\house-prices-advanced-regression-techniques\test.csv')
## First, clean all the data.
## Then, use regression.
## Put the mode where we have null values.


df.drop(['PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Alley', 'Id'], axis=1, inplace=True)
# print(df.shape)
# df = df.dropna()
df['MSZoning'] = LabelEncoder().fit_transform(df['MSZoning'])
df['Street'] = LabelEncoder().fit_transform(df['Street'])
df['LotShape'] = LabelEncoder().fit_transform(df['LotShape'])
df['LandContour'] = LabelEncoder().fit_transform(df['LandContour'])
df['Utilities'] = LabelEncoder().fit_transform(df['Utilities'])
df['LotConfig'] = LabelEncoder().fit_transform(df['LotConfig'])
df['LandSlope'] = LabelEncoder().fit_transform(df['LandSlope'])
df['Neighborhood'] = LabelEncoder().fit_transform(df['Neighborhood'])
df['Condition1'] = LabelEncoder().fit_transform(df['Condition1'])
df['Condition2'] = LabelEncoder().fit_transform(df['Condition2'])
df['BldgType'] = LabelEncoder().fit_transform(df['BldgType'])
df['HouseStyle'] = LabelEncoder().fit_transform(df['HouseStyle'])
df['RoofStyle'] = LabelEncoder().fit_transform(df['RoofStyle'])
df['RoofMatl'] = LabelEncoder().fit_transform(df['RoofMatl'])
df['Exterior1st'] = LabelEncoder().fit_transform(df['Exterior1st'])
df['Exterior2nd'] = LabelEncoder().fit_transform(df['Exterior2nd'])
df['MasVnrType'] = LabelEncoder().fit_transform(df['MasVnrType'])
df['ExterQual'] = LabelEncoder().fit_transform(df['ExterQual'])
df['ExterCond'] = LabelEncoder().fit_transform(df['ExterCond'])
df['Foundation'] = LabelEncoder().fit_transform(df['Foundation'])
df['BsmtQual'] = LabelEncoder().fit_transform(df['BsmtQual'])
df['BsmtCond'] = LabelEncoder().fit_transform(df['BsmtCond'])
df['BsmtExposure'] = LabelEncoder().fit_transform(df['BsmtExposure'])
df['BsmtFinType1'] = LabelEncoder().fit_transform(df['BsmtFinType1'])
df['BsmtFinType2'] = LabelEncoder().fit_transform(df['BsmtFinType2'])
df['Heating'] = LabelEncoder().fit_transform(df['Heating'])
df['HeatingQC'] = LabelEncoder().fit_transform(df['HeatingQC'])
df['CentralAir'] = LabelEncoder().fit_transform(df['CentralAir'])
df['Electrical'] = LabelEncoder().fit_transform(df['Electrical'])
df['KitchenQual'] = LabelEncoder().fit_transform(df['KitchenQual'])
df['Functional'] = LabelEncoder().fit_transform(df['Functional'])
df['GarageType'] = LabelEncoder().fit_transform(df['GarageType'])
df['GarageFinish'] = LabelEncoder().fit_transform(df['GarageFinish'])
df['GarageQual'] = LabelEncoder().fit_transform(df['GarageQual'])
df['GarageCond'] = LabelEncoder().fit_transform(df['GarageCond'])
df['PavedDrive'] = LabelEncoder().fit_transform(df['PavedDrive'])
df['SaleType'] = LabelEncoder().fit_transform(df['SaleType'])
df['SaleCondition'] = LabelEncoder().fit_transform(df['SaleCondition'])


test.drop(['PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Alley', 'Id'], axis=1, inplace=True)
# test = test.dropna()

test['MSZoning'] = LabelEncoder().fit_transform(test['MSZoning'])
test['Street'] = LabelEncoder().fit_transform(test['Street'])
test['LotShape'] = LabelEncoder().fit_transform(test['LotShape'])
test['LandContour'] = LabelEncoder().fit_transform(test['LandContour'])
test['Utilities'] = LabelEncoder().fit_transform(test['Utilities'])
test['LotConfig'] = LabelEncoder().fit_transform(test['LotConfig'])
test['LandSlope'] = LabelEncoder().fit_transform(test['LandSlope'])
test['Neighborhood'] = LabelEncoder().fit_transform(test['Neighborhood'])
test['Condition1'] = LabelEncoder().fit_transform(test['Condition1'])
test['Condition2'] = LabelEncoder().fit_transform(test['Condition2'])
test['BldgType'] = LabelEncoder().fit_transform(test['BldgType'])
test['HouseStyle'] = LabelEncoder().fit_transform(test['HouseStyle'])
test['RoofStyle'] = LabelEncoder().fit_transform(test['RoofStyle'])
test['RoofMatl'] = LabelEncoder().fit_transform(test['RoofMatl'])
test['Exterior1st'] = LabelEncoder().fit_transform(test['Exterior1st'])
test['Exterior2nd'] = LabelEncoder().fit_transform(test['Exterior2nd'])
test['MasVnrType'] = LabelEncoder().fit_transform(test['MasVnrType'])
test['ExterQual'] = LabelEncoder().fit_transform(test['ExterQual'])
test['ExterCond'] = LabelEncoder().fit_transform(test['ExterCond'])
test['Foundation'] = LabelEncoder().fit_transform(test['Foundation'])
test['BsmtQual'] = LabelEncoder().fit_transform(test['BsmtQual'])
test['BsmtCond'] = LabelEncoder().fit_transform(test['BsmtCond'])
test['BsmtExposure'] = LabelEncoder().fit_transform(test['BsmtExposure'])
test['BsmtFinType1'] = LabelEncoder().fit_transform(test['BsmtFinType1'])
test['BsmtFinType2'] = LabelEncoder().fit_transform(test['BsmtFinType2'])
test['Heating'] = LabelEncoder().fit_transform(test['Heating'])
test['HeatingQC'] = LabelEncoder().fit_transform(test['HeatingQC'])
test['CentralAir'] = LabelEncoder().fit_transform(test['CentralAir'])
test['Electrical'] = LabelEncoder().fit_transform(test['Electrical'])
test['KitchenQual'] = LabelEncoder().fit_transform(test['KitchenQual'])
test['Functional'] = LabelEncoder().fit_transform(test['Functional'])
test['GarageType'] = LabelEncoder().fit_transform(test['GarageType'])
test['GarageFinish'] = LabelEncoder().fit_transform(test['GarageFinish'])
test['GarageQual'] = LabelEncoder().fit_transform(test['GarageQual'])
test['GarageCond'] = LabelEncoder().fit_transform(test['GarageCond'])
test['PavedDrive'] = LabelEncoder().fit_transform(test['PavedDrive'])
test['SaleType'] = LabelEncoder().fit_transform(test['SaleType'])
test['SaleCondition'] = LabelEncoder().fit_transform(test['SaleCondition'])

Le = LabelEncoder()
df['LotFrontage'].fillna(int(df['LotFrontage'].mean()), inplace=True)
df['MasVnrArea'].fillna(int(df['MasVnrArea'].mean()), inplace=True)
df['GarageYrBlt'].fillna(int(df['GarageYrBlt'].mean()), inplace=True)

print(test.isna().sum())
test['LotFrontage'].fillna(int(test['LotFrontage'].mean()), inplace=True)
test['MasVnrArea'].fillna(int(test['MasVnrArea'].mean()), inplace=True)
test['BsmtFinSF1'].fillna(int(test['BsmtFinSF1'].mean()), inplace=True)
test['BsmtFinSF2'].fillna(int(test['BsmtFinSF2'].mean()), inplace=True)
test['BsmtUnfSF'].fillna(int(test['BsmtUnfSF'].mean()), inplace=True)
test['TotalBsmtSF'].fillna(int(test['TotalBsmtSF'].mean()), inplace=True)
test['BsmtFullBath'].fillna(int(test['BsmtFullBath'].mean()), inplace=True)
test['BsmtHalfBath'].fillna(int(test['BsmtHalfBath'].mean()), inplace=True)
test['GarageYrBlt'].fillna(int(test['GarageYrBlt'].mean()), inplace=True)
test['GarageCars'].fillna(int(test['GarageCars'].mean()), inplace=True)
test['GarageArea'].fillna(int(test['GarageArea'].mean()), inplace=True)
test['GarageCars'].fillna(int(test['GarageCars'].mean()), inplace=True)


X = df.loc[:, df.columns != 'SalePrice']
y = df['SalePrice']
X_test = test.loc[:, test.columns != 'SalePrice']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y = np.array(y)
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
knn.fit(X, y)

prediction = knn.predict(X_test)
# accuracy = metrics.accuracy_score(y_test, prediction)


final_list = list(zip([i for i in range(1461, (1461+len(X_test)))], prediction))
final_list_in_pandas = pd.DataFrame(final_list)
# final_list_in_pandas = final_list_in_pandas.reset_index(drop=True)
print(final_list_in_pandas.columns)
# final_list_in_pandas.set_index('0')

# final_list_in_pandas.to_csv('new_csv.csv')

f=pd.read_csv("new_csv.csv")
keep_col = ['Id', 'SalePrice']
new_f = f[keep_col]
new_f.to_csv("newFile.csv", index=False)


# f = open('new_csv.csv', 'w')
#
# writer = csv.writer(f)
# writer.writerow(final_list)
# f.close()