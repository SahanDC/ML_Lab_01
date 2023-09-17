import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler as ss
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor as hgbr

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("valid.csv")

df_train = df_train[df_train['label_2'].notna()]

df_train_X = df_train.copy()
df_train_X.drop(['label_1', 'label_3', 'label_4'], axis=1, inplace=True)
df_train_y = df_train_X.pop('label_2')

df_test_X = df_test.copy()
df_test_X.drop(['label_1', 'label_3', 'label_4'], axis=1, inplace=True)
df_test_y = df_test_X.pop('label_2')

scaler = ss()
scaler.fit(df_train_X)

df_train_X = scaler.transform(df_train_X)
df_test_X = scaler.transform(df_test_X)

model = hgbr()
model.fit(df_train_X, df_train_y)

y_pred_before = model.predict(df_test_X)

pca = PCA(0.95) # Retain 95% of variance
pca = pca.fit(df_train_X)

pca_df_train_X = pca.fit_transform(df_train_X)
pca_df_test_X = pca.transform(df_test_X)

model_after = hgbr()
model_after.fit(pca_df_train_X, df_train_y)

y_pred_after = model_after.predict(pca_df_test_X)