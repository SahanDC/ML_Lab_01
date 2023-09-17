import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler as ss
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.decomposition import PCA

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df_train_X = df_train.copy()
df_train_X.drop(['label_2', 'label_3', 'label_1'], axis=1, inplace=True)
df_train_y = df_train_X.pop('label_4')

df_test_X = df_test.copy()
df_test_X.drop(['label_2', 'label_3', 'label_1'], axis=1, inplace=True)
df_test_y = df_test_X.pop('label_4')

over_sampler = RandomOverSampler(sampling_strategy='auto')
df_train_X, df_train_y = over_sampler.fit_resample(df_train_X, df_train_y)

scaler = ss()
scaler.fit(df_train_X)

df_train_X = scaler.transform(df_train_X)
df_test_X = scaler.transform(df_test_X)

k = 5
model = knnc(n_neighbors=k)
model.fit(df_train_X, df_train_y)

y_pred_before = model.predict(df_test_X)

pca = PCA(0.95) # Retain 95% of variance
pca = pca.fit(df_train_X)

pca_df_train_X = pca.fit_transform(df_train_X)
pca_df_test_X = pca.transform(df_test_X)

k = 5
model_after = knnc(n_neighbors=k)
model_after.fit(pca_df_train_X, df_train_y)

y_pred_after = model_after.predict(pca_df_test_X)