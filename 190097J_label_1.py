import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler as ss
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.decomposition import PCA

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df_train_X = df_train.copy()
df_train_X.drop(['label_2', 'label_3', 'label_4'], axis=1, inplace=True)
df_train_y = df_train_X.pop('label_1')

df_test_X = df_test.copy()
df_test_X.drop(['label_2', 'label_3', 'label_4'], axis=1, inplace=True)
df_test_y = df_test_X.pop('label_1')

scaler = ss()
scaler.fit(df_train_X)

df_train_X = scaler.transform(df_train_X)
df_test_X = scaler.transform(df_test_X)

model = rfc()
model.fit(df_train_X, df_train_y)

y_pred_before = model.predict(df_test_X)

pca = PCA(0.95) # Retain 95% of variance
pca = pca.fit(df_train_X)

pca_df_train_X = pca.fit_transform(df_train_X)
pca_df_test_X = pca.transform(df_test_X)

model_after = rfc()
model_after.fit(pca_df_train_X, df_train_y)

y_pred_after = model_after.predict(pca_df_test_X)

importance = model_after.feature_importances_

deleted_columns = []

for i, j in enumerate(importance):
    if j < 0.01:
        deleted_columns.append(i)

reduced_pca_train_X = np.delete(pca_df_train_X, deleted_columns, axis=1)
reduced_pca_valid_X = np.delete(pca_df_test_X, deleted_columns, axis=1)

reduced_pca_train_X.shape

model_final = rfc()
model_final.fit(reduced_pca_train_X, df_train_y)

y_pred_final = model_final.predict(reduced_pca_valid_X)