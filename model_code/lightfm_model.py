import sys
import pandas as pd
import numpy as np
import lightfm
from scipy import sparse

import tensorflow as tf


data = pd.read_csv(sys.argv[1])
data = data.loc[:, data.columns != 'Unnamed: 0']
data.shape

dt = np.array(data)
dt.shape

df_test = pd.read_csv('df_test.csv')
print('date loaded')

# LightFM model
norm = lambda x: (x - np.min(x)) / np.ptp(x)
lightfm_model = lightfm.LightFM(loss="warp")

print('fitting lightfm model...')
lightfm_model.fit(sparse.coo_matrix(data), epochs=200)

print('generating lightfm predictions...')
lightfm_predictions = lightfm_model.predict(df_test["user_id"].values, df_test["item_id"].values)
df_test["lightfm_predictions"] = lightfm_predictions

print('pivoting prediction file')
wide_predictions = df_test.pivot(index="user_id", columns="item_id", values="lightfm_predictions").values
lightfm_predictions = norm(wide_predictions)

print('computing metrics...')
for k in [5, 10, 20, 25]:
    # compute the metrics
    precision_lightfm = tf.keras.metrics.Precision(top_k=k)
    recall_lightfm = tf.keras.metrics.Recall(top_k=k)
    precision_lightfm.update_state(dt, lightfm_predictions)
    recall_lightfm.update_state(dt, lightfm_predictions)
    #precision_lightfm.update_state(data["test"], data["lightfm_predictions"])
    #recall_lightfm.update_state(data["test"], data["lightfm_predictions"])

    print(f"At K = {k}, we have a precision of {precision_lightfm.result().numpy():.5f} and a recall of {recall_lightfm.result().numpy():.5f}")
