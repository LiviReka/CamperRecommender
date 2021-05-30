import datetime
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras.layers import (Concatenate, Dense, Embedding, Flatten, Input, Multiply)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


ncf_model = keras.models.load_model('ncf_model')

df_test = pd.read_csv('df_test.csv')
df_test.pivot(index="user_id", columns="item_id", values="ncf_predictions").values.shape
print('df test pivoted')

#collapse
nfc_pred_df = pd.DataFrame(df_test.pivot(index="user_id", columns="item_id", values="ncf_predictions").values)
print('nfc pred collapsed')

#print("Neural collaborative filtering predictions")
#print(np.array(nfc_pred_df)[:10, :4])
print(nfc_pred_df.shape)
compression_opts = dict(method='zip',archive_name='nncf.csv')
nfc_pred_df.to_csv('nncf.zip', index=False, compression=compression_opts)

#hide
TOP_K = 5
N_EPOCHS = 3

precision_ncf = tf.keras.metrics.Precision(top_k=TOP_K)
recall_ncf = tf.keras.metrics.Recall(top_k=TOP_K)
accuracy_ncf = tf.keras.metrics.BinaryAccuracy(threshold=0.005)

data = pd.read_csv(sys.argv[1])
data = data.loc[:, data.columns != 'Unnamed: 0']
data.shape

dt = np.array(data)

precision_ncf.update_state(dt, nfc_pred_df)
recall_ncf.update_state(dt, nfc_pred_df)
accuracy_ncf.update_state(dt, nfc_pred_df)
#precision_ncf.update_state(data["test"], nfc_pred_df)
#recall_ncf.update_state(data["test"], nfc_pred_df)
print(
    f"At K = {TOP_K}, we have a precision of {precision_ncf.result().numpy():.5f} and a recall of {recall_ncf.result().numpy():.5f}"
)