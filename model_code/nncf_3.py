import sys

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow.keras as keras

data = pd.read_csv(sys.argv[1])
data = data.loc[:, data.columns != 'Unnamed: 0']
data.shape

dt = np.array(data)

nfc_pred_df = pd.read_csv('nncf.zip', compression='zip', header=0, sep=',', quotechar='"')

ncf_model = keras.models.load_model('ncf_model')

for k in [5, 10, 20, 100]:
    precision_ncf = tf.keras.metrics.Precision(top_k=k)
    recall_ncf = tf.keras.metrics.Recall(top_k=k)

    precision_ncf.update_state(dt, nfc_pred_df)
    recall_ncf.update_state(dt, nfc_pred_df)

    print(
        f"At K = {k}, we have a precision of {precision_ncf.result().numpy():.5f} \
        and a recall of {recall_ncf.result().numpy():.5f}"
    )

accuracy_ncf = tf.keras.metrics.BinaryAccuracy(threshold=0.005)
auc_ncf = tf.keras.metrics.AUC()

accuracy_ncf.update_state(dt, nfc_pred_df)
auc_ncf.update_state(dt, nfc_pred_df)

print(f'Binary Accuracy: {accuracy_ncf.result().numpy():.5f}, AUC: {auc_ncf.result().numpy():.5f}')

from sklearn.metrics import confusion_matrix
y_pred = np.argmax(nfc_pred_df, axis=1) # check the name of the columns please
confusion_matrix(nfc_pred_df.interaction, y_pred)
