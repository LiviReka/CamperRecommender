import pandas as pd
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from sklearn import metrics


def auc_score(y, y_hat):
    return metrics.roc_auc_score(y, y_hat)


def precision_score(y, y_hat):
    return metrics.precision_score(y, y_hat)


def f1_score(y, y_hat):
    return metrics.f1_score(y, y_hat)


def recall_score(y, y_hat):
    return metrics.recall_score(y, y_hat)


def sequential_mean(old, new, iteration):
    """Function to sequentially update a mean"""
    return np.round(old + (new - old) / iteration, 3)


def eval(star, pred):
    """Evaluating the recommendations by letting the model recommend on masked users"""
    recall = 0
    f1 = 0
    auc = 0.5
    precision = 0

    for i in range(star.shape[0]):
        y_star = star[i, :].toarray().flatten()  # extract true user item array
        y_hat = pred[i, :].toarray().flatten()  # gives boolean user product array indicating recs

        auc = sequential_mean(old=auc, new=auc_score(y=y_star, y_hat=y_hat), iteration=i + 1)
        precision = sequential_mean(old=precision, new=precision_score(y=y_star, y_hat=y_hat), iteration=i + 1)
        recall = sequential_mean(old=recall, new=recall_score(y=y_star, y_hat=y_hat), iteration=i + 1)
        f1 = sequential_mean(old=f1, new=f1_score(y=y_star, y_hat=y_hat), iteration=i + 1)

        if (i % 1000) == 0:
            print('#### ---- ####')
            print(f'AUC: {auc}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1: {f1}')
            print(f'{np.round(i / star.shape[0] * 100, 3)}% done')
    return auc, precision, recall, f1

# ## Importing Original UIM
# org_uim = pd.read_csv('../data/uim.csv')
# org_uim.drop(['Unnamed: 0'], inplace=True, axis=1)
# org_uim = csr_matrix(org_uim)
# org_uim[org_uim > 0] = 1
# scipy.sparse.save_npz('uim.npz', org_uim)
org_uim = scipy.sparse.load_npz('uim.npz')


# ## Creating Sparse Matrix From Predictions & True Vaules From Result Matrix
# preds = pd.read_csv('df_test.csv')
# preds.drop(['Unnamed: 0'], inplace=True, axis=1)
#
# print(org_uim.shape[1] * org_uim.shape[0] == len(preds))  # check output for dimensions

# ## Set Item Locations From Output Matrix
# cols = preds.item_id.values
# rows = preds.user_id.values
#
# ## saving interation column as sparse
# vals = preds.interaction.values
# interactions_sparse = csr_matrix((vals, (rows, cols)), shape=(len(preds.user_id.unique()), len(preds.item_id.unique())))
# interactions_sparse[interactions_sparse > 0] = 1
# scipy.sparse.save_npz('interactions_sparse.npz', interactions_sparse)
#
# ## saving prediction columns as sparse
# vals = preds.ncf_predictions.values
# preds_sparse = csr_matrix((vals, (rows, cols)), shape=(len(preds.user_id.unique()), len(preds.item_id.unique())))
# scipy.sparse.save_npz('pred_sparse.npz', preds_sparse)
preds_sparse = scipy.sparse.load_npz('pred_sparse.npz')
interactions_sparse = scipy.sparse.load_npz('interactions_sparse.npz')
interactions_sparse[interactions_sparse > 0] = 1

K = 10
recommendations = np.array(np.argsort(preds_sparse.todense(), axis=1)[:, -K:])
user_rec = np.zeros(org_uim.shape)

for i in range(org_uim.shape[0]):
    user_rec[i, recommendations[i]] = 1

user_rec = csr_matrix(user_rec)
scipy.sparse.save_npz('recommendations_sparse.npz', user_rec)
preds_sparse = scipy.sparse.load_npz('recommendations_sparse.npz')

print(np.mean(user_rec.sum(axis=1)) == 10)

# auc, precision, recall, f1 = eval(org_uim, user_rec)

print('hello world')
