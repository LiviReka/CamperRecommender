import pandas as pd
import numpy as np
import scipy.sparse as sparse
import random
import implicit
from sklearn import metrics

# user item combinations df
cleandata_m = pd.read_csv('../data/cleandata.csv')
cleandata_m = cleandata_m[['CUSTOMER_ID', 'PRODUCT_ID']]
cleandata_m['quantity'] = 1

grouped_purchased = cleandata_m.groupby(by=['CUSTOMER_ID', 'PRODUCT_ID']).min().reset_index()

## creating sparse user_item matrix
cust_dict = {i: count for count, i in enumerate(grouped_purchased['CUSTOMER_ID'].unique())}
prod_dict = {i: count for count, i in enumerate(grouped_purchased['PRODUCT_ID'].unique())}

rows = [cust_dict[i] for i in grouped_purchased['CUSTOMER_ID']]
cols = [prod_dict[i] for i in grouped_purchased['PRODUCT_ID']]
vals = list(grouped_purchased['quantity'])

user_item_sparse = sparse.csr_matrix((vals, (rows, cols)), shape=(len(cust_dict), len(prod_dict)))


## Measuring Sparsity
# matrix_size = purchases_sparse.shape[0]*purchases_sparse.shape[1]
# num_purchases = len(purchases_sparse.nonzero()[0])
# sparsity = 100*(1 - (num_purchases/matrix_size))


def make_train(ratings, pct_test=0.2):
    test_set = ratings.copy()  # Make a copy of the original set to be the test set.
    test_set[test_set != 0] = 1  # Store the test set as a binary preference matrix
    training_set = ratings.copy()  # Make a copy of the original data we can alter as our training set.
    nonzero_inds = training_set.nonzero()  # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))  # Zip these pairs together of user,item index into list
    random.seed(0)  # Set the random seed to zero for reproducibility
    num_samples = int(
        np.ceil(pct_test * len(nonzero_pairs)))  # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples)  # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples]  # Get the user row indices
    item_inds = [index[1] for index in samples]  # Get the item column indices
    training_set[user_inds, item_inds] = 0  # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros()  # Get rid of zeros in sparse array storage after update to save space
    return training_set, test_set, list(set(user_inds))  # Output the unique list of user rows that were altered


def auc_score(y, y_hat):
    return metrics.roc_auc_score(y, y_hat)




train, test, masked_uid = make_train(user_item_sparse, pct_test=0.1)

# initialize a model
model = implicit.als.AlternatingLeastSquares(factors=50, num_threads=7, iterations=15)
model.fit(train.T)


def make_recommendation(user, k=10):
    return model.recommend(random.choice(list(cust_dict.values())), train)


# make_recommendation(random.choice(list(cust_dic.keys())))


# recomms = model.recommend(random.choice(list(cust_dict.values())), train)


def ranking(user, n_recs):
    ranks = model.rank_items(userid=user, user_items=train,
                             selected_items=list(prod_dict.values()))
    preds = np.zeros(len(prod_dict))
    for count, r in enumerate(ranks):
        if count < n_recs:
            preds[r[0]] = 1
        else:
            preds[r[0]] = 0
    return preds


ranks = ranking(user=random.choice(list(cust_dict.values())), n_recs=5)


def sequential_mean(old, new, iteration):
    """Function to sequentially update a mean"""
    return np.round(old + (new - old) / iteration, 3)

def eval(y, masked_uid):
    m = 0
    for count, i in enumerate(masked_uid):
        score = auc_score(y=y[i].toarray().flatten(), y_hat=ranking(user=i, n_recs=3))
        m = sequential_mean(m, score, count+1)
        print(m)
    return m


    # return np.mean([auc_score(y=y[i].toarray().flatten(), y_hat=ranking(user=i, n_recs=3)) for i in masked_uid])


mean_score = eval(y=test, masked_uid=masked_uid)
print(mean_score)

print('hello world')
