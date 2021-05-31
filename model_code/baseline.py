import pandas as pd
import numpy as np
import scipy.sparse as sparse
import random
import implicit
from sklearn import metrics
import matplotlib.pyplot as plt

from model_code.bayesian_opt.bayesian_opt import BayesianOptEI
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# from line_profiler import LineProfiler
import time


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


class BaselineImplicit:
    def __init__(self, transactions, min_items=0, max_items=None, n_recs=3, model_fact=None, model_iter=None):
        self.transactions_raw = transactions  # array with exactly 3 rows: 'CUSTOMER_ID', 'PRODUCT_ID', 'feedback'
        self.transactions_filtered = self._select_users(min_items=min_items, max_items=max_items) # filter users

        self.uim_sparse = self._sparsify_uim()  # transforming transaction list to sparse user item matrix
        self.tr, self.te, self.masked_uid = self._make_train(pct_test=0.1)  # create implicit train set

        self.auc, self.precision, self.recall, self.f1 = 0.5, 0, 0, 0  # performance metrics

        self.model_fact = model_fact  # number of latent factors in the model
        self.model_iter = model_iter  # number of als iterations
        self.n_recs = n_recs  # number of recommendations per user

        self.model = self._init_model()  # train implicit model

    def _select_users(self, min_items, max_items):
        """Filtering Transaction list for relevant users given count of bought products"""
        assert sum(self.transactions_raw.columns == ['customer_id', 'product_id', 'feedback']) == 3

        if max_items is None:
            max_items = np.inf

        u_counts = self.transactions_raw.customer_id.value_counts()
        u_filt = u_counts[(u_counts >= min_items) & (u_counts <= max_items)]
        print(f'{np.round(len(u_filt)/len(u_counts)*100, 3)}% of all users.')
        return self.transactions_raw[self.transactions_raw['customer_id'].isin(u_filt.index)]

    def _sparsify_uim(self):
        print('Creating Sparse UIM...')
        ## check columns names

        ## creating sparse user_item matrix
        self.cust_dict = {i: count for count, i in enumerate(self.transactions_filtered['customer_id'].unique())}
        self.prod_dict = {i: count for count, i in enumerate(self.transactions_filtered['product_id'].unique())}

        rows = [self.cust_dict[i] for i in self.transactions_filtered['customer_id']]
        cols = [self.prod_dict[i] for i in self.transactions_filtered['product_id']]
        vals = list(self.transactions_filtered['feedback'])
        sp = sparse.csr_matrix((vals, (rows, cols)), shape=(len(self.cust_dict), len(self.prod_dict)))
        # pd.DataFrame(sp.todense()).to_csv('uim.csv')
        # sp = sparse.csr_matrix(sp.todense() * 40 + 1)
        return sp

    def _make_train(self, pct_test=0.001):
        """Split UIM into training and test by masking user item combinations"""
        print('Building Training Set...')
        test_set = self.uim_sparse.copy()  # Make a copy of the original set to be the test set.
        test_set[test_set != 0] = 1  # Store the test set as a binary preference matrix
        training_set = self.uim_sparse.copy()  # Make a copy of the original data we can alter as our training set.
        nonzero_inds = training_set.nonzero()  # Find the indices in the ratings data where an interaction exists
        nonzero_pairs = list(
            zip(nonzero_inds[0], nonzero_inds[1]))  # Zip these pairs together of user,item index into list
        random.seed(0)  # Set the random seed to zero for reproducibility
        num_samples = int(
            np.ceil(pct_test * len(nonzero_pairs)))  # Round the number of samples needed to the nearest integer
        samples = random.sample(nonzero_pairs, num_samples)  # Sample a random number of user-item pairs without replacement
        user_inds = [index[0] for index in samples]  # Get the user row indices
        item_inds = [index[1] for index in samples]  # Get the item column indices
        training_set[user_inds, item_inds] = 0  # Assign all of the randomly chosen user-item pairs to zero
        training_set.eliminate_zeros()  # Get rid of zeros in sparse array storage after update to save space

        # training_set = sparse.csr_matrix(training_set.todense() * 40 + 1) # adding confidence level to sparse training
        return training_set, test_set, list(set(user_inds))  # Output the unique list of user rows that were altered

    def _init_model(self):
        """Initializing and training model: Implicit ALS"""
        print('Training Implicit Model')
        assert self.model_fact is not None, 'Number of latent factors cannot be None'
        assert self.model_iter is not None, 'Number of iterations cannot be None'

        # initialize & train the model
        model = implicit.als.AlternatingLeastSquares(factors=self.model_fact, num_threads=7, iterations=self.model_iter)
        model.fit(self.tr.T)
        return model

    def recommend(self, user, k=10):
        return self.model.recommend(user, self.tr, N=k)

    def ranking(self, user, n_recs):
        ranks = self.model.rank_items(userid=user, user_items=self.tr,
                                      selected_items=list(self.prod_dict.values()))
        preds = np.zeros(len(self.prod_dict))
        for count, r in enumerate(ranks):
            if count < n_recs:
                preds[r[0]] = 1
            else:
                preds[r[0]] = 0
        return preds

    def eval(self):
        """Evaluating the recommendations by letting the model recommend on masked users"""
        # recommendations = self.model.recommend_all(user_items=self.tr, N=5, filter_already_liked_items=True)
        l1 = []
        l2 = []
        l3 = []

        # y_star = self.te[self.masked_uid]
        # rec = self.model.recommend_all(user_items=self.tr, N=5)[self.masked_uid, :]

        for count, i in enumerate(self.masked_uid):
            # l1_start = time.time()
            y_star = self.te[i].toarray().flatten()
            # y_star = self.te[i]
            # l1.append(time.time() - l1_start)

            # l2_start = time.time()
            y_hat = self.ranking(user=i, n_recs=self.n_recs).astype('int')  # gives boolean user product array indicating recs
            # y_hat = sparse.csr_matrix(y_hat)
            # l2.append(time.time() - l2_start)

            # l3_start = time.time()
            # self.auc = sequential_mean(old=self.auc, new=auc_score(y=y_star, y_hat=y_hat), iteration=count + 1)
            # self.precision = sequential_mean(old=self.precision, new=precision_score(y=y_star, y_hat=y_hat),
            #                                  iteration=count + 1)

            self.recall = sequential_mean(old=self.recall, new=recall_score(y=y_star, y_hat=y_hat),
                                          iteration=count + 1)

            # self.f1 = sequential_mean(old=self.f1, new=f1_score(y=y_star, y_hat=y_hat), iteration=count + 1)
            # l3.append(time.time() - l3_start)

            if (count % 1000) == 0:
                print('#### ---- ####')
                # print(f'AUC: {self.auc}')
                # print(f'Precision: {self.precision}')
                print(f'Recall: {self.recall}')
                # print(f'F1: {self.f1}')
                print(f'{np.round(count / len(self.masked_uid) * 100, 3)}% done')

        # print(f'Total Time: {np.sum(l1) + np.sum(l2) + np.sum(l3)}')
        # print(f'L1: {np.round(np.sum(l1) / (np.sum(l1) + np.sum(l2) + np.sum(l3))*100, 3)}%')
        # print(f'L2: {np.round(np.sum(l2) / (np.sum(l1) + np.sum(l2) + np.sum(l3))*100, 3)}%')
        # print(f'L3: {np.round(np.sum(l3) / (np.sum(l1) + np.sum(l2) + np.sum(l3))*100, 3)}%')
        return self.auc, self.precision, self.recall, self.f1


class BaselinePopularity(BaselineImplicit):
    def __init__(self, transactions, min_items=5, max_items=50, n_recs=5):
        super().__init__(transactions, min_items=5, max_items=50, n_recs=5)

    def _init_model(self):
        """Compute list of most popular items"""
        item_popularity = self.tr.sum(axis=0).flat
        return np.argsort(item_popularity)  # position is index of respective sorted position

    def ranking(self, user, n_recs):
        """Ranking = list of most popular items"""
        preds = np.zeros(len(self.prod_dict))

        count = 1
        while count < n_recs:
            preds[self.model[-count]] = 1
            count += 1

        return preds

    def recommend(self, user, k=10):
        """recommendation corresponds to most popular items in total sales"""

        return


def optim_objective(x, min_items=5, max_items=50, n_recs=5): #5 - 50 ## 50-1000
    """Objective Function for bayesian hyperparameter tuning"""
    model = BaselineImplicit(transactions=grouped_purchased, min_items=min_items,
                             max_items=max_items, n_recs=n_recs, model_fact=x, model_iter=15)
    auc, precision, recall, f1 = model.eval()
    return recall  # np.random.randint(low=0, high=50, size=1)


if __name__ == '__main__':
    # user item combinations df
    cleandata_m = pd.read_csv('../data/cleandata.csv')
    cleandata_m = cleandata_m[['CUSTOMER_ID', 'PRODUCT_ID']]
    cleandata_m['quantity'] = 1
    cleandata_m.columns = ['customer_id', 'product_id', 'feedback']
    grouped_purchased = cleandata_m.groupby(by=['customer_id', 'product_id']).sum().reset_index()

    X = np.array([100, 500, 1000, 1500])
    bounds = [(0, 2000)]

    model = BayesianOptEI(X=X, model=GaussianProcessRegressor(kernel=RBF()), exploration=1e-1, n_target_obs=20,
                          objective_f=optim_objective, objective_bounds=bounds)
    opt, xs, ys = model.optimize()


    plt.title('Model Performance per n_factors')
    plt.ylabel('Recall')
    plt.xlabel('Number of latent factors')
    plt.scatter(np.array(xs).flatten(), np.array(ys).flatten())
    plt.show()

    ## Testing for factor size
    # auc_l, precision_l, recall_l, f1_l = [], [], [], []
    # lis = [1000]  #[50, 100, 200, 400, 800, 1000]
    # for i in lis:
    #     model = BaselineImplicit(transactions=grouped_purchased, min_items=5, max_items=50, n_recs=5,
    #                              model_fact=1000, model_iter=5)
    #
    #     auc, precision, recall, f1 = model.eval()
    #     auc_l.append(auc)
    #     precision_l.append(precision)
    #     recall_l.append(recall)
    #     f1_l.append(f1)

    # plt.title('Testing for Factor Size')
    # plt.xticks(np.arange(len(lis)), lis)
    # plt.plot(auc_l, label='AUC')
    # plt.plot(precision_l, label='Precision')
    # plt.plot(recall_l, label='Recall')
    # plt.plot(f1_l, label='F1')
    # plt.legend()
    # plt.show()

    print('hello world')

# <7607x40529 sparse matrix of type '<class 'numpy.int64'>'
# 	with 308304103 stored elements in Compressed Sparse Column format>