import pandas as pd
from src.model_framework import RecommenderFramework
import json
import os
import numpy as np
from scipy.spatial import distance


class CollFilt(RecommenderFramework):
    def __init__(self, user_df, item_df, user_id, item_id, min_item_n, min_trans_n,
                 inv_by_cust_dict, prod_by_inv_dict, imp_similarity=False):
        super().__init__(user_df, item_df, user_id, item_id, min_item_n, min_trans_n,
                         inv_by_cust_dict, prod_by_inv_dict)

        # Creating User Item Matrix
        self.user_item_m = None  # collecting user-item pair relevance
        self._user_item_matrix()  # create user-item matrix on object initialization

        # # Creating/ Importing User Similarity Matrix
        # self.user_sim_matrix = None
        # if imp_similarity:
        #     self.user_sim_matrix = pd.read_csv("../data/user_sim_matrix.csv").to_numpy()
        # if not imp_similarity:
        #     self._user_sim_matrix()

    def _user_item_relevance(self, cust_id, prod_id):
        """computing relevance score for one user-item pair"""
        # collect all purchased products for the given customer id
        prod_purchased = []
        for inv in self.inv_by_cust_dict[cust_id]:
            prod_purchased += self.prod_by_inv_dict[str(inv)]
        # return 1 or 0 on purchase condition
        if str(prod_id) in prod_purchased:
            return 1
        elif str(prod_id) not in prod_purchased:
            return 0

    def _user_item_matrix(self):
        """creating user-item-matrix"""
        print('Generating User-Item-Matrix...')
        user_item_dict = {}
        for user in self.user_dict.keys():
            user_items = []
            for inv in self.inv_by_cust_dict[user]:
                user_items += self.prod_by_inv_dict[str(inv)]
            user_item_dict[user] = set(user_items)
        # create 0 matrix and fill user product combinations
        self.user_item_m = np.zeros(shape=(len(self.user_dict.keys()), len(self.item_dict.keys())))
        for user in user_item_dict:
            self.user_item_m[self.user_dict[user], [self.item_dict[prod] for prod in self.user_item_dict[user]]] = 1

    def _user_sim_matrix(self):
        print('Generating User Similarity Matrix...')
        self.user_sim_matrix = np.zeros(shape=(len(self.user_dict), len(self.user_dict)))
        progress = 0

        self.user_sim_matrix = distance.pdist(self.user_df.to_numpy(), lambda u, v: self._count_dist(u, v, norm=True))
        self.user_sim_matrix = distance.squareform(self.user_sim_matrix)

        # for count, user in enumerate(self.user_dict):
        #     self.user_sim_matrix[self.user_dict[user], :] = [self._u_sim(user, u) for u in self.user_dict]
        #     if np.round(count / len(self.user_dict), 2) == progress + .01:
        #         progress += .01
        #         print(f'{progress * 100}% done...')
        # pd.DataFrame(self.user_sim_matrix).to_csv("./data/user_sim_matrix.csv")
        # print('Similarity Matrix Done & Saved!')

    def _avg_user_relevance(self, u):
        return np.mean(self.user_item_m[self.user_dict[u], :])

    def _u_sim(self, u1, u2):
        """Compute similarity of two given user profiles"""
        return self._count_dist(self.user_item_m[self.user_dict[u1], :],
                                self.user_item_m[self.user_dict[u2], :],
                                norm=False)

    def _cf_item_score(self, query_u, query_i):
        """Compute relevance for a given user and item"""
        numerator = np.sum([self.user_item_m[self.user_dict[user_], self.item_dict[query_i]] -
                            self._avg_user_relevance(user_) *
                            self.user_sim_matrix[self.user_dict[query_u], self.user_dict[user_]]
                            for user_ in self.user_dict.keys()]
                           )
        denominator = np.sum(self.user_sim_matrix[self.user_dict[query_u], :]) - self._u_sim(query_u, query_u)
        return self._avg_user_relevance(query_u) * (numerator / denominator)

    def make_recommendation(self, n=10, user_id=None, user_profile=None):
        """make recommendation for one user based on computed item scores"""
        not_purchased = [p for p in self.item_dict.keys() if p not in self.user_item_dict[user_id]]
        new_ratings = [self._cf_item_score(user_id, i) for i in not_purchased]

        rated = pd.DataFrame({'item': not_purchased, 'rating': new_ratings}).sort_values(by='rating', ascending=False)
        return rated.iloc[:n, 0]


if __name__ == '__main__':

    for i in range(1, 6):
        print('Generating matrices for group ', i)

        user_m = pd.read_csv(os.getcwd() + f"/../data/group{i}/user_m.csv")
        item_m = pd.read_csv(os.getcwd() + f"/../data/group{i}/item_m.csv")
        #cleandata_m = pd.read_csv('../data/cleandata.csv')

        with open(os.getcwd() + f"/../data/group{i}/invoice_by_customer_dict.json") as json_file:
            invoice_by_customer_dict = json.load(json_file)

        print('initializing model')

        rec = CollFilt(user_df=user_m, item_df=item_m, user_id='CUSTOMER_ID', item_id='PRODUCT_ID', min_trans_n=1,
                       min_item_n=1, inv_by_cust_dict=invoice_by_customer_dict, prod_by_inv_dict=product_by_invoice_dict,
                       imp_similarity=False)

        pd.DataFrame(rec.user_item_m).to_csv(os.getcwd() + f"/../data/group{i}/user_item_m.csv")
        # rec.user_sim_matrix.to_csv(os.getcwd() + f"/../data/group{i}/user_sim_m.csv", index=False)

    # rec.make_recommendation(user_id='//VNwGkmnK8q2RMfoYb0dqUuGJNfP+hNs5117i4DtYw=')
    #
    # # rec.eval()
    #
    # print('hello world')
