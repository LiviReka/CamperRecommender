import pandas as pd
from src.model_framework import RecommenderFramework
import json
import numpy as np
from scipy.spatial import distance
from scipy import sparse
import surprise
from surprise import KNNBasic
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import NormalPredictor, SVD
import random


class CollFilt(RecommenderFramework):
    def __init__(self, user_df, item_df, user_id, item_id, min_item_n, min_trans_n,
                 inv_by_cust_dict, prod_by_inv_dict, imp_similarity=False):
        super().__init__(user_df, item_df, user_id, item_id, min_item_n, min_trans_n,
                         inv_by_cust_dict, prod_by_inv_dict)

        # Creating User Item Matrix
        self.user_item_m = None  # collecting user-item pair relevance
        # self._user_item_matrix()  # create user-item matrix on object initialization
        self.create_dataset()
        self.surprise_model()

    def create_dataset(self):
        ul = []
        il = []

        for u in self.user_item_dict:
            ul += [u] * len(self.user_item_dict[u])
            il += self.user_item_dict[u]
        ra = [1] * len(il)
        self.surp_dataset = pd.DataFrame(data={'userID': ul, 'itemID': il, 'rating': ra})

    def surprise_model(self):
        reader = surprise.reader.Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(self.surp_dataset[['userID', 'itemID', 'rating']], reader=reader)

        cross_validate(SVD(), data, cv=2)

        # Retrieve the trainset.

        trainset = data.build_full_trainset()

        # Build an algorithm, and train it.
        # algo = KNNBasic()
        algo = SVD()
        algo.fit(trainset)
        algo.predict(uid=random.choice(list(self.user_dict.keys())), iid=random.choice(list(self.item_dict.keys())))


if __name__ == '__main__':
    user_m = pd.read_csv('../data/user_m.csv')
    item_m = pd.read_csv('../data/item_m.csv')
    cleandata_m = pd.read_csv('../data/cleandata.csv')

    with open('../data/invoice_by_customer_dict.json') as json_file:
        invoice_by_customer_dict = json.load(json_file)

    with open('../data/product_by_invoice_dict.json') as json_file:
        product_by_invoice_dict = json.load(json_file)

    print('initializing model')

    rec = CollFilt(user_df=user_m, item_df=item_m, user_id='CUSTOMER_ID', item_id='PRODUCT_ID', min_trans_n=5,
                   min_item_n=5, inv_by_cust_dict=invoice_by_customer_dict, prod_by_inv_dict=product_by_invoice_dict,
                   imp_similarity=False)

    rec.make_recommendation(user_id='//VNwGkmnK8q2RMfoYb0dqUuGJNfP+hNs5117i4DtYw=')

    print('hello world')