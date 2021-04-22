import pandas as pd
from src.model_framework import RecommenderFramework
import json
import numpy as np


class CollFilt(RecommenderFramework):
    def __init__(self, user_df, item_df, user_id, item_id, min_item_n, min_trans_n,
                 inv_by_cust_dict, prod_by_inv_dict):
        super().__init__(user_df, item_df, user_id, item_id, min_item_n, min_trans_n,
                         inv_by_cust_dict, prod_by_inv_dict)

        print('init cf')
        self.user_item_m = None  # collecting user-item pair relevance

        self._user_item_matrix()  # create user-item matrix on object initialization

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

    #'//VNwGkmnK8q2RMfoYb0dqUuGJNfP+hNs5117i4DtYw=', 'K200313-009'   - 8 products purchased
    def _user_item_matrix(self):
        """creating user-item-matrix"""
        print('Generating User-Item-Matrix...')
        user_item_dict = {}
        for user in self.user_dict.keys():
            user_items = []
            for inv in self.inv_by_cust_dict[user]:
                user_items += self.prod_by_inv_dict[str(inv)]
            user_item_dict[user] = set(user_items)

        user_item_matrix = np.zeros(shape=(len(self.user_dict.keys()), len(self.item_dict.keys())))

        ## alternative approach fill 0 matrix with purchased products!!!
        # per user iterate purchased products and fill respective items with relevance score

    def make_recommendation(self, n=10, user_id=None, user_profile=None):
        """make recommendation based on user similarity in the user-item matrix"""
        return


if __name__ == '__main__':
    user_m = pd.read_csv('../data/user_m.csv')
    item_m = pd.read_csv('../data/item_m.csv')
    cleandata_m = pd.read_csv('../data/cleandata.csv')

    with open('../data/invoice_by_customer_dict.json') as json_file:
        invoice_by_customer_dict = json.load(json_file)

    with open('../data/product_by_invoice_dict.json') as json_file:
        product_by_invoice_dict = json.load(json_file)

    print('initializing model')

    rec = CollFilt(user_df=user_m, item_df=item_m, user_id='CUSTOMER_ID', item_id='PRODUCT_ID', min_trans_n=2,
                   min_item_n=5, inv_by_cust_dict=invoice_by_customer_dict, prod_by_inv_dict=product_by_invoice_dict)

    rec.eval()

    print('hello world')
