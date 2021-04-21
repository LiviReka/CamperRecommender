from scipy.spatial import distance
import numpy as np
import random


class RecommenderFramework:
    def __init__(self, user_df, item_df, user_item_df, user_id, item_id, min_item_n, min_trans_n,
                 inv_by_cust_dict, prod_by_inv_dict):
        self.user_df, self.item_df, self.user_item_df = user_df, item_df, user_item_df  # user, item & user-item matrix
        self.user_id, self.item_id = user_id, item_id  # columns that contains user & item identifiers
        self.min_item_n, self.min_trans_n = min_item_n, min_trans_n  # define the min# of trans and items to keep user
        self.inv_by_cust_dict, self.prod_by_inv_dict = inv_by_cust_dict, prod_by_inv_dict

        self._select_users()
        self._identifiers()

    def _identifiers(self):
        """Create Identifier dicts for users & items to respective rows"""
        # dict to index user identifier & row
        self.user_dict, self.item_dict = {i: count for count, i in enumerate(self.user_df[self.user_id])},\
                                         {i: count for count, i in enumerate(self.item_df[self.item_id])}
        # drop identifier columns from original matrices
        self.user_df.drop([self.user_id], axis=1, inplace=True)
        self.item_df.drop([self.item_id], axis=1, inplace=True)

    @staticmethod
    def _cosine_dist(u, v):
        """Cosine Distance between two arrays"""
        return distance.cosine(u, v)

    @staticmethod
    def _euclidean_dist(u, v):
        """Euclidean Distance between two arrays"""
        return distance.euclidean(u, v)

    @staticmethod
    def _count_dist(u, v):
        """Custom Similarity Measure that directly compares 0/1 matches of vectors"""
        return np.round(sum(u * v) / ((len(u) == len(v))*len(u)), 4)

    def eval(self, invoice_by_customer_dict, product_by_invoice_dict):
        """Evaluate Recommendations based on validation set"""
        n = 1000  # number of evaluation iterations
        correctness = []

        for count, i in enumerate(range(n)):
            print(f'Validation Progress: {count/n}')
            # choose random user id to test on
            check_user_id = random.choice(list(self.user_dict.keys()))

            # random chose invoice that is attributed to user
            check_invoice_id = random.choice(invoice_by_customer_dict[check_user_id])

            # random chose product that the user purchased
            check_product_id = random.choice(product_by_invoice_dict[str(check_invoice_id)])

            # get all other product ids that the user purchased
            # for invoice in invoice_by_customer_dict[check_user_id]:

            prod_id_list = [product_by_invoice_dict[str(i)] for i in invoice_by_customer_dict[check_user_id]]
            prod_id_list = list(np.concatenate(prod_id_list).flat)
            prod_id_list.remove(check_product_id)

            # get attributes from all other products purchased by the user
            item_indices = set([self.item_dict[item] for item in prod_id_list])
            item_attributes = self.item_df.loc[item_indices, :]

            # aggregate attributes to user profile
            user_profile = item_attributes.max().drop(columns=['PRODUCT_ID'])

            # compare to prediction
            recommendations = self.make_recommendation(n=10, user_profile=user_profile)

            if check_product_id in recommendations['product_id'].values:
                correctness.append(1)
            else:
                correctness.append(0)
            print(f'{np.sum(correctness) / len(correctness) * 100}%')
        print(f'Accuracy: {np.sum(correctness)/len(correctness)*100}%')
        return

    def _select_users(self):
        """Filter Users for number of products and invoices as specified"""
        print(self.min_trans_n)
        users = [user for user in self.inv_by_cust_dict if
                 len(self.inv_by_cust_dict[user]) >= self.min_trans_n]
        for user in users:
            items_per_user = []
            for inv in self.inv_by_cust_dict[user]:
                items_per_user += self.prod_by_inv_dict[str(inv)]
            if len(items_per_user) < self.min_item_n:
                users.remove(user)
        print(f'{len(users)/len(self.inv_by_cust_dict.keys())*100}% of all users.')
        self.user_df = self.user_df[self.user_df[self.user_id].isin(users)]
