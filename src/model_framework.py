from scipy.spatial import distance
import numpy as np
import random
import pandas as pd


class RecommenderFramework:
    def __init__(self, user_df, item_df, user_id, item_id, min_item_n, min_trans_n,
                 inv_by_cust_dict, prod_by_inv_dict):
        self.user_df, self.item_df = user_df, item_df  # user, item & user-item matrix
        self.user_id, self.item_id = user_id, item_id  # columns that contains user & item identifiers
        self.min_item_n, self.min_trans_n = min_item_n, min_trans_n  # define the min# of trans and items to keep user

        # index for cust, inv & prod (containing all users and item, also if excluded for learning, info is filtered
        # respectively in self._identifiers())
        self.inv_by_cust_dict, self.prod_by_inv_dict = inv_by_cust_dict, prod_by_inv_dict

        self._select_users()  # calling user selection based on conditions
        self._identifiers()  # calling user identifier init

    def _identifiers(self):
        """Create Identifier dicts for users & items to respective rows"""
        print('Customer Identifiers...')
        # dict to index user identifier & row (this dict contains ONLY the users & items considered for the learning)
        self.user_dict, self.item_dict = {i: count for count, i in enumerate(self.user_df[self.user_id].values)},\
                                         {i: count for count, i in enumerate(self.item_df[self.item_id].values)}
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
    def _jaccard(u, v):
        return distance.jaccard(u, v)

    @staticmethod
    def _count_dist(u, v, norm=True):
        """Custom Similarity Measure that directly compares 0/1 matches of vectors"""
        if norm:
            return np.log((sum(u * v) / len(u)) + 1)
        elif not norm:
            return np.sum(u * v)

    def eval(self):
        """Evaluate Recommendations based on validation set"""
        n = 1000  # number of evaluation iterations
        correctness = []

        for count, i in enumerate(range(n)):
            print(f'Validation Progress: {count/n}')
            # choose random user id to test on
            check_user_id = random.choice(list(self.user_dict.keys()))

            # random chose invoice that is attributed to user
            check_invoice_id = random.choice(self.inv_by_cust_dict[check_user_id])

            # random chose product that the user purchased
            check_product_id = random.choice(self.prod_by_inv_dict[str(check_invoice_id)])

            # get all other product ids that the user purchased
            prod_id_list = [self.prod_by_inv_dict[str(i)] for i in self.inv_by_cust_dict[check_user_id]]
            prod_id_list = list(np.concatenate(prod_id_list).flat)
            prod_id_list.remove(check_product_id)

            # get attributes from all other products purchased by the user
            item_indices = set([self.item_dict[item] for item in prod_id_list])
            item_attributes = self.item_df.iloc[list(item_indices), :]

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

    def _user_item_count(self, user):
        return sum(list(map(lambda x: len(self.prod_by_inv_dict[str(x)]), self.inv_by_cust_dict[user])))

    def _select_users(self):
        """Filter Users for number of products and invoices as specified"""
        print('Customer Filtering...')
        # filter users by minimum number of invoices in db using cust -> inv dict
        users = [user for user in self.user_df[self.user_id] if
                 len(self.inv_by_cust_dict[user]) >= self.min_trans_n]

        # filter (already filtered) users by number of purchased products
        n_items_df = pd.DataFrame({'users': users, 'n_items': list(map(lambda x: self._user_item_count(x), users))})
        n_items_df = n_items_df[n_items_df['n_items'] >= self.min_item_n]

        print(f'{np.round(len(n_items_df["users"])/len(self.inv_by_cust_dict.keys())*100, 3)}% of all users.')
        self.user_df = self.user_df[self.user_df[self.user_id].isin(n_items_df['users'])]  # update users dataframe

        # compute user-item dict
        self.user_item_dict = {}
        total_items = []
        for user in self.user_df[self.user_id].values:
            user_items = []
            for inv in self.inv_by_cust_dict[user]:
                user_items += self.prod_by_inv_dict[str(inv)]
                total_items += self.prod_by_inv_dict[str(inv)]
            self.user_item_dict[user] = set(user_items)
        total_items = set(total_items)

        self.item_df = self.item_df[self.item_df[self.item_id].isin(total_items)]  # update items dataframe
