from scipy.spatial import distance
import numpy as np
import random


class RecommenderFramework:
    def __init__(self, user_df, item_df, clean_data, user_item_df, user_id, item_id):
        self.user_df, self.item_df, self.user_item_df = user_df, item_df, user_item_df  # user, item & user-item matrix
        self.clean_df = clean_data
        self.user_id, self.item_id = user_id, item_id  # columns that contains user & item identifiers

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

    def _train_test_split(self):
        """Separate User-Item-Matrix in train test split"""
        return

    def eval(self, invoice_by_customer_dict, product_by_invoice_dict):
        """Evaluate Recommendations based on validation set"""
        n = 1000  # number of evaluation iterations

        for i in range(n):
            # choose random user id to test on
            check_user_id = random.choice(list(invoice_by_customer_dict.keys()))

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
            self.make_recommendation(user_profile=user_profile)
            # TODO: Make sure that recommendations are attributable to product IDs

        return

    def predict(self):
        """Make Recommendation for a given"""
        return


# class ItemBased(RecommenderFramework):
#     def __init__(self, user_path, item_path, user_item_path):
#         super().__init__(user_path, item_path, user_item_path)
#
#     def fit_recommender(self):
#
#         return
