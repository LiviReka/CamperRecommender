import pandas as pd
from scipy.spatial import distance


class RecommenderFramework:
    def __init__(self, user_df, item_df, user_item_df, user_id, item_id):
        self.user_df, self.item_df, self.user_item_df = user_df, item_df, user_item_df  # user, item & user-item matrix
        self.user_id, self.item_id = user_id, item_id # columns that contains user & item identifiers

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
    def _eucledian_dist(u, v):
        """Eucledian Distance between two arrays"""
        return distance.euclidean(u, v)

    def _train_test_split(self):
        """Separate User-Item-Matrix in train test split"""
        return

    def _eval(self):
        """Evaluate Recommendations based on validation set"""
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
