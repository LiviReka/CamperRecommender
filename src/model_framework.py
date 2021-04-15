import pandas as pd
from scipy.spatial import distance
import random
import numpy as np


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

    @staticmethod
    def _count_dist(u, v):
        """Custom Similarity Measure that directly compares 0/1 matches of vectors"""
        return np.round(sum(u * v) / ((len(u) == len(v))*len(u)), 4)

    def _train_test_split(self):
        """Separate User-Item-Matrix in train test split"""
        return

    def eval(self):
        """Evaluate Recommendations based on validation set"""
        deviation_list_sq = []
        deviation_list_ab = []
        n = 1000  # number of evaluation iterations

        # for i in range(0, n):
        #     random_u = random.choice(self.user_df)  # select a random user from user pool
        #     user_ratings_real = matrix.loc[random_u]  # extract documented ratings for chosen user
        #     items_rated = user_ratings_real[user_ratings_real > 0].index  # extract items that have been rated by user
        #     test_item = random.choice(items_rated)  # choose random item to evaluate from rated items
        #
        #     original_rating = matrix.loc[random_u, test_item]  # define originally documented rating
        #     matrix_mock = matrix.copy()  # define copy as evaluation matrix
        #     matrix_mock.loc[random_u, test_item] = 0  # manipulate mock matrix as if item has never been rated by user
        #
        #     #### Algo 1 ####
        #     # predicted_score = recommender(matrix_mock, random_u)[1][test_item]
        #     #### Algo 2 ####
        #     # predicted_score = getRecommendations(matrix_mock.transpose(), random_u, corr_matrix)[1][test_item]
        #     #### Algo 3 ####
        #     predicted_score = getRecommendations2(matrix_mock, random_u, corr_matrix)[1][test_item]
        #
        #     deviation_list_sq.append(
        #         (original_rating - predicted_score) ** 2)  # append squared deviation to deviation list (squared)
        #     deviation_list_ab.append(
        #         (original_rating - predicted_score))  # append squared deviation to deviation list (not squared)
        #
        # mse = sum(deviation_list_sq) / n
        # mae = sum(deviation_list_ab) / n
        #
        # print(f'Mean Squared Error = {mse}')
        # print(f'Mean Absolute Error = {mae}')
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
