import pandas as pd
from scipy.spatial import distance


class RecommenderFramework:
    def __init__(self, user_path, item_path, user_item_path):
        self.user_m = pd.read_csv(user_path)
        self.item_m = pd.read_csv(item_path)
        self.user_item_m = pd.read_csv(user_item_path)

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



