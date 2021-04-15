import pandas as pd
from src.model_framework import RecommenderFramework


class ItemBased(RecommenderFramework):
    def __init__(self, user_df, item_df, user_item_df, user_id, item_id):
        super().__init__(user_df, item_df, user_item_df, user_id, item_id)

    def make_recommendation(self, user_id):
        user_index = self.user_dict[user_id]
        print('computing similarities...')
        # similarities = [self._cosine_dist(self.user_df.iloc[user_index, :], i) for i in self.item_df]
        # print(similarities)

        similarities = [self._cosine_dist(self.user_df.iloc[user_index, :], i[1]) for i in self.item_df.iterrows()]


if __name__ == '__main__':
    user_m = pd.read_csv('../data/user_m.csv')
    item_m = pd.read_csv('../data/item_m.csv')
    rec = ItemBased(user_df=user_m, item_df=item_m, user_item_df=None, user_id='CUSTOMER_ID', item_id='PRODUCT_ID')

    rec.make_recommendation('+6F9xZ4in/zBbuY9ZysWwnoH0yxqbg7GFSK/9wNk+24=')

    print('hello world')