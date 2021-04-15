import pandas as pd
from src.model_framework import RecommenderFramework


class ItemBased(RecommenderFramework):
    def __init__(self, user_df, item_df, user_item_df, user_id, item_id):
        super().__init__(user_df, item_df, user_item_df, user_id, item_id)

    def make_recommendation(self, user_id):
        user_index = self.user_dict[user_id]
        print('computing similarities...')

        similarities_cos = [self._cosine_dist(self.user_df.iloc[user_index, :], i[1]) for i in self.item_df.iterrows()]
        similarities_euc = [self._eucledian_dist(self.user_df.iloc[user_index, :], i[1]) for i in self.item_df.iterrows()]
        similarities_count = [self._count_dist(self.user_df.iloc[user_index, :], i[1]) for i in self.item_df.iterrows()]
        import matplotlib.pyplot as plt
        plt.title('Cosin Sim')
        plt.hist(similarities_cos)
        plt.show()
        plt.title('Eucl Sim')
        plt.hist(similarities_euc)
        plt.show()
        plt.title('Count Sim')
        plt.hist(similarities_count)
        plt.show()


if __name__ == '__main__':
    user_m = pd.read_csv('../data/user_m.csv')
    item_m = pd.read_csv('../data/item_m.csv')
    rec = ItemBased(user_df=user_m, item_df=item_m, user_item_df=None, user_id='CUSTOMER_ID', item_id='PRODUCT_ID')

    # rec.eval()

    rec.make_recommendation('+6F9xZ4in/zBbuY9ZysWwnoH0yxqbg7GFSK/9wNk+24=')

    print('hello world')