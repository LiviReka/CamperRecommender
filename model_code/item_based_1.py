import pandas as pd
from src.model_framework import RecommenderFramework
import json


class ItemBased(RecommenderFramework):
    def __init__(self, user_df, item_df, clean_data, user_item_df, user_id, item_id):
        super().__init__(user_df, item_df, clean_data, user_item_df, user_id, item_id)

    def make_recommendation(self, user_id=None, user_profile=None):
        if user_id is not None and user_profile is not None:
            raise Exception('Recommendation must be based on either ID or Profile, but never both!')
        elif user_id is not None:
            user_index = self.user_dict[user_id]
            print('computing similarities...')
            # similarities_cos = [self._cosine_dist(self.user_df.iloc[user_index, :], i[1]) for i in
            #                     self.item_df.iterrows()]
            # similarities_euc = [self._euclidean_dist(self.user_df.iloc[user_index, :], i[1]) for i in
            #                     self.item_df.iterrows()]
            similarities_count = [self._count_dist(self.user_df.iloc[user_index, :], i[1]) for i in
                                  self.item_df.iterrows()]
        elif user_profile is not None:
            print('computing similarities...')
            # similarities_cos = [self._cosine_dist(user_profile, i[1]) for i in
            #                     self.item_df.iterrows()]
            # similarities_euc = [self._euclidean_dist(user_profile, i[1]) for i in
            #                     self.item_df.iterrows()]
            similarities_count = [self._count_dist(user_profile, i[1]) for i in
                                  self.item_df.iterrows()]

        # import matplotlib.pyplot as plt
        # plt.title('Cosin Sim')
        # plt.hist(similarities_cos)
        # plt.show()
        # plt.title('Eucl Sim')
        # plt.hist(similarities_euc)
        # plt.show()
        # plt.title('Count Sim')
        # plt.hist(similarities_count)
        # plt.show()

        return []


if __name__ == '__main__':
    user_m = pd.read_csv('../data/user_m.csv')
    item_m = pd.read_csv('../data/item_m.csv')
    cleandata_m = pd.read_csv('../data/cleandata.csv')

    with open('../data/invoice_by_customer_dict.json') as json_file:
        invoice_by_customer_dict = json.load(json_file)

    with open('../data/product_by_invoice_dict.json') as json_file:
        product_by_invoice_dict = json.load(json_file)

    rec = ItemBased(user_df=user_m, item_df=item_m, clean_data=cleandata_m, user_item_df=None,
                    user_id='CUSTOMER_ID', item_id='PRODUCT_ID')

    rec.eval(invoice_by_customer_dict, product_by_invoice_dict)

    rec.make_recommendation('+6F9xZ4in/zBbuY9ZysWwnoH0yxqbg7GFSK/9wNk+24=')

    print('hello world')