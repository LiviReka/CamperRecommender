import pandas as pd
from src.model_framework import RecommenderFramework
import json


class ItemBased(RecommenderFramework):
    def __init__(self, user_df, item_df, user_item_df, user_id, item_id, min_item_n, min_trans_n,
                 inv_by_cust_dict, prod_by_inv_dict):
        super().__init__(user_df, item_df, user_item_df, user_id, item_id, min_item_n, min_trans_n,
                         inv_by_cust_dict, prod_by_inv_dict)

    def make_recommendation(self, n=10, user_id=None, user_profile=None):
        product_ids, similarities_count = None, None
        if user_id is not None and user_profile is not None:
            raise Exception('Recommendation must be based on either ID or Profile, but never both!')
        elif user_id is not None:
            user_index = self.user_dict[user_id]
            print('Computing similarities based on user id...')
            product_ids = list(self.item_dict.keys())
            similarities_count = [self._count_dist(self.user_df.iloc[user_index, :], i[1]) for i in
                                  self.item_df.iterrows()]
        elif user_profile is not None:
            print('Computing similarities based on user profile...')
            product_ids = list(self.item_dict.keys())
            similarities_count = [self._count_dist(user_profile, self.item_df.iloc[self.item_dict[prod_id], :])
                                  for prod_id in product_ids]
        return pd.DataFrame(data={'product_id': product_ids, 'similarities': similarities_count}
                            ).sort_values(by=['similarities'], ascending=False).iloc[:n, :]


if __name__ == '__main__':
    user_m = pd.read_csv('../data/user_m.csv')
    item_m = pd.read_csv('../data/item_m.csv')
    cleandata_m = pd.read_csv('../data/cleandata.csv')

    with open('../data/invoice_by_customer_dict.json') as json_file:
        invoice_by_customer_dict = json.load(json_file)

    with open('../data/product_by_invoice_dict.json') as json_file:
        product_by_invoice_dict = json.load(json_file)

    rec = ItemBased(user_df=user_m, item_df=item_m, user_item_df=None,
                    user_id='CUSTOMER_ID', item_id='PRODUCT_ID', min_trans_n=2, min_item_n=5,
                    inv_by_cust_dict=invoice_by_customer_dict, prod_by_inv_dict=product_by_invoice_dict)

    rec.eval(invoice_by_customer_dict, product_by_invoice_dict)

    print('hello world')