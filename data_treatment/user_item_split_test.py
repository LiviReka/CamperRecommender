import unittest
import os, pandas as pd
import pandas.testing as pdtest


class TestOneHotData(unittest.TestCase):

    def test_item_mtx(self):
        item_m = pd.read_csv(os.getcwd() + "/../data/item_m.csv")
        print(item_m[item_m.duplicated(['PRODUCT_ID'])].sort_values('PRODUCT_ID'))
        pdtest.assert_series_equal(item_m.PRODUCT_ID, item_m.PRODUCT_ID.drop_duplicates())


    def test_user_mtx(self):
        user_m = pd.read_csv(os.getcwd() + "/../data/user_m.csv")
        pdtest.assert_series_equal(user_m.CUSTOMER_ID, user_m.CUSTOMER_ID.drop_duplicates())


if __name__ == '__main__':
    unittest.main()
