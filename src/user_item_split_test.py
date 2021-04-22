import unittest
import os, pandas as pd


class TestOneHotData(unittest.TestCase):

    def test_item_mtx(self):
        item_m = pd.read_csv(os.getcwd() + "/../data/item_m.csv")
        self.assertEqual(item_m.PRODUCT_ID.size, item_m.PRODUCT_ID.drop_duplicates().size)

    def test_user_mtx(self):
        user_m = pd.read_csv(os.getcwd() + "/../data/user_m.csv")
        self.assertEqual(user_m.CUSTOMER_ID.size, user_m.CUSTOMER_ID.drop_duplicates().size)


if __name__ == '__main__':
    unittest.main()
