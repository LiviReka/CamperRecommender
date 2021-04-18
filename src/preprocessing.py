import pandas as pd
import os
import numpy as np
import re


class Data:
    def __init__(self, data):
        self.data_in = data
        self.english_cols = {'FACTURA_ID': 'INVOICE_ID', 'FACTURA_POSICION_ID': 'INVOICE_POSITION_ID',
                             'CUSTOMER_ID': 'CUSTOMER_ID',
                             'FACTURA_CLASE_DOCUMENTO_ID': 'INVOICE_DOCUMENT_CLASS_ID',
                             'ANO_MES_FACTURA': 'INVOICE_MONTH_YEAR',
                             'ANO_FACTURA': 'ORDER_YEAR', 'MES_FACTURA': 'ORDER_MONTH', 'FECHA_FACTURA': 'INVOICE_DATE',
                             'IMP_VENTA_NETO_EUR': 'NET_SALES_EUR', 'CANAL_VENTA_ID': 'SALES_CHANNEL_ID',
                             'CANAL_VENTA_DESC': 'SALES_CHANNEL_DESC', 'TEMPORADA_COMERCIAL_ID': 'SEASON_ID',
                             'TEMPORADA_COMERCIAL_DESC': 'SEASON_DESC', 'PRODUCTO_ID': 'PRODUCT_ID', 'TALLA': 'SIZE',
                             'MATERIAL_ID': 'MATERIAL_ID', 'NUMERO_DEUDOR_PAIS_ID': 'SALE_COUNTRY_ID',
                             'NUMERO_DEUDOR_PAIS_DESC': 'SALE_COUNTRY_DESC', 'VENTA_DEVOLUCION': 'SALES_FILTER',
                             'JERARQUIA_PROD_ID': 'PRODUCT_HIERARCHY_ID',
                             'GRUPO_ARTICULO_PRODUCTO_ID': 'PRODUCT_GROUP_ID',
                             'GRUPO_ARTICULO': 'PRODUCT_GROUP', 'CONCEPTO': 'CONCEPT', 'LINEA': 'LINE',
                             'GENERO_PRODUCTO': 'PRODUCT_GENDER', 'CATEGORIA': 'PRODUCT_CATEGORY',
                             'TIPOLOGIA': 'PRODUCT_TYPE',
                             'COLOR': 'COLOR_INTERN', 'CONSUMER_COLOR': 'COLOR_EXTERN', 'CREMALLERA': 'ZIPPER',
                             'CORDONES': 'LACES',
                             'OUTSOLE_SUELA_TIPO': 'SOLE_TYPE', 'OUTSOLE_SUELA_SUBTIPO': 'SOLE_SUBTYPE',
                             'PLANTILLA_EXTRAIBLE': 'REMOVABLE_SOLE', 'CONTACTO_SN': 'PHYSICAL_CONTACT',
                             'EDAD_SN': 'AGE_AVAILABLE',
                             'GENERO_CONTACTO': 'GENDER_AVAILABLE', 'EDAD_COMPRA': 'AGE_AT_PURCHASE',
                             'EDAD_RANGO_COMPRA': 'AGE_RANGE',
                             'PAIS_CONTACTO': 'COUNTRY_CONTACT_ID', 'PAIS_CONTACTO_DESC': 'COUNTRY_CONTACT_DESC',
                             'CIUDAD_CONTACTO': 'CITY_CONTACT', 'IDIOMA_CONTACTO': 'LANGUAGE_CONTACT'}
        self.data_out = self.preprocess(self.make_bools(self.to_english()))
        self.dropcols = ['INVOICE_MONTH_YEAR', 'ORDER_YEAR', 'ORDER_MONTH', 'SALES_CHANNEL_ID',
                         'SALES_CHANNEL_DESC', 'AGE_AVAILABLE', 'MATERIAL_ID', 'COUNTRY_CONTACT_DESC']

    def to_english(self):
        d_copy = self.data_in.rename(columns=self.english_cols)

        d_copy['REMOVABLE_SOLE'] = d_copy['REMOVABLE_SOLE'].apply(lambda x: True if x == 'Extraible' else False)

        invoice_id_dict = {'ZTON': 'Sale', 'ZDVN': 'Return', 'ZDAN': 'Cancelled Return'}
        d_copy['INVOICE_DOCUMENT_CLASS_ID'] = d_copy['INVOICE_DOCUMENT_CLASS_ID'].apply(lambda x: invoice_id_dict[x])

        product_group_dict = {'Zapatos Adulto': 'Adult Shoes',
                              'Bolsos': 'Bag',
                              'Bolsos Cartujano': 'Bag',
                              'Ropa': 'Clothings',
                              'Complementos': 'Complements'
                              }
        d_copy.PRODUCT_GROUP = d_copy.PRODUCT_GROUP.apply(
            lambda x: product_group_dict[x] if x in product_group_dict else x)

        color_dict = {'Мульти ассорти': 'Multi - Assorted',
                      'красный': 'Red',
                      'розовый': 'Pink',
                      'желтый': 'Yellow'
                      }
        d_copy.COLOR_INTERN = d_copy.COLOR_INTERN.apply(lambda x: color_dict[x] if x in color_dict else x)

        return d_copy

    def make_bools(self, d):
        d.ZIPPER = d.ZIPPER.apply(lambda x: True if x in ('SI', 'YES') else False)
        d.LACES = d.LACES.apply(lambda x: True if x in ('With laces', 'Con cordones') else False)
        d.PHYSICAL_CONTACT = d.PHYSICAL_CONTACT.apply(lambda x: True if x == 'S' else False)
        return d

    def preprocess(self, d):
        d['INVOICE_DATE'] = pd.to_datetime(d.INVOICE_DATE)
        d.AGE_AT_PURCHASE = d.AGE_AT_PURCHASE.replace(0, np.nan)
        d.SEASON_DESC = d.SEASON_DESC.apply(lambda x: "".join(re.split('^\d{2}|\W+', x)))
        d.SOLE_SUBTYPE = d.SOLE_SUBTYPE.apply(lambda x: np.nan if pd.isnull(x) else "".join(re.split('\(.*\)', x)))

        return d

    def reduce_dims(self, d):
        color_adjs = ['Medium', 'Bright', 'Lt.Pastel', 'Light', 'Dark', 'Lt/Pastel']
        d.COLOR_INTERN = d.COLOR_INTERN.apply(lambda x: remove_words_from_list(x, color_adjs))
        return d


class Items(Data):
    def __init__(self, data):
        Data.__init__(self, data)
        self.item_fields = ['PRODUCT_GROUP', 'CONCEPT', 'LINE',
                            'PRODUCT_GENDER', 'PRODUCT_CATEGORY', 'PRODUCT_TYPE',
                            'LACES', 'ZIPPER', 'SOLE_TYPE', 'SOLE_SUBTYPE', 'REMOVABLE_SOLE', 'SEASON_DESC',
                            'COLOR_INTERN']
        self.items_attributes = self.data_out[['PRODUCT_ID'] + self.item_fields].drop_duplicates().dropna(
            subset=['PRODUCT_ID'])
        self.item_df = self.onehot()

    def onehot(self):
        item_bools = ['LACES', 'ZIPPER', 'REMOVABLE_SOLE']
        item_categoricals = ['PRODUCT_GROUP', 'CONCEPT', 'PRODUCT_GENDER',
                             'PRODUCT_CATEGORY', 'PRODUCT_TYPE', 'SOLE_TYPE',
                             'SOLE_SUBTYPE', 'SEASON_DESC', 'COLOR_INTERN'
                             ]

        onehot_categoricals = pd.get_dummies(self.items_attributes[item_categoricals])
        return pd.concat([self.items_attributes.PRODUCT_ID, self.items_attributes[item_bools] * 1, onehot_categoricals],
                         axis=1)


class Users(Items):
    def __init__(self, data):
        Items.__init__(self, data)
        self.customer_product_lookup = self.data_out[['CUSTOMER_ID', 'PRODUCT_ID']].drop_duplicates().dropna(
            subset=['CUSTOMER_ID'])
        self.user_df = self.customer_product_lookup.merge(self.item_df, on='PRODUCT_ID').groupby(
            'CUSTOMER_ID').max().drop(columns=['PRODUCT_ID'])


# generates key : field values for most commonly appearing value for field
def country_dict(df, id_field, desc_field):
    return {x[id_field]: x[desc_field] for x in
            df.groupby(id_field)[desc_field].apply(lambda x: x.mode()).to_frame().reset_index().to_dict(
                'index').values()}


def remove_words_from_list(str, sublist):
    for word in sublist:
        return str.replace(word, '')


if __name__ == '__main__':
    infile = pd.read_csv(os.getcwd() + '/../data/Consumidor_Venta_Producto_UPC_Recom_2018_2020.csv')

    testdata = infile  # .head(200)

    cleandata = Data(testdata)

    print('Item Matrix...')
    items = Items(testdata)
    items.to_csv(os.getcwd() + "/data/item_m.csv", index=False)

    print('User Matrix...')
    users = Users(testdata)
    users.to_csv(os.getcwd() + "/data/user_m.csv")
