import pandas as pd
import numpy as np
import re
import json
import os


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
        self.eng_data = self.preprocess(self.make_bools(self.to_english()))
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


class OneHotData(Data):
    def __init__(self, data):
        Data.__init__(self, data)
        self.item_fields = ['PRODUCT_GROUP', 'CONCEPT',
                            'PRODUCT_GENDER', 'PRODUCT_CATEGORY', 'PRODUCT_TYPE',
                            'LACES', 'ZIPPER', 'SOLE_TYPE', 'SOLE_SUBTYPE', 'REMOVABLE_SOLE',
                            'COLOR_INTERN']
        self.items_attributes = self.eng_data[['PRODUCT_ID'] + self.item_fields].dropna(
            subset=['PRODUCT_ID']).drop_duplicates()
        self.item_df = self.onehot()
        self.customer_product_lookup = self.eng_data[['CUSTOMER_ID', 'PRODUCT_ID']].dropna(
            subset=['CUSTOMER_ID']).drop_duplicates()
        self.user_df = self.customer_product_lookup.merge(self.item_df, on='PRODUCT_ID').groupby(
            'CUSTOMER_ID').max().drop(columns=['PRODUCT_ID'])

    def onehot(self):
        item_bools = ['LACES', 'ZIPPER', 'REMOVABLE_SOLE']
        item_categoricals = ['PRODUCT_GROUP', 'CONCEPT', 'PRODUCT_GENDER',
                             'PRODUCT_CATEGORY', 'PRODUCT_TYPE', 'SOLE_TYPE',
                             'SOLE_SUBTYPE', 'COLOR_INTERN'
                             ]

        encoded = pd.get_dummies(data=self.items_attributes, columns=item_categoricals)
        encoded[item_bools] = encoded[item_bools] * 1
        return encoded


# generates key : field values for most commonly appearing value for field
def country_dict(df, id_field, desc_field):
    return {x[id_field]: x[desc_field] for x in
            df.groupby(id_field)[desc_field].apply(lambda x: x.mode()).to_frame().reset_index().to_dict(
                'index').values()}


def remove_words_from_list(str, sublist):
    for word in sublist:
        return str.replace(word, '')


def user_invoice_item_dict(clean_df, ):
    print('Creating Lookup Tables...')
    customer_invoice_lookup = clean_df[['CUSTOMER_ID', 'INVOICE_ID']].drop_duplicates().dropna(
        subset=['CUSTOMER_ID'])
    invoice_product_lookup = clean_df[['INVOICE_ID', 'PRODUCT_ID']].drop_duplicates().dropna(
        subset=['INVOICE_ID'])

    print('Grouping Data...')
    invoice_by_customer = customer_invoice_lookup.groupby('CUSTOMER_ID')
    product_by_invoice = invoice_product_lookup.groupby('INVOICE_ID')

    print('Creating Index Dicts...')
    invoice_by_customer_dict = {customer: invoices['INVOICE_ID'].values.tolist() for customer, invoices in
                                invoice_by_customer}
    product_by_invoice_dict = {invoice: products['PRODUCT_ID'].values.tolist() for invoice, products in
                               product_by_invoice}
    return invoice_by_customer_dict, product_by_invoice_dict


if __name__ == '__main__':
    infile = pd.read_csv(os.getcwd() + '/../data/Consumidor_Venta_Producto_UPC_Recom_2018_2020.csv')

    testdata = infile#.head(500000)

    cleandata = OneHotData(testdata)

    print('Item Matrix...')
    items = cleandata.item_df
    items.to_csv(os.getcwd() + "/../data/item_m.csv", index=False)

    print('User Matrix...')
    users = cleandata.user_df
    users.to_csv(os.getcwd() + "/../data/user_m.csv")

    print('User Invoice Item Dict ...')
    invoice_by_customer_dict, product_by_invoice_dict = user_invoice_item_dict(cleandata.eng_data)
    with open(os.getcwd() + "/../data/invoice_by_customer_dict.json", "w") as outfile:
        json.dump(invoice_by_customer_dict, outfile)
    with open(os.getcwd() + "/../data/product_by_invoice_dict.json", "w") as outfile:
        json.dump(product_by_invoice_dict, outfile)
