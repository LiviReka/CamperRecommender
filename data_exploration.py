import pandas as pd
import numpy as np
import re

data = pd.read_csv('./data/Consumidor_Venta_Producto_UPC_Recom_2018_2020.csv')

data = pd.read_csv('./data/Consumidor_Venta_Producto_UPC_Recom_2018_2020.csv')
english_cols = {'FACTURA_ID': 'INVOICE_ID', 'FACTURA_POSICION_ID': 'INVOICE_POSITION_ID', 'CUSTOMER_ID': 'CUSTOMER_ID',
                'FACTURA_CLASE_DOCUMENTO_ID': 'INVOICE_DOCUMENT_CLASS_ID', 'ANO_MES_FACTURA': 'INVOICE_MONTH_YEAR',
                'ANO_FACTURA': 'ORDER_YEAR', 'MES_FACTURA': 'ORDER_MONTH', 'FECHA_FACTURA': 'INVOICE_DATE',
                'IMP_VENTA_NETO_EUR': 'NET_SALES_EUR', 'CANAL_VENTA_ID': 'SALES_CHANNEL_ID',
                'CANAL_VENTA_DESC': 'SALES_CHANNEL_DESC', 'TEMPORADA_COMERCIAL_ID': 'SEASON_ID',
                'TEMPORADA_COMERCIAL_DESC': 'SEASON_DESC', 'PRODUCTO_ID': 'PRODUCT_ID', 'TALLA': 'SIZE',
                'MATERIAL_ID': 'MATERIAL_ID', 'NUMERO_DEUDOR_PAIS_ID': 'SALE_COUNTRY_ID',
                'NUMERO_DEUDOR_PAIS_DESC': 'SALE_COUNTRY_DESC', 'VENTA_DEVOLUCION': 'SALES_FILTER',
                'JERARQUIA_PROD_ID': 'PRODUCT_HIERARCHY_ID', 'GRUPO_ARTICULO_PRODUCTO_ID': 'PRODUCT_GROUP_ID',
                'GRUPO_ARTICULO': 'PRODUCT_GROUP', 'CONCEPTO': 'CONCEPT', 'LINEA': 'LINE',
                'GENERO_PRODUCTO': 'PRODUCT_GENDER', 'CATEGORIA': 'PRODUCT_CATEGORY', 'TIPOLOGIA': 'PRODUCT_TYPE',
                'COLOR': 'COLOR_INTERN', 'CONSUMER_COLOR': 'COLOR_EXTERN', 'CREMALLERA': 'ZIPPER', 'CORDONES': 'LACES',
                'OUTSOLE_SUELA_TIPO': 'SOLE_TYPE', 'OUTSOLE_SUELA_SUBTIPO': 'SOLE_SUBTYPE',
                'PLANTILLA_EXTRAIBLE': 'REMOVABLE_SOLE', 'CONTACTO_SN': 'PHYSICAL_CONTACT', 'EDAD_SN': 'AGE_AVAILABLE',
                'GENERO_CONTACTO': 'GENDER_AVAILABLE', 'EDAD_COMPRA': 'AGE_AT_PURCHASE',
                'EDAD_RANGO_COMPRA': 'AGE_RANGE',
                'PAIS_CONTACTO': 'COUNTRY_CONTACT_ID', 'PAIS_CONTACTO_DESC': 'COUNTRY_CONTACT_DESC',
                'CIUDAD_CONTACTO': 'CITY_CONTACT', 'IDIOMA_CONTACTO': 'LANGUAGE_CONTACT'}

dropcols = ['INVOICE_MONTH_YEAR', 'ORDER_YEAR', 'ORDER_MONTH', 'SALES_CHANNEL_ID', \
            'SALES_CHANNEL_DESC', 'AGE_AVAILABLE', 'MATERIAL_ID', 'COUNTRY_CONTACT_DESC']


# gets most common country for the code
def country_dict(df, id_field, desc_field):
    return {x[id_field]: x[desc_field] for x in
            df.groupby(id_field)[desc_field].apply(lambda x: x.mode()).to_frame().reset_index().to_dict(
                'index').values()}


def preprocess(d):
    d_copy = d.rename(columns=english_cols)
    d_copy['REMOVABLE_SOLE'] = d_copy['REMOVABLE_SOLE'].apply(lambda x: True if x == 'Extraible' else False)

    invoice_id_dict = {'ZTON': 'Sale', 'ZDVN': 'Return', 'ZDAN': 'Cancelled Return'}
    d_copy['INVOICE_DOCUMENT_CLASS_ID'] = d_copy['INVOICE_DOCUMENT_CLASS_ID'].apply(lambda x: invoice_id_dict[x])

    d_copy['INVOICE_DATE'] = pd.to_datetime(d_copy.INVOICE_DATE)

    d_copy.AGE_AT_PURCHASE = d_copy.AGE_AT_PURCHASE.replace(0, np.nan)

    d_copy.ZIPPER = d_copy.ZIPPER.apply(lambda x: True if x in ('SI', 'YES') else False)
    d_copy.LACES = d_copy.LACES.apply(lambda x: True if x in ('With laces', 'Con cordones') else False)

    product_group_dict = {'Zapatos Adulto': 'Adult Shoes',
                          'Bolsos': 'Bag',
                          'Bolsos Cartujano': 'Bag',
                          'Ropa': 'Clothings',
                          'Complementos': 'Complements'
                          }
    d_copy.PRODUCT_GROUP = d_copy.PRODUCT_GROUP.apply(lambda x: product_group_dict[x] if x in product_group_dict else x)

    color_dict = {'Мульти ассорти': 'Multi - Assorted',
                  'красный': 'Red',
                  'розовый': 'Pink',
                  'желтый': 'Yellow'
                  }
    d_copy.COLOR_INTERN = d_copy.COLOR_INTERN.apply(lambda x: color_dict[x] if x in color_dict else x)

    d_copy.PHYSICAL_CONTACT = d_copy.PHYSICAL_CONTACT.apply(lambda x: True if x == 'S' else False)

    d_copy.SEASON_DESC = d_copy.SEASON_DESC.apply(lambda x: "".join(re.split('^\d{2}|\W+', x)))

    d_copy.SOLE_SUBTYPE = d_copy.SOLE_SUBTYPE.apply(
        lambda x: np.nan if pd.isnull(x) else "".join(re.split('\(.*\)', x)))

    return d_copy


def get_item_mtx(df):
    item_fields = ['PRODUCT_GROUP', 'CONCEPT', 'LINE', \
                   'PRODUCT_GENDER', 'PRODUCT_CATEGORY', 'PRODUCT_TYPE', \
                   'LACES', 'ZIPPER', 'SOLE_TYPE', 'SOLE_SUBTYPE', 'REMOVABLE_SOLE', 'SEASON_DESC', \
                   'COLOR_INTERN']

    item_bools = ['LACES', 'ZIPPER', 'REMOVABLE_SOLE']

    item_categoricals = ['PRODUCT_GROUP', 'CONCEPT', 'LINE', 'PRODUCT_GENDER', \
                         'PRODUCT_CATEGORY', 'PRODUCT_TYPE', 'SOLE_TYPE', \
                         'SOLE_SUBTYPE', 'SEASON_DESC', 'COLOR_INTERN'
                         ]

    items_attributes = df[['PRODUCT_ID'] + item_fields].drop_duplicates().dropna(subset=['PRODUCT_ID'])

    onehot_categoricals = pd.get_dummies(items_attributes[item_categoricals])
    item_mtx_df = pd.concat([items_attributes.PRODUCT_ID, items_attributes[item_bools] * 1, onehot_categoricals],
                            axis=1)
    return item_mtx_df


def user_mtx_from_item(i_mtx, df):
    customer_product_lookup = df[['CUSTOMER_ID', 'PRODUCT_ID']].drop_duplicates().dropna(subset=['CUSTOMER_ID'])
    merged_customer_product = customer_product_lookup.merge(i_mtx, on='PRODUCT_ID')
    user_mtx_df = merged_customer_product.groupby('CUSTOMER_ID').max().drop(columns=['PRODUCT_ID'])
    return user_mtx_df


if __name__ == '__main__':
    cleandata = preprocess(data)

    country_lookup = country_dict(data, 'PAIS_CONTACTO', 'PAIS_CONTACTO_DESC')
    country_lookup.update(country_dict(data, 'NUMERO_DEUDOR_PAIS_ID', 'NUMERO_DEUDOR_PAIS_DESC'))

    # cleandata = cleandata.drop(columns=dropcols)

    print('Item Matrix Done')
    item_df = get_item_mtx(cleandata)
    # item_mtx = np.asmatrix(item_df)
    item_df.to_csv("item_m.csv")

    print('User Matrix Done')
    user_df = user_mtx_from_item(item_df, cleandata)
    # user_mtx = np.asmatrix(user_df)
    user_df.to_csv("user_m.csv")

    print('hello world')