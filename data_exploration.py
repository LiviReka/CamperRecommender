import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./data/Consumidor_Venta_Producto_UPC_Recom_2018_2020.csv')

print(f'Number of Observations: {len(data)}')
print(f'Number of Features: {len(data.columns)}')
print(f'Features: {data.columns}')

print(data.dtypes)

feature_dict = {'FACTURA_ID': 'INVOICE_ID', 'FACTURA_POSICION_ID': 'INVOICE_POSITION_ID', 'CUSTOMER_ID': 'CUSTOMER_ID',
                'FACTURA_CLASE_DOCUMENTO_ID': 'INVOICE_DOCUMENT_CLASS_ID', 'ANO_MES_FACTURA': 'INVOICE_MONTH_YEAR',
                'ANO_FACTURA': 'ORDER_YEAR', 'MES_FACTURA': 'ORDER_MONTH', 'FECHA_FACTURA': 'INVOICE_DATE',
                'IMP_VENTA_NETO_EUR': 'NET_SALES_EUR', 'CANAL_VENTA_ID': 'SALES_CHANNEL_ID',
                'CANAL_VENTA_DESC': 'SALES_CHANNEL_DESC', 'TEMPORADA_COMERCIAL_ID': 'SEASON_ID',
                'TEMPORADA_COMERCIAL_DESC': 'SEASON_DESC', 'PRODUCTO_ID': 'PRODUCT_ID', 'TALLA': 'SIZE',
                'MATERIAL_ID': 'MATERIAL_ID', 'NUMERO_DEUDOR_PAIS_ID': 'DEBTOR_COUNTRY_ID',
                'NUMERO_DEUDOR_PAIS_DESC': 'DEBTOR_COUNTRY_ID', 'VENTA_DEVOLUCION': 'SALES_FILTER',
                'JERARQUIA_PROD_ID': 'PRODUCT_HIERARCHY_ID', 'GRUPO_ARTICULO_PRODUCTO_ID': 'PRODUCT_GROUP_ID',
                'GRUPO_ARTICULO': 'PRODUCT_GROUP', 'CONCEPTO': 'CONCEPT', 'LINEA': 'LINE',
                'GENERO_PRODUCTO': 'PRODUCT_GENDER', 'CATEGORIA': 'PRODUCT_CATEGORY', 'TIPOLOGIA': 'PRODUCT_TYPE',
                'COLOR': 'COLOR_INTERN', 'CONSUMER_COLOR': 'COLOR_EXTERN', 'CREMALLERA': 'ZIPPER', 'CORDONES': 'LACES',
                'OUTSOLE_SUELA_TIPO': 'SOLE_TYPE', 'OUTSOLE_SUELA_SUBTIPO': 'SOLE_SUBTYPE',
                'PLANTILLA_EXTRAIBLE': 'REMOVABLE_SOLE', 'CONTACTO_SN': 'CONTACT', 'EDAD_SN': 'AGE_AVAILABLE',
                'GENERO_CONTACTO': 'GENDER_AVAILABLE', 'EDAD_COMPRA': 'AGE_AT_PURCHASE',
                'EDAD_RANGO_COMPRA': 'AGE_RANGE',
                'PAIS_CONTACTO': 'COUNTRY_CONTACT_ID', 'PAIS_CONTACTO_DESC': 'COUNTRY_CONTACT_DESC',
                'CIUDAD_CONTACTO': 'CITY_CONTACT', 'IDIOMA_CONTACTO': 'LANGUAGE_CONTACT'}


def preprocess(d):
    data = d.rename(columns=feature_dict)
    data['REMOVABLE_SOLE'] = data['REMOVABLE_SOLE'].apply(lambda x: True if x == 'Extraible' else False)
    return data

data = preprocess(data)