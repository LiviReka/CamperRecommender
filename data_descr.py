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

data = data.rename(columns=feature_dict)

### Basic Data Exploration

# number of invoices
plt.title('Total and unique number of Invoice IDs')
invoice_id_total = len(data.INVOICE_ID)
invoice_id_unique = len(data.INVOICE_ID.unique())
plt.bar(x=['total', 'unique'], height=[invoice_id_unique, invoice_id_total])
plt.show()

# invoice id frequency
plt.title('Invoice ID Frequencies (log scale)')
unique_invoice_count = data.INVOICE_ID.value_counts()
plt.hist(unique_invoice_count)
plt.yscale('log')
plt.show()

# invoice position ID check
print(f'Max Invoice ID Index: {max(data.INVOICE_POSITION_ID)}')
plt.hist(data.INVOICE_POSITION_ID)
plt.yscale('log')
plt.show()
# TODO:Invoice ID position should be in linde with the occurences of INVOICE IDs?!?!?

# number of total and unique customers
plt.title('Total and unique number of Customer IDs')
invoice_id_total = len(data.CUSTOMER_ID)
invoice_id_unique = len(data.CUSTOMER_ID.unique())
plt.bar(x=['total', 'unique'], height=[invoice_id_unique, invoice_id_total])
plt.show()

# customer id frequency
plt.title('Customer ID Frequencies (log scale)')
unique_invoice_count = data.CUSTOMER_ID.value_counts()
plt.hist(unique_invoice_count)
plt.yscale('log')
plt.show()

# Missing values
nan = data.isnull().sum()
nan_perc = nan/len(data)
print(nan_perc)

# invoice types
inv_types = data.INVOICE_DOCUMENT_CLASS_ID.value_counts()
plt.bar(x=['Sale', 'Return', 'Cancelled Return'], height=inv_types.values)
plt.show()

# product IDs
print(f'Number of unique product IDs: {len(data.PRODUCT_ID.unique())}')
plt.title('Product ID Purchase Frequencies')
unique_invoice_count = data.PRODUCT_ID.value_counts()
plt.hist(unique_invoice_count)
plt.yscale('log')
plt.show()

