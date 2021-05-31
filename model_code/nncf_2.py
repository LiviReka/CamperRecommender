
import pandas as pd

df_test = pd.read_csv('df_test.csv')
df_test.pivot(index="user_id", columns="item_id", values="ncf_predictions").values.shape
print('df test pivoted')

# collapse
nfc_pred_df = pd.DataFrame(df_test.pivot(index="user_id", columns="item_id", values="ncf_predictions").values)
print('nfc pred collapsed')


print(nfc_pred_df.shape)
compression_opts = dict(method='zip', archive_name='nncf.csv')
nfc_pred_df.to_csv('nncf.zip', index=False, compression=compression_opts)


