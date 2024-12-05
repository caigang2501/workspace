import tushare as ts
import pandas as pd

pro = ts.pro_api('dbaf7a76b4e4076a6bac684178e8db659ae659c79d65afa092f52cd1')

# data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
# data.to_csv(f'stock_list.csv',index=False)
df = pro.daily(ts_code='000001.SZ', start_date='20180101', end_date='20180118')  # 一次多个 ts_code='000001.SZ,600000.SH' 月：monthly
print(df)  # pre_close	float	昨收价【除权价，前复权】



