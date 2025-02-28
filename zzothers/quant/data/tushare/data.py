import requests
import tushare as ts
import akshare as ak
import yfinance as yf
import pandas as pd
from datetime import datetime

def data_ts1(ts_code:str,start_date,save=True):
    pro = ts.pro_api('dbaf7a76b4e4076a6bac684178e8db659ae659c79d65afa092f52cd1')
    # data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    # data.to_csv(f'stock_list.csv',index=False)
    # df_month = pro.monthly(ts_code='000001.SZ', start_date='20240121', end_date=str(datetime.now().date()).replace('-',''))  # 一次多个 ts_code='000001.SZ,600000.SH' 月：monthly
    df_day = pro.daily(ts_code=ts_code, start_date=start_date, end_date=str(datetime.now().date()).replace('-',''))  # 一次多个 ts_code='000001.SZ,600000.SH' 月：monthly
    if save:
        csv_name = ts_code.replace('.','')
        df_day.to_csv(f'data/quant/{csv_name}.csv')
    return df_day

def data_ts():
    pro = ts.pro_api('dbaf7a76b4e4076a6bac684178e8db659ae659c79d65afa092f52cd1')
    df = pro.index_basic(market='SW')
    return df


def data_yahoo():
    stock = yf.Ticker("000001.SZ")  # 示例：苹果公司
    hist = stock.history(period="5d")  # ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    print(hist.columns,hist.iloc[0,0],type(hist.iloc[0,0]),type(datetime.now().date()))
    return hist

def data_ak():
    # stock_changes_em_df = ak.stock_changes_em(symbol="大笔买入")
    stock_individual_info_em_df = ak.stock_individual_info_em(symbol="000001")
    print(stock_individual_info_em_df)

if __name__=='__main__':
    df = data_ak()
    # print(df)


