import requests
import tushare as ts
import akshare as ak
import yfinance as yf
import pandas as pd
from datetime import datetime



def data_ts1():
    pro = ts.pro_api('dbaf7a76b4e4076a6bac684178e8db659ae659c79d65afa092f52cd1')
    # data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    # data.to_csv(f'stock_list.csv',index=False)
    df = pro.daily(ts_code='000001.SZ', start_date='20241121', end_date=str(datetime.now().date()).replace('-',''))  # 一次多个 ts_code='000001.SZ,600000.SH' 月：monthly
    print(df.columns)
    return df

def data_ts():
    pro = ts.pro_api('dbaf7a76b4e4076a6bac684178e8db659ae659c79d65afa092f52cd1')
    df = pro.index_basic(market='SW')
    return df


def data_yahoo():
    stock = yf.Ticker("000002.SZ")  # 示例：苹果公司
    hist = stock.history(period="5d")  # ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    print(hist.columns,hist.iloc[0,0],type(hist.iloc[0,0]),type(datetime.now().date()))
    return hist

# def data_quantl():
#     my_api_key = 'tMv2k5ytCEegPn6iGF9X'
#     data = quandl.get("CHINA_STOCKS/600519")  # 例如贵州茅台
#     return data


from jqdatasdk import auth, get_price,get_query_count
def data_jukuan():
    auth("18980526371", "NFD2=Whk$VK!EEf")
    print(get_query_count())
    data = get_price(
        "000001.SZ", start_date="2024-10-01", end_date="2024-11-01", frequency="daily"
    )

    return data

if __name__=='__main__':
    df = data_ts()
    print(df)


