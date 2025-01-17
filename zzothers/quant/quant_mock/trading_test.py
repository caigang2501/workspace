import random,datetime


def predict(data):

    return random.random()*2-1

def mock_trading(stock_data,double_direction=False):
    base_money = 100
    data_len = 20
    i = data_len
    while i<len(stock_data)-1:
        prediction = predict(stock_data[i-data_len:i])
        if not double_direction:
            if prediction>0:
                base_money *= stock_data[i]/stock_data[i-1]
        else:
            if prediction>0:
                base_money *= stock_data[i]/stock_data[i-1]
            else:
                base_money *= max((2*stock_data[i-1]-stock_data[i]),0)/stock_data[i-1]
        i += 1
    return base_money

def trading_score(stock_data,double_direction=False):
    base_score = 100
    final_score = 0
    data_len = 20
    i = data_len
    while i<len(stock_data)-1:
        prediction = predict(stock_data[i-data_len:i])
        if not double_direction:
            if prediction>0:
                final_score += base_score*(stock_data[i]-stock_data[i-1])/stock_data[i-1]
        else:
            final_score += base_score*(stock_data[i-1]-stock_data[i])/stock_data[i-1]
        i += 1
    return final_score

if __name__=='__mian__':
    stock_data = []
    result = mock_trading(stock_data)
    print(result)














