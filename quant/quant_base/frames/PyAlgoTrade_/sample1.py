
from pyalgotrade import strategy
from pyalgotrade.barfeed import quandlfeed
from pyalgotrade.barfeed import pandasfeed
from pyalgotrade.bar import Frequency
from pyalgotrade.technical import ma
import quandl

# Set your Quandl API key
quandl.ApiConfig.api_key = 'JA9JqQJmTe75z8Ev1zPK'

class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, short_window, long_window):
        super(MyStrategy, self).__init__(feed)
        self.__instrument = instrument
        self.__short_window = short_window
        self.__long_window = long_window
        self.__sma_short = ma.SMA(feed[instrument].getPriceDataSeries(), short_window)
        self.__sma_long = ma.SMA(feed[instrument].getPriceDataSeries(), long_window)

    def onBars(self, bars):
        if self.__sma_short[-1] > self.__sma_long[-1] and self.__sma_short[-2] <= self.__sma_long[-2]:
            # If the short-term moving average crosses above the long-term moving average, execute a buy order.
            self.enterLong(self.__instrument, 10)
        elif self.__sma_short[-1] < self.__sma_long[-1] and self.__sma_short[-2] >= self.__sma_long[-2]:
            # If the short-term moving average crosses below the long-term moving average, execute a sell order.
            self.enterShort(self.__instrument, 10)

# Download data from Quandl
data = quandl.get("WIKI/AAPL", start_date="2022-01-01", end_date="2023-01-01")

# Create a Quandl feed
feed = pandasfeed.Feed(Frequency.DAY)
feed.addBarsFromDataFrame("AAPL", data)

# Run the strategy
strategy = MyStrategy(feed, "AAPL", short_window=40, long_window=100)
strategy.run()

# Access the results if needed
print(f"Final portfolio value: {strategy.getResult()}")
