import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da
import datetime as dt
import matplotlib.pyplot as plt
import pandas
from pylab import *


def find_bollinger_values(symbols, startday, endday, periods):
    timeofday = dt.timedelta(hours=16)
    timestamps = du.getNYSEdays(startday, endday, timeofday)
    dataobj = da.DataAccess('Yahoo')
    voldata = dataobj.get_data(timestamps, symbols, "volume")
    adjcloses = dataobj.get_data(timestamps, symbols, "close")
    actualclose = dataobj.get_data(timestamps, symbols, "actual_close")
    
    adjcloses = adjcloses.fillna()
    adjcloses = adjcloses.fillna(method='backfill')
    
    means = pandas.rolling_mean(adjcloses, periods, min_periods=periods)
    stds = pandas.rolling_std(adjcloses, periods, min_periods=periods)

    bands = (adjcloses - means) / stds

    print bands
    
    plt.clf()
    
    symtoplot = 'GOOG'
    plot(adjcloses.index, adjcloses[symtoplot].values, label=symtoplot)
    plot(adjcloses.index, means[symtoplot].values)
    plt.legend([symtoplot, 'Mean'])
    plot(adjcloses.index, means[symtoplot].values + stds[symtoplot].values)
    plt.legend([symtoplot, 'Up - Band'])
    plot(adjcloses.index, means[symtoplot].values - stds[symtoplot].values)
    plt.legend([symtoplot, 'Low Band'])
    plt.ylabel('Adjusted Close')
    
    savefig("bands-ex.png", format='png')


#
# Prepare to read the data
#
if __name__ == '__main__':
    symbols = ["AAPL", "GOOG", "IBM", "MSFT"]
    startday = dt.datetime(2010, 1, 1)
    endday = dt.datetime(2010, 6, 24)
    
    find_bollinger_values(symbols, startday, endday, 20)
    # Plot the prices
