import numpy as np
import copy
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da
import datetime as dt
import matplotlib.pyplot as plt
import QSTK.qstkstudy.EventProfiler as ep
import pandas
import math




def find_bollinger_values(adjcloses, periods):
    means = pandas.rolling_mean(adjcloses, periods, min_periods=periods)
    stds = pandas.rolling_std(adjcloses, periods, min_periods=periods)
    bands = (adjcloses - means) / stds
    return bands

def find_events(ls_symbols, d_data, outputfile):
    ''' Finding the event dataframe '''
    df_close = d_data['close']
    ts_market = df_close['SPY']
    market_bands = find_bollinger_values(ts_market, 20)
    print "-----------------SPY"    
    print market_bands
    print "Finding Events"

    # Creating an empty dataframe
    df_events = copy.deepcopy(df_close)
    df_events = df_events * np.NAN

    # Time stamps for the event range
    ldt_timestamps = df_close.index
    f = open(outputfile, "w");

    for s_sym in ls_symbols:
        if s_sym == "SPY":
            continue
        sym_bands = find_bollinger_values(df_close[s_sym], 20)
        print "-----------------" + s_sym
        print sym_bands
        for i in range(1, len(ldt_timestamps)):
            # Calculating the returns for this timestamp
            f_symband_today = sym_bands.ix[ldt_timestamps[i]]
            f_symband_yest = sym_bands.ix[ldt_timestamps[i - 1]]
            f_marketband_today = market_bands.ix[ldt_timestamps[i]]

            # Event is found if the symbol is down more then 3% while the
            # market is up more then 2%
            #if f_symreturn_today <= -0.03 and f_marketreturn_today >= 0.02:
            if f_symband_today < -2.0 and f_symband_yest >= -2.0 and f_marketband_today >= 1.1:
                f.write(ldt_timestamps[i].__format__('%Y,%m,%d,'))
                f.write(s_sym + ',BUY,100,\n')
                if (i+ 5)>=len(ldt_timestamps):
                    f.write(ldt_timestamps[len(ldt_timestamps)-1].__format__('%Y,%m,%d,'))
                else :
                    f.write(ldt_timestamps[i+ 5].__format__('%Y,%m,%d,'))
                f.write(s_sym + ',SELL,100,\n')
                
#                df_events[s_sym].ix[ldt_timestamps[i]] = 1

    return df_events

def simulate(initial_cash, orders_file, values_file): 
    trades = np.loadtxt(orders_file, dtype='i4,i2,i2,S4,S4,float',
                        delimiter=',', comments="#", skiprows=0)
    trades = sorted(trades, key=lambda x: x[0]*10000+ x[1]*100 + x[2])
    dt_start = dt.datetime(trades[0][0], trades[0][1], trades[0][2])
    dt_end = dt.datetime(trades[-1][0], trades[-1][1], trades[-1][2]+1)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))
    ls_symbols = list(np.unique([x[3] for x in trades]))
    dataobj = da.DataAccess('Yahoo')
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ['close'])
    ls_keys = ['close']
    d_data = dict(zip(ls_keys, ldf_data))
    for s_key in ls_keys:
            d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
            d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
            d_data[s_key] = d_data[s_key].fillna(1.0)
    stock_prices = d_data['close'].values
    ownership = np.zeros((len(ldt_timestamps), len(ls_symbols)+1 ))
    cash = np.ones(len(ldt_timestamps)) * initial_cash
    for trade in trades:
        dt_trade = dt.datetime(trade[0], trade[1], trade[2], 16)
        price_index = ldt_timestamps.index(dt_trade)
        symbol_index = ls_symbols.index(trade[3])
        if trade[4]=='BUY':
            ownership[price_index:,symbol_index] += trade[5]
            cash[price_index:] -= trade[5] * stock_prices[price_index, symbol_index]  
        else:
            ownership[price_index:,symbol_index] -= trade[5] 
            cash[price_index:] += trade[5] * stock_prices[price_index, symbol_index]  
                
    ownership[:,-1] = np.sum(ownership[:, 0:-1] * stock_prices[:, :], 1)
    result = [(t.year, t.month, t.day, ownership[ldt_timestamps.index(t),-1] + cash[ldt_timestamps.index(t)]) for t in ldt_timestamps]
    print result[-1]
    np.savetxt(values_file, result, fmt='%d,%d,%d,%.2f', delimiter=',');


def calculate(values):
    start_value = values[0]
    cum_return = values / start_value
    portfolio_values = np.array(values)
    daily_returns = portfolio_values.copy()
    tsu.returnize0(daily_returns)
    average_daily_return = daily_returns[1:].mean()
    daily_return_stddev = daily_returns[1:].std()
    sharpe_ratio = math.sqrt(252) * average_daily_return / daily_return_stddev
    return daily_return_stddev, average_daily_return, sharpe_ratio, cum_return, portfolio_values

def analyze(values_file, benchmark_symbol, diagram):
    values = np.loadtxt(values_file, dtype='i4,i2,i2,float',
                        delimiter=',', comments="#", skiprows=0)
    values = sorted(values, key=lambda x: x[0]*10000+ x[1]*100 + x[2])
    dt_start = dt.datetime(values[0][0], values[0][1], values[0][2])
    dt_end = dt.datetime(values[-1][0], values[-1][1], values[-1][2]+1)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))
    ls_symbols = [benchmark_symbol]
    dataobj = da.DataAccess('Yahoo')
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ['close'])
    ls_keys = ['close']
    d_data = dict(zip(ls_keys, ldf_data))
    for s_key in ls_keys:
            d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
            d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
            d_data[s_key] = d_data[s_key].fillna(1.0)
    
        
            
    stock_prices = d_data['close'].values
    stock_value = stock_prices
#    tsu.returnize0(stock_prices)
    daily_return_stddev_spx, average_daily_return_spx, sharpe_ratio_spx, cum_return_spx, portfolio_values_spx = calculate(stock_value)
    daily_return_stddev, average_daily_return, sharpe_ratio, cum_return, portfolio_values = calculate([x[3] for x in values])
    #Sharpe ratio (Always assume you have 252 trading days in an year. And risk free rate = 0) of the total portfolio
    print 'Sharp Ratio Portfolio: ' + sharpe_ratio.__str__()
    print 'Sharp Ratio $SPX: ' + sharpe_ratio_spx.__str__()
    #Cumulative return of the total portfolio
    print 'Cumulative return of portfolio: ' + cum_return[-1].__str__()
    print 'Cumulative return of $SPX: ' + cum_return_spx[-1].__str__()
    #Standard deviation of daily returns of the total portfolio
    print 'Daily Return Std dev Portfolio: ' + daily_return_stddev.__str__()
    print 'Daily Return Std dev $SPX: ' + daily_return_stddev_spx.__str__()
    #Average daily return of the total portfolio
    print 'Average Daily Return Portfolio: ' + average_daily_return.__str__()
    print 'Average Daily Return $SPX: ' + average_daily_return_spx.__str__()

    # Plotting the results
    plt.clf()
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(ldt_timestamps, stock_value*(portfolio_values[0]/stock_value[0]))
    plt.plot(ldt_timestamps, portfolio_values, alpha=0.4)
    ls_names = ls_symbols
    ls_names.append('Portfolio')
    plt.legend(ls_names)
    plt.ylabel('Value')
    plt.xlabel('Date')
    fig.autofmt_xdate(rotation=45)
    plt.savefig(diagram, format='pdf')

#
# Prepare to read the data
#
if __name__ == '__main__':
    symbols = ["AAPL", "GOOG", "IBM", "MSFT"]
    dt_start = dt.datetime(2008, 1, 1)
    dt_end = dt.datetime(2009, 12, 31)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))

    dataobj = da.DataAccess('Yahoo')
    ls_symbols = dataobj.get_symbols_from_list('sp5002012')
    ls_symbols.append('SPY')
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    for s_key in ls_keys:
            d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
            d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
            d_data[s_key] = d_data[s_key].fillna(1.0)
    df_events = find_events(ls_symbols, d_data, 'h7_orders1.csv')
    print 'Found events'
    
    simulate(100000, 'h7_orders1.csv', 'h7_orders_values1.csv')
    
    analyze('h7_orders_values1.csv', '$SPX', 'homework7-1.pdf')
    # Plot the prices
