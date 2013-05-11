'''
(c) 2011, 2012 Georgia Tech Research Corporation
This source code is released under the New BSD license.  Please see
http://wiki.quantsoftware.org/index.php?title=QSTK_License
for license details.

Created on January, 24, 2013

@author: Sourabh Bajaj
@contact: sourabhbajaj@gatech.edu
@summary: Example tutorial code.
'''

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

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
        if trade[4]=='Buy':
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

def main():
    ''' Main Function'''
    simulate(1000000, 'orders.csv', 'values.csv')
    
    analyze('values.csv', '$SPX', 'homework3-orders.pdf')

    simulate(1000000, 'orders2.csv', 'values2.csv')
    
    analyze('values2.csv', '$SPX', 'homework3-orders2.pdf')

    
    # Create two list for symbol names and allocation
#    ls_port_syms = []
#    lf_port_alloc = []
#    for port in na_portfolio:
#        ls_port_syms.append(port[0])
#        lf_port_alloc.append(port[1])
#
#    # Creating an object of the dataaccess class with Yahoo as the source.
#    c_dataobj = da.DataAccess('Yahoo')
#    ls_all_syms = c_dataobj.get_all_symbols()
#    # Bad symbols are symbols present in portfolio but not in all syms
#    ls_bad_syms = list(set(ls_port_syms) - set(ls_all_syms))
#
#    if len(ls_bad_syms) != 0:
#        print "Portfolio contains bad symbols : ", ls_bad_syms
#
#    for s_sym in ls_bad_syms:
#        i_index = ls_port_syms.index(s_sym)
#        ls_port_syms.pop(i_index)
#        lf_port_alloc.pop(i_index)
#
#    # Reading the historical data.
#    dt_end = dt.datetime(2011, 1, 1)
#    dt_start = dt_end - dt.timedelta(days=1095)  # Three years
#    # We need closing prices so the timestamp should be hours=16.
#    dt_timeofday = dt.timedelta(hours=16)
#
#    # Get a list of trading days between the start and the end.
#    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
#
#    # Keys to be read from the data, it is good to read everything in one go.
#    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
#
#    # Reading the data, now d_data is a dictionary with the keys above.
#    # Timestamps and symbols are the ones that were specified before.
#    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_port_syms, ls_keys)
#    d_data = dict(zip(ls_keys, ldf_data))
#
#    # Copying close price into separate dataframe to find rets
#    df_rets = d_data['close'].copy()
#    # Filling the data.
#    df_rets = df_rets.fillna(method='ffill')
#    df_rets = df_rets.fillna(method='bfill')
#
#    # Numpy matrix of filled data values
#    na_rets = df_rets.values
#    # returnize0 works on ndarray and not dataframes.
#    tsu.returnize0(na_rets)
#
#    # Estimate portfolio returns
#    na_portrets = np.sum(na_rets * lf_port_alloc, axis=1)
#    na_port_total = np.cumprod(na_portrets + 1)
#    na_component_total = np.cumprod(na_rets + 1, axis=0)
#
#    # Plotting the results
#    plt.clf()
#    fig = plt.figure()
#    fig.add_subplot(111)
#    plt.plot(ldt_timestamps, na_component_total, alpha=0.4)
#    plt.plot(ldt_timestamps, na_port_total)
#    ls_names = ls_port_syms
#    ls_names.append('Portfolio')
#    plt.legend(ls_names)
#    plt.ylabel('Cumulative Returns')
#    plt.xlabel('Date')
#    fig.autofmt_xdate(rotation=45)
#    plt.savefig('tutorial3.pdf', format='pdf')

if __name__ == '__main__':
    main()
