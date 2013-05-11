import math
import numpy as np
import datetime as dt
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu



def simulate (start_date, end_date, symbol_list, allocation):
    initial_investment = 1000000
    market_close_time = dt.timedelta(hours=16)
    # Get a list of trading days between the start and the end.
    trade_day_list = du.getNYSEdays(start_date, end_date, market_close_time)

    # Creating an object of the dataaccess class with Yahoo as the source.
    data_access_service = da.DataAccess('Yahoo')

    # Keys to be read from the data, it is good to read everything in one go.
    column_list = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    trade_data = data_access_service.get_data(trade_day_list, symbol_list, column_list)
    trade_dictionary = dict(zip(column_list, trade_data))

    # Getting the numpy ndarray of close prices.
    price_list = trade_dictionary['close'].values
    init_allocation = np.array([item * initial_investment for item in allocation], np.float); 
    start_price = price_list[0, :]
    number_of_stocks = (init_allocation//start_price).astype(int)
    
    portfolio_daily_values = (number_of_stocks * price_list)[:, :].sum(1)
     
    portfolio_daily_returns = portfolio_daily_values.copy()

    # Calculate the daily returns of the prices. (Inplace calculation)
    # returnize0 works on ndarray and not dataframes.
    tsu.returnize0(portfolio_daily_returns) 
    
    average_daily_return = portfolio_daily_returns[:].mean()
    daily_return_stddev = portfolio_daily_returns[:].std()
    sharpe_ratio = math.sqrt(portfolio_daily_returns.size) * average_daily_return / daily_return_stddev;
    cum_return = portfolio_daily_returns[:].sum();  
    return (daily_return_stddev, average_daily_return, sharpe_ratio, cum_return)


def findOptimalAllocation(start_date, end_date, symbols):
    max_sharp_ratio = -10000000
    allocations = list(np.ndindex(11, 11, 11, 11))
    for allocationTuple in allocations:
        allocation = np.array(allocationTuple)
        allocation = allocation * 0.1
        if allocation.sum() == 1.0:
            print allocation
            daily_return_stddev, average_daily_return, sharpe_ratio, cum_return = simulate(start_date, end_date, symbols, allocation)
            if (sharpe_ratio > max_sharp_ratio):
                max_sharp_ratio = sharpe_ratio
                max_allocation = allocation
    
    daily_return_stddev, average_daily_return, sharpe_ratio, cum_return = simulate(start_date, end_date, symbols, max_allocation)
    return max_allocation, average_daily_return, daily_return_stddev, sharpe_ratio, cum_return

def main():
    print 'main is finished'
    start_date = dt.datetime(2011, 1, 4)
    end_date = dt.datetime(2011, 12, 30)
    symbols = ['GOOG','AAPL','GLD','SPY']
    max_allocation, average_daily_return, daily_return_stddev, sharpe_ratio, cum_return = findOptimalAllocation(start_date, end_date, symbols)
    print start_date
    print end_date
    print symbols
    print max_allocation
    
    print average_daily_return
    print daily_return_stddev
    print sharpe_ratio
    print cum_return

    start_date = dt.datetime(2010, 1, 4)
    end_date = dt.datetime(2010, 12, 30)
    symbols = ['GOOG','AAPL','GLD','SPY']
    max_allocation, average_daily_return, daily_return_stddev, sharpe_ratio, cum_return = findOptimalAllocation(start_date, end_date, symbols)
    print start_date
    print end_date
    print symbols
    print max_allocation
    
    print average_daily_return
    print daily_return_stddev
    print sharpe_ratio
    print cum_return
    
    

if __name__ == '__main__':
    main()