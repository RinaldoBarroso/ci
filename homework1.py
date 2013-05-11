import math
import numpy as np
import datetime as dt
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu



def simulate (data_access_service, start_date, end_date, symbol_list, allocation):
    market_close_time = dt.timedelta(hours=16)
    # Get a list of trading days between the start and the end.
    trade_day_list = du.getNYSEdays(start_date, end_date, market_close_time)



    # Keys to be read from the data, it is good to read everything in one go.
    column_list = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    trade_data = data_access_service.get_data(trade_day_list, symbol_list, column_list)
    trade_dictionary = dict(zip(column_list, trade_data))

    # Getting the numpy ndarray of close prices.
    original_price_list = trade_dictionary['close'].values
    price_list = original_price_list.copy()
    tsu.returnize0(price_list)
     
    portfolio_daily_returns = (allocation * price_list)[:, :].sum(1)

    average_daily_return = portfolio_daily_returns[:].mean()
    daily_return_stddev = portfolio_daily_returns[:].std()
    sharpe_ratio = math.sqrt(252) * average_daily_return / daily_return_stddev;
    cum_returns = original_price_list[-1, :]/original_price_list[0, :];
    portfolio_cum_return = (allocation * cum_returns)[:].sum();
  
    return (daily_return_stddev, average_daily_return, sharpe_ratio, portfolio_cum_return)

def findOptimalAllocation(data_access_service, start_date, end_date, symbols):
    max_sharp_ratio = -10000000
    allocations = list(np.ndindex(11, 11, 11, 11))
    for allocationTuple in allocations:
        allocation = np.array(allocationTuple)
        allocation = allocation * 0.1
        if allocation.sum() == 1.0:
            print allocation
            daily_return_stddev, average_daily_return, sharpe_ratio, cum_return = simulate(data_access_service, start_date, end_date, symbols, allocation)
            if (sharpe_ratio > max_sharp_ratio):
                max_sharp_ratio = sharpe_ratio
                max_allocation = allocation
    
    daily_return_stddev, average_daily_return, sharpe_ratio, cum_return = simulate(data_access_service, start_date, end_date, symbols, max_allocation)
    return max_allocation, average_daily_return, daily_return_stddev, sharpe_ratio, cum_return

def main():
    # Creating an object of the dataaccess class with Yahoo as the source.
    data_access_service = da.DataAccess('Yahoo')
    start_date = dt.datetime(2011, 1, 1)
    end_date = dt.datetime(2011, 12, 31)
    symbols = ['BRCM', 'ADBE', 'AMD', 'ADI']
    max_allocation, average_daily_return, daily_return_stddev, sharpe_ratio, cum_return = findOptimalAllocation(data_access_service, start_date, end_date, symbols)
    print '---------2011------'
    print start_date
    print end_date
    print symbols
    print max_allocation
    
    print 'average daily : ' + average_daily_return.__str__()
    print 'daily return stdev: ' + daily_return_stddev.__str__()
    print 'sharpe:' + sharpe_ratio.__str__()
    print 'cum return:' + cum_return.__str__()


#    print '---------2010------'
#    start_date = dt.datetime(2010, 1, 1)
#    end_date = dt.datetime(2010, 12, 31)
#    symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
#    max_allocation, average_daily_return, daily_return_stddev, sharpe_ratio, cum_return = findOptimalAllocation(data_access_service, start_date, end_date, symbols)
##    average_daily_return, daily_return_stddev, sharpe_ratio, cum_return = simulate(data_access_service, start_date, end_date, symbols, [0, 0, 0, 1])
#    print start_date
#    print end_date
#    print symbols
#    #print max_allocation
#    
#    print 'average daily : ' + average_daily_return.__str__()
#    print 'daily return stdev: ' + daily_return_stddev.__str__()
#    print 'sharpe:' + sharpe_ratio.__str__()
#    print 'cum return:' + cum_return.__str__()
    
    

if __name__ == '__main__':
    main()