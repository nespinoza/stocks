import os
import csv
from datetime import datetime as dt
import time

import numpy as np
import matplotlib.pyplot as plt 

import yfinance as yf

# Function from https://stackoverflow.com/questions/6451655/how-to-convert-python-datetime-dates-to-decimal-float-years:
def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

# Function that, given input time-series, takes X values befor a target date and uses it to predict Y values in the future using the mean of the X values:
def predict_with_mean(time, values, target_date, X, Y, plot = False):

    # First, figure out where the closest value to the target_date is:
    difference = np.abs(time - target_date)
    minimum_of_difference = np.min(difference)
    index = np.where(minimum_of_difference == difference)[0][0]

    # All right, now we have the index of the target-date. Use X values in the past (including target date) to calculate the mean:
    mean = np.mean( values[ index - X + 1: index + 1 ] )

    if plot:

        plt.plot(time[index], values[index], 'go', ms = 10)
        plt.plot(time[ index - X + 1: index + 1], values[ index - X + 1: index + 1],'ro')
        plt.plot(time[index + 1: index + Y + 1], values[index + 1: index + Y + 1],'bo')

    # And return the Y values from the future, which are those same means. To avoid confusion, return the time-stamps too:
    return time[index + 1: index + Y + 1], np.ones( Y ) * mean

# Same as predict_with_mean, but with a line instead of the mean:
def predict_with_line(time, values, target_date, X, Y, plot = False):

    # First, figure out where the closest value to the target_date is:
    difference = np.abs(time - target_date)
    minimum_of_difference = np.min(difference)
    index = np.where(minimum_of_difference == difference)[0][0]

    # All right, now we have the index of the target-date. Use X values in the past (including target date) to calculate the line 
    # predicting future values:
    coeffs = np.polyfit(time[ index - X + 1: index + 1 ], values[ index - X + 1: index + 1 ], 1)
    line = np.polyval( coeffs, time[index + 1: index + Y + 1] ) 

    if plot:

        plt.plot(time[index], values[index], 'go', ms = 10)
        plt.plot(time[ index - X + 1: index + 1], values[ index - X + 1: index + 1],'ro')
        plt.plot(time[index + 1: index + Y + 1], values[index + 1: index + Y + 1],'bo')

    # And return the Y values from the future, which are those same means. To avoid confusion, return the time-stamps too:
    return time[index + 1: index + Y + 1], line

# Check if data has been downloaded. If not, download, save, transform:
if not os.path.exists('stock_value.txt'):

    df = yf.download(tickers='META', period='120mo', interval='1d')

    # Transform dates to year fractions (accurate to within a year):
    times = np.zeros( len(df.index) )
    for i in range(len(df.index)):

        times[i] = toYearFraction( df.index[i] )

    # Extract value at closing time:
    stock_value = df['Close'].values

    # Save:
    with open('stock_value.txt', 'w') as f:

        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(times,stock_value))

else:

    times, stock_value = np.loadtxt('stock_value.txt', unpack = True)

# Use function that uses the mean of X days prior to predict stock Y 
# days in the future using target index 1000 (around the middle of the time series, to test):
index = 1000
X = 7
Y = 14

output_times, prediction_mean = predict_with_mean(times, stock_value, times[index], X, Y, plot = True)
output_times, prediction_line = predict_with_line(times, stock_value, times[index], X, Y, plot = True)

# Print errors for each:
error_mean = np.mean( np.abs( stock_value[index + 1: index + Y + 1] - prediction_mean ) )
error_line = np.mean( np.abs( stock_value[index + 1: index + Y + 1] - prediction_line ) )

print('Error on the mean: ', error_mean)
print('Error on line: ', error_line)

# Plot prediction:
plt.plot(times[index], stock_value[index], 'go', ms = 10)
plt.plot(times[ index - X + 1: index + 1], stock_value[ index - X + 1: index + 1],'ro')
plt.plot(times[index + 1: index + Y + 1], stock_value[index + 1: index + Y + 1],'bo')


plt.plot(times, stock_value, label = 'Value')
plt.plot(output_times, prediction_mean, label = 'Prediction with mean')
plt.plot(output_times, prediction_line, label = 'Prediction with line')
plt.xlabel('Fractional year', fontsize = 18) 
plt.ylabel('Dollar value of META', fontsize = 18) 
plt.xticks(fontsize = 16) 
plt.yticks(fontsize = 16) 
plt.legend()
plt.show()
