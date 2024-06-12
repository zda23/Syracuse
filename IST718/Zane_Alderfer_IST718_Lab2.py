#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
import statsmodels.api as st
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from math import sqrt
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from scipy.stats import kstest
import missingno
from prophet import Prophet


# Retrieving Data and Beginning Analysis

# In[2]:


data = pd.read_csv("http://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfr_month.csv")
df = pd.DataFrame(data.values, columns=data.columns, index=data.index)


# In[3]:


print(data.describe(include='all'))
print(data.isnull().sum())
print(data.duplicated().values.any())
print(data.info())


# Step 2: Cleaning Data

# In[4]:


#checking dups
missingno.bar(df)
plt.show()


# In[5]:


#cleaning Nulls
df.dropna(subset=['Metro'], inplace=True)
df.dropna(subset=['City'], inplace=True)


# In[6]:


# Transforming categorical variables from object to factor and quantitative variables from object to numeric

cols_to_convert = df.columns[:9]
# Convert selected columns to category type
df[cols_to_convert] = df[cols_to_convert].astype('category')

# Transforming quantitative data into numeric format
cols_to_convert = df.columns[9:]
# Convert the selected columns to numeric
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')


# Verify the changes
print('Data Types:')
print(df.dtypes)
print()

df.describe()
print()
for column in df.select_dtypes(include='number').columns:
    df[column].fillna(df[column].median(), inplace=True)
    
# Verify changes
print('Number of Null Values:')
print(df.isnull().sum().sum())


# In[35]:


#dropping unneccesary columns
#df.drop(columns=['SizeRank','RegionName', 'RegionType', 'StateName'], inplace=True)
melted = pd.melt(df, id_vars=['RegionID', 'State', 'City', 'Metro','CountyName'], 
                     var_name='Date')
melted['Date'] = pd.to_datetime(melted['Date'], infer_datetime_format=True)
melted = melted.dropna(subset=['value'])
melted


# In[36]:


#creating df that distinguishes averages for dates
date_ave = melted.groupby('Date')['value'].mean()
date_ave = date_ave.reset_index()
date_ave.columns = ['Date', 'ave_value']
date_ave = date_ave[date_ave['Date']>='2008-12-01']
print(date_ave.head())


# In[37]:


#plotted data
plt.figure(figsize=(12,9))
date_ave.plot(x='Date', y='ave_value')
_ = plt.xlabel('Date')
_ = plt.ylabel('Avg Value')
_ = plt.title('Avg Value by Date')
# Rotate x-labels
plt.xticks(rotation=45)
plt.show()


# In[38]:


#same thing but with all dates
#creating df that distinguishes averages for dates
date_ave_all = melted.groupby('Date')['value'].mean()
date_ave_all = date_ave_all.reset_index()
date_ave_all.columns = ['Date', 'ave_value']
date_ave_all = date_ave_all[date_ave_all['Date']>='1996-01-01']

plt.figure(figsize=(20,12))
date_ave_all.plot(x='Date', y='ave_value')
_ = plt.xlabel('Date')
_ = plt.ylabel('Avg Value')
_ = plt.title('Avg Value by Date')
# Rotate x-labels
plt.xticks(rotation=45)
plt.show()


# In[39]:


#showing the average cost per home in each state
plt.figure(figsize=(18,10))
sns.set(style='whitegrid')
sns.countplot(x='State', data=melted)

plt.xticks(rotation=45)
plt.show()  # Show the plot


# Based on this data, we can see that New York, California, Pennsylvania, Texas, and Illinois seem to be on average the most expensive states to live in.

# In[40]:


# Create a new data frame looking at values at the state level
state = melted.groupby(['Date', 'State'])['value'].mean()
state = state.reset_index()
state.columns = ['Date', 'state', 'ave_value']
state = state[state['Date']>='1996-01-01']
print(state.head())


# In[41]:


# developing a line plot
state_df = state.pivot(index='Date', columns='state', values='ave_value')

# Plot the data
state_df.plot(figsize=(15,12))
_ = plt.xlabel('Year')
_ = plt.ylabel('Average Value')
_ = plt.title('Average Value by Year')
plt.legend(loc='center right', bbox_to_anchor=[1.1, 0.5])
plt.show()


# In[42]:


#separating Arkansas data
arkansas = melted[melted['State'] == 'AR']
arkansas.head()


# In[43]:


#separating Arkansas specific data
state_name = 'AR'

# Define the metro areas of interest
metros = ['Hot Springs, AR', 'Little Rock-North Little Rock-Conway, AR', 'Fayetteville-Springdale-Rogers, AR', 'Searcy, AR']

# Select data for the specified state and metro areas
state_data = melted[(melted['State'] == state_name) & (melted['Metro'].isin(metros))]

# Convert 'Date' column to datetime type
state_data['Date'] = pd.to_datetime(state_data['Date'])

# Group data by metro area and year, and calculate the yearly average home value
yearly_avg_values = state_data.groupby([state_data['Date'].dt.year, 'Metro'])['value'].mean()

# Unstack the multi-index to pivot the data and create separate columns for each metro area
yearly_avg_values = yearly_avg_values.unstack()

# Plot the yearly average home value for each metro area
plt.figure(figsize=(12, 6))
for metro in metros:
    yearly_avg_values[metro].plot(marker='o', linestyle='-', label=metro)

plt.title('Arkansas Home Value data from 96-23')
plt.ylabel('Yearly Average home value')
plt.xlabel('Year')
plt.legend(title='Metro')
plt.grid(True)
plt.show()


# In[44]:


arkansas_average = state_data.groupby('Date')['value'].mean()
arkansas_average.plot(x='date', y='value')
_ = plt.xlabel('Year')
_ = plt.ylabel('Avg Value')
_ = plt.title('Avg Value per Year')
plt.show()


# In[45]:


# Convert the 'Date' column to datetime type
melted['Date'] = pd.to_datetime(melted['Date'])
melted.set_index('Date', inplace=True)
states_high_values = ['NY', 'MA', 'CA', 'PA', 'IL', 'IO','OH','TX']
df_states = melted.loc[(melted.index.year >= 1997) & (melted.index.year <= 2018) & (melted['State'].isin(states_high_values))]


# In[46]:


# Group data by RegionID and resample the data to monthly frequency, calculating the mean
region_average = df_states.groupby('RegionID').resample('MS')['value'].mean()

# Reset index to flatten the DataFrame
new_df = region_average.reset_index()

# Print the first few rows of the resulting DataFrame
print(new_df)


# In[47]:


df_states.reset_index(inplace=True)
print(df_states)
#creating training and testing datasets
train = df_states[(df_states['Date'] >= '1997-01-31') & (df_states['Date'] <= '2017-12-31')]
test = df_states[(df_states['Date'] >='2018-01-31') & (df_states['Date'] <='2018-12-31')]


# In[48]:


#choosing columns to keep
to_keep = ['Date','value', 'State', 'RegionID']
train = train[to_keep]
train.columns = ['ds', 'y', 'State', 'zipcode']
print(train.head())

test = test[to_keep]
test.columns = ['ds', 'y', 'State', 'zipcode']
print(test.head())


# In[49]:


# Aggregate the data to ensure unique (ds, zipcode) pairs
train_agg = train.groupby(['ds', 'zipcode']).agg({'y': 'mean'}).reset_index()
test_agg = test.groupby(['ds', 'zipcode']).agg({'y': 'mean'}).reset_index()

# Now pivot the aggregated data
train_pivot = train_agg.pivot(index='ds', columns='zipcode', values='y')
test_pivot = test_agg.pivot(index='ds', columns='zipcode', values='y')


# In[54]:


def prophet_train(data):

# Initialize Prophet model
    model = Prophet(interval_width=0.95, changepoint_prior_scale=6, yearly_seasonality=True,
                seasonality_prior_scale=1, weekly_seasonality=False, daily_seasonality=False)

# Add monthly seasonality
    model.add_seasonality(name='monthly', period=30.5, fourier_order=4)  # Assuming monthly seasonality

# Fit the model
    model.fit(data)

# Make future dataframe for predictions
    future_dates = model.make_future_dataframe(periods=12, freq='MS')

# Generate forecast
    forecast = model.predict(future_dates)

# Select relevant columns and add zipcode for identification
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast['zipcode'] = data['zipcode'].iloc[0]

    return forecast


# In[55]:


zipcodes = train['zipcode'].unique()


# In[56]:


forecast_dfs = []


# In[57]:


for i in zipcodes: 
    data = train[train['zipcode'] == i] 
    result = prophet_train(data) 
    forecast_dfs.append(result)


# In[74]:


forecasts = pd.concat(forecast_dfs, ignore_index=True)


# In[93]:


forecasts.to_csv('forecasts.csv', index=False)


# In[94]:


# Load csv with forecasted results
train_results = pd.read_csv('forecasts.csv', index_col=False)

# Convert ds to datetime and zipcode to object to merge with the test results. 
train_results['ds'] = pd.to_datetime(train_results['ds'])
train_results['ds'] = train_results['ds'] + pd.offsets.MonthEnd(0)

train_results['zipcode'] = train_results.zipcode.astype(object)
train_results.tail()


# In[82]:


train_results.head()


# In[95]:


test['ds'] = pd.to_datetime(test['ds'])
test['zipcode'] = test['zipcode'].astype(str)

train_results['ds'] = pd.to_datetime(train_results['ds'])
train_results['zipcode'] = train_results['zipcode'].astype(str)

# Merge the actual observed values with the predicted values
evaluation_df = test.merge(train_results, on=['ds', 'zipcode'], how='left')
evaluation_df.tail()


# In[96]:


forecast_results = evaluation_df
#finding unique zip codes
zipcode = forecast_results.zipcode.unique()


# In[71]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd


# In[97]:


evaluation_results = pd.DataFrame()

# Iterate through each unique zipcode
for i in zipcode:
    data = forecast_results[forecast_results['zipcode'] == i].reset_index(drop=True)
    
    # Drop rows where y or yhat is NaN before calculating metrics
    data = data.dropna(subset=['y', 'yhat'])
    
    if not data.empty and len(data) > 1:
        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(data['y'], data['yhat'])
        
        # Calculate R-squared (R2)
        rsq = r2_score(data['y'], data['yhat'])
        
        # Assuming data is sorted by 'ds' so that the first and last values make sense for pct_change calculation
        first_value = data.iloc[0]['yhat']
        last_value = data.iloc[-1]['yhat']
        
        # Calculate percentage change, handle division by zero
        pct_change = np.round(((last_value - first_value) / first_value) * 100, 2) if first_value != 0 else np.nan
        
        # Create a DataFrame for the evaluation results of the current zipcode
        data_evaluation = pd.DataFrame({
            'zipcode': i,
            'MAE': mae,
            'R2': rsq,
            'Pct_change': pct_change
        }, index=[0])
        
        # Concatenate the current evaluation results to the main evaluation DataFrame
        evaluation_results = pd.concat([evaluation_results, data_evaluation], ignore_index=True)

# Print the first few rows of evaluation_results to verify the calculations
print(evaluation_results.head())


# In[98]:


plt.figure(figsize=(12,9))
sns.distplot(evaluation_results.MAE)
# Rotate x-labels
plt.xticks(rotation=30)
plt.show()


# In[99]:


# Enlarge the plot
plt.figure(figsize=(12,9))
sns.scatterplot(x='R2', y='Pct_change', data=evaluation_results)
# Rotate x-labels
plt.xticks(rotation=30)
plt.show()


# In[100]:


limit_results = evaluation_results[evaluation_results['R2']>=0]
# Enlarge the plot
plt.figure(figsize=(9,5))
sns.scatterplot(x='R2', y='Pct_change', data=limit_results) # Rotate x-labels
plt.xticks(rotation=30)
plt.show()
# How many models are we losing?
print("There are {} zipcodes in the original data".format(len(evaluation_results)))
print("Now we're limiting to {} zipcodes".format(len(limit_results)))


# In[102]:


# Calculate high risk - high reward
hr_hr = limit_results[limit_results['Pct_change'] >= 0].groupby('R2')[['zipcode', 'Pct_change']].min().sort_values('Pct_change')

# Calculate low risk - high reward
lr_hr = limit_results[limit_results['Pct_change'] > 0].groupby('R2')[['zipcode', 'Pct_change']].max().sort_values('Pct_change', ascending=False)

print(hr_hr.head(10))
print(lr_hr.head(10))


# In[106]:


#finding the markets based on risk reward
final_zip = [77498, 76453, 98096, 97822]
cities = melted[melted.RegionID.isin(final_zip)][['RegionID', 'City', 'State', 'Metro', 'CountyName']].drop_duplicates()


# In[113]:


#top 3 options for zip codes
best_three = lr_hr.groupby(['R2', 'Pct_change'])[['zipcode']].max().reset_index()
best_three = best_three.sort_values(['R2', 'Pct_change'], ascending=False)
best_three_array = np.asarray(best_three.iloc[0:3, 2])
print(best_three)
print(best_three_array)


# SUMMARY
# After the initial observation and data cleaning steps, some basic overview of the data was done which identified the states with the highest average homes across the US.  I used those cities to determine where the best investment opportunities would be.  These cities included Pennsylvania, New York, Iowa, California, Illinois, Texas and Ohio.  
# Before I did this however I did a breakdown on some Arkansas counties as requested.  This included a breakdown on home values in a time series plot in those four counties as can be seen in the graph above.  
# To determine where the best investment opportunities were in the states selected, I developed an r squared, mean absolute error, and percent change statistic model for all the zip codes using the SARIMAX model.  These results were retrieved from a prophet model created.  With these models I was able to determine that zip code 62019 with an impressive R-squared of 0.958594 which is Donnellson, Illinois would be the most ideal place to invest.  This was followed by zip 92034 wiht another impressive R-squared of 0.937428.  This is San Diego, California.  Finally the third best place to invest has been determined to be zip code 91738 with an R-squared of 0.937250 which is Rancho Cucamongo, California. 
