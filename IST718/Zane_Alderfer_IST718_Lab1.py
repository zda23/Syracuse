#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd

import numpy as np
from scipy.stats import uniform
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

import seaborn as sns
from seaborn import PairGrid
#merged dataset from week 2 case study with coaches dataset from coaches9 csv file.  Merged on "School"
df1 = pd.read_csv('/Users/zanealderfer/Downloads/Coaches9.csv')
df2 = pd.read_csv('/Users/zanealderfer/Downloads/coaches.csv')
merged_df = pd.merge(df1, df2, on='School')


# In[15]:


print(merged_df)
merged_df.describe()


# In[16]:


#identifying all the columns
merged_df.columns


# In[17]:


#deleting repeating columns from merged dataset
del merged_df['Coach_y']
del merged_df['SchoolPay_y']
del merged_df['TotalPay_y']


# In[18]:


merged_df.columns


# In[19]:


#removing conferences deemed irrelevant or unimportant
df_v1 = merged_df.drop(merged_df[merged_df['Conference'] == 'Sun Belt'].index)
df_v2 = df_v1.drop(df_v1[df_v1['Conference'] == 'Ind.'].index)


# In[20]:


print(df_v2)


# In[21]:


conference_list = df_v2['Conference'].tolist()
print(conference_list)


# In[27]:


#removing unwanted characters from total pay column in order to convert to numeric
df_v2['TotalPay_x'] = df_v2['TotalPay_x'].str.replace(',', '')
df_v2['TotalPay_x'] = df_v2['TotalPay_x'].str.replace('$', '')
df_v2['TotalPay_x'] = df_v2['TotalPay_x'].str.replace('-', '')
df_v2['TotalPay_x'] = pd.to_numeric(df_v2['TotalPay_x'])
df_v2['TotalPay_x']


# In[28]:


#scaling salaries to single digits for easier readibility and developing conference name list
df_v2['salary_000'] = df_v2['TotalPay_x']/1000000
Mt_West = df_v2[df_v2['Conference'] == 'Mt. West']
MAC = df_v2[df_v2['Conference'] == 'MAC']
SEC = df_v2[df_v2['Conference'] == 'SEC']
Pac_12 = df_v2[df_v2['Conference'] == 'Pac-12']
Big_12 = df_v2[df_v2['Conference'] == 'Big 12']
C_USA = df_v2[df_v2['Conference'] == 'C-USA']
ACC = df_v2[df_v2['Conference'] == 'ACC']
Big_Ten = df_v2[df_v2['Conference'] == 'Big Ten']
data = [Mt_West['salary_000'], MAC['salary_000'], 
    SEC['salary_000'], Pac_12['salary_000'], 
    Big_12['salary_000'], C_USA['salary_000'], 
    ACC['salary_000'], Big_Ten['salary_000']]
ordered_Conference_names = ['Mt. West', 'MAC', 'SEC', 'Pac-12', 'Big 12', 'C-USA', 'ACC', 'Big Ten']


# In[74]:


print(ACC)
ACC.describe()


# In[75]:


print(Big_Ten)
Big_Ten.describe()


# In[31]:


#box plots distinguishing average pay for each conference
fig, axis = plt.subplots()
axis.set_xlabel('Conference')
axis.set_ylabel('Salary (millions)')
day_plot = plt.boxplot(data, sym='o', vert=1, whis=1.5)
plt.setp(day_plot['boxes'], color = 'black')    
plt.setp(day_plot['whiskers'], color = 'black')    
plt.setp(day_plot['fliers'], color = 'black', marker = 'o')
axis.set_xticklabels(ordered_Conference_names)
plt.show()
plt.savefig('fig_advert_promo_dodgers_eda_day_of_week_Python.pdf', 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)


# In[46]:


#Facetgrids that distinguish correlations between total pay and another variable
sns.set(style="darkgrid")

g = sns.FacetGrid(df_v2, hue="Conference",
                 hue_order=['Mt. West', 'MAC', 'SEC', 'Pac-12', 'Big 12', 'C-USA', 'ACC', 'Big Ten'],)
g.map(plt.scatter, " Graduation Rate (GSR) ", "TotalPay_x", alpha=.7)
g.add_legend();
plt.show()

g = sns.FacetGrid(df_v2, hue="Conference",
                 hue_order=['Mt. West', 'MAC', 'SEC', 'Pac-12', 'Big 12', 'C-USA', 'ACC', 'Big Ten'],)
g.map(plt.scatter, "StadSize", "TotalPay_x", alpha=.7)
g.add_legend();
plt.show()

g = sns.FacetGrid(df_v2, hue="Conference",
                 hue_order=['Mt. West', 'MAC', 'SEC', 'Pac-12', 'Big 12', 'C-USA', 'ACC', 'Big Ten'],)
g.map(plt.scatter, " Ratio ", "TotalPay_x", alpha=.7)
g.add_legend();
plt.show()

g = sns.FacetGrid(df_v2, hue="Conference",
                 hue_order=['Mt. West', 'MAC', 'SEC', 'Pac-12', 'Big 12', 'C-USA', 'ACC', 'Big Ten'],)
g.map(plt.scatter, " TrueRank ", "TotalPay_x", alpha=.7)
g.add_legend();
plt.show()

g = sns.FacetGrid(df_v2, hue="Conference",
                 hue_order=['Mt. West', 'MAC', 'SEC', 'Pac-12', 'Big 12', 'C-USA', 'ACC', 'Big Ten'],)
g.map(plt.scatter, "PointsPerGame", "TotalPay_x", alpha=.7)
g.add_legend();
plt.show()


# In[51]:


#frequency of pay across the NCAA
plt.hist(df_v2['TotalPay_x'], stacked = False, rwidth = .9)
plt.title("Total Pay Histogram")
plt.xlabel('Total Pay')
plt.ylabel('Frequency')

plt.show()


# In[57]:


# BLOCK FOR ORDERING DATA

# map day_of_week to ordered_day_of_week 
Conf_to_ordered_Conf = {'Mt. West' : '1Mt. West', 
    'MAC' : '2MAC', 
    'SEC' : '3SEC', 
    'Pac-12' : '4Pac-12', 
    'Big 12' : '5Big 12',
    'C-USA' : '6C-USA',
    'ACC' : '7ACC',
    'Big Ten' : '8Big Ten'}
df_v2['ordered_Conf'] = df_v2['Conference'].map(Conf_to_ordered_Conf)   


# In[69]:


np.random.seed(1234)
df_v2['runiform'] = uniform.rvs(loc = 0, scale = 1, size = len(df_v2))
df_train = df_v2[df_v2['runiform'] >= 0.33]
df_test = df_v2[df_v2['runiform'] < 0.33]
# check training data frame
print('\ndf_train data frame (rows, columns): ',df_train.shape)
print(df_train.head())
# check test data frame
print('\ndd_test data frame (rows, columns): ',df_test.shape)
print(df_test.head())

# specify a simple
my_model = str('TotalPay_x ~ Conference + PointsPerGame')
 
# fit the model to the training set
train_model_fit = smf.ols(my_model, data = df_v2).fit()
# summary of model fit to the training set
print(train_model_fit.summary())
# training set predictions from the model fit to the training set
df_train['predict_TotalPay'] = train_model_fit.fittedvalues

# test set predictions from the model fit to the training set
df_test['predict_TotalPay'] = train_model_fit.predict(df_v2)


# In[70]:


np.random.seed(1234)
df_v2['runiform'] = uniform.rvs(loc = 0, scale = 1, size = len(df_v2))
df_train = df_v2[df_v2['runiform'] >= 0.33]
df_test = df_v2[df_v2['runiform'] < 0.33]
# check training data frame
print('\ndf_train data frame (rows, columns): ',df_train.shape)
print(df_train.head())
# check test data frame
print('\ndd_test data frame (rows, columns): ',df_test.shape)
print(df_test.head())

# specify a simple
my_model_2 = str('TotalPay_x ~ Conference + StadSize')
 
# fit the model to the training set
train_model_fit = smf.ols(my_model_2, data = df_v2).fit()
# summary of model fit to the training set
print(train_model_fit.summary())
# training set predictions from the model fit to the training set
df_train['predict_TotalPay'] = train_model_fit.fittedvalues

# test set predictions from the model fit to the training set
df_test['predict_TotalPay'] = train_model_fit.predict(df_v2)


# In[71]:


print('\nProportion of Test Set Variance Accounted for: ',    round(np.power(df_test['TotalPay_x'].corr(df_test['predict_TotalPay']),2),3))

# use the full data set to obtain an estimate of the increase in
# attendance due to bobbleheads, controlling for other factors 
my_model_fit = smf.ols(my_model, data = df_v2).fit()
print(my_model_fit.summary())

print('\nEstimated Effect of PointsPerGame on Total Pay: ',    round(my_model_fit.params[9],0))


# In[72]:


print('\nProportion of Test Set Variance Accounted for: ',    round(np.power(df_test['TotalPay_x'].corr(df_test['predict_TotalPay']),2),3))

# use the full data set to obtain an estimate of the increase in
# attendance due to bobbleheads, controlling for other factors 
my_model_fit = smf.ols(my_model_2, data = df_v2).fit()
print(my_model_fit.summary())

print('\nEstimated Effect of Stadium Size on Total Pay: ',    round(my_model_fit.params[9],0))

