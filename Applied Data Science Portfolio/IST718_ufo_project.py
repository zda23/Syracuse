#!/usr/bin/env python
# coding: utf-8

# # IST 718 Final Project Code
# Zane Alderfer, Ben Heindl, Victoria Haley
# 3/18/2024

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


# In[2]:


# Load the CSV file
file_path = './master_ufo_data.csv'
ufo_data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
ufo_data.head()


# In[3]:


ufo_data_usa = ufo_data[ufo_data['country'].str.lower() == 'us']
ufo_data_usa.head()


# In[11]:


#Funtion to analyze the data
def analyze_data(data):
    stats = data.describe(include='all')
    print('Descriptive Statistics')
    print(stats)
    nulls = data.isnull().sum()
    print()
    print('Any null values?')
    print(nulls)
    dups = data.duplicated().values.any()
    print()
    print('Any duplicates?')
    print(dups)
    print()
    print('First 5 Rows of Data Frame:')
    print(data.head())


# In[12]:


# Drop 'state' because there is already a 'State' column and it includes Canadian states
ufo_data_usa.drop(columns=['state'], inplace=True)


analyze_data(ufo_data_usa)
print(ufo_data_usa.dtypes)


# In[13]:


# Necessary data transformations

# Date time
ufo_data['datetime'] = pd.to_datetime(ufo_data['datetime'])
ufo_data['date posted'] = pd.to_datetime(ufo_data['date posted'])


# Categorical columns
categorical_columns = ['State', 'country', 'shape', 'city', 'FIPS']
ufo_data[categorical_columns] = ufo_data[categorical_columns].astype('category')

# Numerical columns
ufo_data['duration (seconds)'] = ufo_data['duration (seconds)'].astype(int)

# Creating USA based data frame
ufo_data_usa = ufo_data[ufo_data['country'].str.lower() == 'us']


# In[4]:


# Calculate the number of sightings per state
state_counts = ufo_data_usa['State'].value_counts().reset_index()

# Start the plot
plt.figure(figsize=(12, 10))
sns.set_style("whitegrid")

# Assign the 'state' to hue to ensure colors are consistent
sns.barplot(x='State', y='index', data=state_counts, dodge=False, palette='Spectral')

# Set the title and labels
plt.title('UFO Sightings in United States')
plt.xlabel('Number of Sightings')
plt.ylabel('State')

# Remove the right and upper spines
sns.despine()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[5]:


#pip install geopandas shapely


# In[6]:


import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd

# Create a GeoDataFrame from the DataFrame
geometry = [Point(xy) for xy in zip(ufo_data_usa['longitude'], ufo_data_usa['latitude'])]
gdf = gpd.GeoDataFrame(ufo_data_usa, geometry=geometry)

# Load the state boundaries Shapefile
states_shapefile = gpd.read_file("C:/Users/benjh/Desktop/School/IST 718/UFO Project/Data/ne_10m_admin_0_countries_usa/ne_10m_admin_0_countries_usa.shp")

# Plot the map with detailed state borders
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
states_shapefile.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5)

# Plot the UFO sightings
gdf.plot(ax=ax, marker='o', column='state', cmap='tab20', markersize=1, alpha=0.6)

# Set the axis limits to focus on the mainland US
ax.set_xlim([-130, -65])
ax.set_ylim([24, 50])

# Remove axis spines, ticks, and labels for a cleaner look
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

# Set background color
fig.patch.set_facecolor('white')  # Set the background color of the figure
ax.set_facecolor('white')  # Set the background color of the axes

# Set the title
plt.title('Map UFO sightings in United States', fontsize=20, weight='bold')

# Adjust the padding and show the plot
plt.tight_layout(pad=0.5)
plt.show()


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Calculate the number of sightings for each shape
shape_counts = ufo_data_usa['shape'].value_counts().reset_index(name='count')

# Start the plot
plt.figure(figsize=(12, 8))  # Adjust the size as needed
sns.set_style("whitegrid")

# Create a bar plot
ax = sns.barplot(x='count', y='index', data=shape_counts, palette='Spectral')

# Set the title and labels
plt.title('Number of UFO Sightings by Shape')
plt.xlabel('Number of Sightings')
plt.ylabel('Shape')

# Show the plot
plt.show()


# In[16]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# Extract the hour part and assign it to a new column 'hour'
ufo_data_usa['hour'] = ufo_data_usa['datetime'].dt.hour

# Bins
counts, bins = np.histogram(ufo_data_usa['hour'], bins=24, range=(0, 24))

# Normalize the counts for color intensity
normed_counts = counts / counts.max()

# Create a color map using the new function
cmap = mpl.colormaps.get_cmap('Blues')

# Create the bar plot
plt.figure(figsize=(10, 6))
for count, bin_edge, normed_count in zip(counts, bins, normed_counts):
    plt.bar(bin_edge, count, width=bins[1] - bins[0], color=cmap(normed_count), edgecolor='grey')

# Set the limits of the x-axis to fit the bins
plt.xlim([bins.min(), bins.max()])

# Set the title and labels
plt.title('Correlation between daytime and UFO sightings')
plt.xlabel('Hour of the day')
plt.ylabel('Number of sightings')

# Show the plot
plt.show()


# In[17]:


import plotly.express as px

# Group the data by hour and shape and count the occurrences
shapes_daytime = ufo_data_usa.groupby([ufo_data_usa['datetime'].dt.hour, 'shape']).size().reset_index(name='count')

# Create the scatter plot
fig = px.scatter(shapes_daytime, x='datetime', y='shape', size='count', color='count')

# Update the layout to adjust the width and height of the figure
fig.update_layout(
    title='Correlation between daytime and UFO Shape',
    width=1000,  
    height=600   
)

# Show the plot
fig.show()


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Extract the year from the 'datetime' and 'date posted' columns
ufo_data_usa['year'] = ufo_data_usa['datetime'].dt.year
ufo_data_usa['report_year'] = ufo_data_usa['date posted'].dt.year

# Prep the data for plotting
sightings_per_year = ufo_data_usa.groupby('year').size().reset_index(name='sightings')
reports_per_year = ufo_data_usa.groupby('report_year').size().reset_index(name='reports')

# Set background
sns.set_style('whitegrid')

# Create the plot
plt.figure(figsize=(12, 6))

# Plot sightings per year with a trendline
sns.regplot(data=sightings_per_year, x='year', y='sightings', scatter_kws={'s':0}, line_kws={'color':'green', 'label':"Sightings Trend"})

# Plot reports per year with a trendline
sns.regplot(data=reports_per_year, x='report_year', y='reports', scatter_kws={'s':0}, line_kws={'color':'red', 'label':"Reports Trend"})

# Add actual data points on top of the trendline for sightings
sns.scatterplot(data=sightings_per_year, x='year', y='sightings', color='green', label='Sightings')

# Add actual data points on top of the trendline for reports
sns.scatterplot(data=reports_per_year, x='report_year', y='reports', color='red', label='Reports')

# Set the title and labels
plt.title('UFO sightings / UFO reports by year')
plt.xlabel('Year')
plt.ylabel('Count')

# Remove the spines and gridlines for a cleaner look
sns.despine()

# Show the legend
plt.legend()

# Show the plot
plt.show()


# # Potentially Interesting Finds

# In[42]:


import matplotlib.pyplot as plt
import pandas as pd

# Convert the datetime column to datetime format if it's not already
ufo_data_usa['datetime'] = pd.to_datetime(ufo_data_usa['datetime'])

# Group the data by datetime and count the number of sightings for each time period
sightings_over_time = ufo_data_usa.groupby(pd.Grouper(key='datetime', freq='Y')).size()

# Plot the time series
plt.figure(figsize=(12, 6))
sightings_over_time.plot()
plt.title('Number of UFO Sightings Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Sightings')
plt.show()


# Filter the DataFrame to include data starting from the year 1995
ufo_data_since_1995 = ufo_data_usa[ufo_data_usa['datetime'].dt.year >= 1995]

# Group the filtered data by datetime and count the number of sightings for each time period
sightings_over_time = ufo_data_since_1995.groupby(pd.Grouper(key='datetime', freq='M')).size()

# Plot the time series
plt.figure(figsize=(12, 6))
sightings_over_time.plot()
plt.title('Number of UFO Sightings Over Time (Monthly Since 1995)')
plt.xlabel('Date')
plt.ylabel('Number of Sightings')

plt.show()


# In[35]:


ufo_data_usa['year'] = ufo_data_usa['datetime'].dt.year

# Group the data by state and year, and count the number of sightings
sightings_by_state_year = ufo_data_usa.groupby(['State', 'year']).size().unstack(fill_value=0)


from matplotlib.animation import FuncAnimation
from IPython.display import HTML


# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Define the update function
def update(frame):
    ax.clear()
    ax.plot(sightings_by_state_year.columns[:frame+1], sightings_by_state_year.iloc[:, :frame+1].T)
    ax.set_title(f'UFO Sightings Over Time by State (Year {sightings_by_state_year.columns[frame]})')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Sightings')
    ax.legend(sightings_by_state_year.index, loc='upper left', bbox_to_anchor=(1, 1), title='States', fancybox=True, shadow=True)

# Create the animation
ani = FuncAnimation(fig, update, frames=len(sightings_by_state_year.columns), repeat=False)

# Display the animation
HTML(ani.to_jshtml())


# In[38]:


# Specify the filename for the GIF
ufo_TS_gif = 'animated_time_series.gif'

# Create a writer object to save the animation as a GIF
writer = animation.PillowWriter(fps=10)

# Save the animation as a GIF file
ani.save(ufo_TS_gif, writer=writer)

# Print a message to confirm the successful saving of the GIF file
print(f"Animation saved as '{ufo_TS_gif}'")


# In[49]:


# Extract the year from the datetime column
ufo_data_usa['year'] = ufo_data_usa['datetime'].dt.year

# Find the year with the highest number of sightings
year_with_highest_sightings = ufo_data_usa['year'].value_counts().idxmax()

# Count the number of sightings for that year
count_of_sightings_in_highest_year = sightings_in_highest_year.shape[0]

# Filter the DataFrame for sightings in the year with the highest number
sightings_in_highest_year = ufo_data_usa[ufo_data_usa['year'] == year_with_highest_sightings]

# Find the state, shape, and duration for these sightings
state_with_highest_sightings = sightings_in_highest_year['state'].value_counts().idxmax()
shape_with_highest_sightings = sightings_in_highest_year['shape'].value_counts().idxmax()
duration_with_highest_sightings = sightings_in_highest_year['duration (seconds)'].max()

from datetime import timedelta

# Convert the duration to a timedelta object
duration_timedelta = timedelta(seconds=duration_with_highest_sightings)

# Extract hours, minutes, and seconds from the timedelta object
duration_hours = duration_timedelta.days * 24 + duration_timedelta.seconds // 3600
duration_minutes = (duration_timedelta.seconds % 3600) // 60
duration_seconds = duration_timedelta.seconds % 60


# Print the results
print(f"The year with the highest number of sightings: {year_with_highest_sightings}")
print(f"The count of sightings in the year with the highest number of sightings: {count_of_sightings_in_highest_year}")
print(f"The state with the highest number of sightings: {state_with_highest_sightings}")
print(f"The most common shape of sightings: {shape_with_highest_sightings}")
print(f"The longest duration of sighting: {duration_hours} hours, {duration_minutes} minutes, {duration_seconds} seconds")


# In[48]:


# Find the index of the row with the maximum duration
max_duration_index = ufo_data['duration (seconds)'].idxmax()

# Retrieve the row with the maximum duration
row_with_max_duration = ufo_data.loc[max_duration_index]

# Print the row
print(row_with_max_duration)


# I checked this out because of the duration that printed above to see what was going on and it looks like something weird happened with the duration(seconds) column. Apparently, this sighting lasted 66,276,000 seconds, which is 767.5 days. Unfortunately, it looks like something went wrong with the data here.

# # Demographic Analysis

# In[62]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Aggregate sightings by FIPS
sightings_count = ufo_data_usa.groupby('FIPS').size().reset_index(name='sightings_count')

# Aggregate socio-economic variables by FIPS code
socio_economic_agg = ufo_data_usa.groupby('FIPS').agg({'Poverty Estimate, All Ages': 'mean',
                                                       'Poverty Percent, All Ages': 'mean',
                                                       'Median Household Income': 'mean',
                                                       'Estimated Population 2022': 'mean',
                                                       'BIRTHS 2022': 'mean',
                                                       'DEATHS 2022': 'mean'})

# Calculate correlation between the number of sightings and each socio-economic variable
correlation_p_values = {}
for column in socio_economic_agg.columns:
    correlation_coefficient, p_value = pearsonr(sightings_count['sightings_count'], socio_economic_agg[column])
    correlation_p_values[column] = (correlation_coefficient, p_value)

# Print correlation coefficients and p-values
for column, (correlation_coefficient, p_value) in correlation_p_values.items():
    print(f"Correlation between Number of Sightings and {column}: {correlation_coefficient}")
    print(f"P-value: {p_value}")


# ## Interpretations:
# 
# Magnitude of Correlation Coefficients:
# 
# - Weak correlation coefficients indicate that the relationship between the demographic variables and the number of UFO sightings is very weak, regardless of statistical significance. This suggests that changes in demographic variables are not strongly associated with changes in the number of UFO sightings.
# 
# Significance Level:
# 
# - The statistically significant p-values suggest that the observed correlations are unlikely to have occurred by random chance alone. However, statistical significance does not imply practical significance. In this case, while the correlations are statistically significant, their practical relevance may be limited due to their weak magnitude.
# 
# Interpretation of Weak Correlations:
# 
# - Weak correlations do not necessarily imply causation or strong associations between variables. Other factors not captured in the dataset may influence the number of UFO sightings.
# - It's possible that the demographic variables included in the analysis may not be strong predictors of UFO sighting. Consider exploring additional variables or factors that could potentially influence UFO sightings.
# 
# Contextual Considerations:
# 
# - Consider the context and limitations of the data when interpreting correlation results. Factors such as data quality, measurement error, and sample size could influence the observed correlations.
# - Demographic variables may have indirect or complex relationships with UFO sightings that are not fully captured by simple correlation analysis. Exploring more sophisticated modeling techniques or considering interactions between variables may provide further insights.
# 
# # Shocking: Aliens not appearing more often/longer among higher populations
# # Also, Aliens don't appear to discriminate against income levels

# In[11]:


# Aggregate sightings by state
sightings_count = ufo_data_usa.groupby('State').size().reset_index(name='sightings_count')

# Find the state code with the highest number of sightings
state_with_highest_sightings = sightings_count['State'].iloc[sightings_count['sightings_count'].idxmax()]

# Count the number of sightings for that FIPS code
count_of_sightings_in_highest_state = sightings_count['sightings_count'].max()

# Filter the DataFrame for sightings in the FIPS code with the highest number
sightings_in_highest_state = ufo_data_usa[ufo_data_usa['State'] == state_with_highest_sightings]

# Calculate socio-economic statistics for these sightings
poverty_estimate_in_highest_state = sightings_in_highest_state['Poverty Estimate, All Ages'].mean()
poverty_percent_in_highest_state = sightings_in_highest_state['Poverty Percent, All Ages'].mean()
median_household_income_in_highest_state = sightings_in_highest_state['Median Household Income'].mean()
estimated_population_in_highest_state = sightings_in_highest_state['Estimated Population 2022'].mean()
births_in_highest_state = sightings_in_highest_state['BIRTHS 2022'].mean()
deaths_in_highest_state = sightings_in_highest_state['DEATHS 2022'].mean()

# Print the results
print(f"The State code with the highest number of sightings: {state_with_highest_sightings}")
print(f"The count of sightings in the State with the highest number of sightings: {count_of_sightings_in_highest_state}")
print(f"The mean Poverty Estimate: {poverty_estimate_in_highest_state}")
print(f"The mean Poverty Percent: {poverty_percent_in_highest_state}")
print(f"The mean Median Household Income: {median_household_income_in_highest_state}")
print(f"The mean Estimated Population 2022: {estimated_population_in_highest_state}")
print(f"The mean BIRTHS 2022: {births_in_highest_state}")
print(f"The mean DEATHS 2022: {deaths_in_highest_state}")

print()

# Filter out Washington DC from the DataFrame before finding the state with the lowest number of sightings
sightings_count_without_dc = sightings_count[sightings_count['State'] != 'DC']

# Find the state code with the lowest number of sightings (excluding Washington DC)
state_with_lowest_sightings = sightings_count_without_dc.loc[sightings_count_without_dc['sightings_count'].idxmin(), 'State']

# Count the number of sightings for that FIPS code
count_of_sightings_in_lowest_state = sightings_count_without_dc['sightings_count'].min()

# Filter the DataFrame for sightings in the FIPS code with the lowest number (excluding Washington DC)
sightings_in_lowest_state = ufo_data_usa[ufo_data_usa['State'] == state_with_lowest_sightings]

# Find the mean socio-economic statistics for these sightings
mean_poverty_estimate = sightings_in_lowest_state['Poverty Estimate, All Ages'].mean()
mean_poverty_percent = sightings_in_lowest_state['Poverty Percent, All Ages'].mean()
mean_median_income = sightings_in_lowest_state['Median Household Income'].mean()
mean_population = sightings_in_lowest_state['Estimated Population 2022'].mean()
mean_births = sightings_in_lowest_state['BIRTHS 2022'].mean()
mean_deaths = sightings_in_lowest_state['DEATHS 2022'].mean()

# Print the results
print(f"The State with the lowest number of sightings: {state_with_lowest_sightings}")
print(f"The count of sightings in the state with the lowest number of sightings: {count_of_sightings_in_lowest_state}")
print(f"The mean Poverty Estimate: {mean_poverty_estimate}")
print(f"The mean Poverty Percent: {mean_poverty_percent}")
print(f"The mean Median Household Income: {mean_median_income}")
print(f"The mean Estimated Population 2022: {mean_population}")
print(f"The mean BIRTHS 2022: {mean_births}")
print(f"The mean DEATHS 2022: {mean_deaths}")


# In[19]:


# Median Household Income

import plotly.graph_objects as go


sightings_by_state = ufo_data.groupby('State')['datetime'].count().reset_index()
sightings_by_state.columns = ['State', 'Total Sightings']

# Merge sightings_by_fips with ufo_data_usa to get demographic variables
merged_data = pd.merge(sightings_by_state, ufo_data_usa, on='State', how='left')

# Select and append demographic variables to sightings_by_fips
demographic_variables = ['Poverty Estimate, All Ages', 'Poverty Percent, All Ages', 
                         'Median Household Income', 'Estimated Population 2022', 
                         'BIRTHS 2022', 'DEATHS 2022']
sightings_by_state[demographic_variables] = merged_data[demographic_variables]

# Calculate average
average_sightings = sightings_by_state['Total Sightings'].mean()

# Creating the grouped bar chart
fig = go.Figure()

# Bars for total sightings
fig.add_trace(go.Bar(
    x=sightings_by_state['State'],
    y=sightings_by_state['Total Sightings'],
    name='Total Sightings',
    marker_color='indianred'
))

# Line for average sighting
fig.add_trace(go.Scatter(
    x=sightings_by_state['State'],
    y=[average_sightings] * len(sightings_by_state),
    mode='lines',
    name='Average Sightings',
    line=dict(color='blue', width=3, dash='dot')
))

# Update layout
fig.update_layout(
    title='UFO Sightings: States vs Average',
    xaxis_title='State',
    yaxis_title='Total Sightings',
    barmode='group'
)

# Run this in a Python environment to display the figure
fig.show()


# In[20]:


# Starting with Poverty Percent, All Ages

# Calculate quantiles
quantiles = ufo_data_usa['Poverty Percent, All Ages'].quantile([0.25, 0.5, 0.75])

# Define bin boundaries based on quantiles
bins = [ufo_data_usa['Poverty Percent, All Ages'].min(), quantiles[0.25], quantiles[0.5], quantiles[0.75], ufo_data_usa['Median Household Income'].max()]

# Create labels for bins
labels = ['Bottom 25%', '25% - 50%', '50% - 75%', 'Top 25%']

# Bin Poverty Percent Category
ufo_data_usa['Poverty Percent Category'] = pd.cut(ufo_data_usa['Poverty Percent, All Ages'], bins=bins, labels=labels)


# In[21]:


# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(ufo_data_usa[ufo_data_usa['Poverty Percent Category'] == 'Bottom 25%']['duration (seconds)'], alpha=0.5, label='Bottom 25%')
plt.hist(ufo_data_usa[ufo_data_usa['Poverty Percent Category'] == '25% - 50%']['duration (seconds)'], alpha=0.5, label='25% - 50%')
plt.hist(ufo_data_usa[ufo_data_usa['Poverty Percent Category'] == '50% - 75%']['duration (seconds)'], alpha=0.5, label='50% - 75%')
plt.hist(ufo_data_usa[ufo_data_usa['Poverty Percent Category'] == 'Top 25%']['duration (seconds)'], alpha=0.5, label='Top 25%')
plt.title('Distribution of UFO Sightings by Poverty Percent Category')
plt.xlabel('Duration (in seconds)')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[22]:


# Count sightings in each income category
sightings_by_income = ufo_data_usa['Poverty Percent Category'].value_counts()

# Plot bar chart
plt.figure(figsize=(10, 6))
sightings_by_income.plot(kind='bar', color='skyblue')
plt.title('UFO Sightings by Poverty Percent Category')
plt.xlabel('Poverty Percent Category')
plt.ylabel('Number of Sightings')
plt.xticks(rotation=45)
plt.show()


# In[23]:


from scipy.stats import kruskal

# Perform Kruskal-Wallis H Test
h_statistic, p_value = kruskal(ufo_data_usa[ufo_data_usa['Poverty Percent Category'] == 'Bottom 25%']['duration (seconds)'],
                               ufo_data_usa[ufo_data_usa['Poverty Percent Category'] == '25% - 50%']['duration (seconds)'],
                               ufo_data_usa[ufo_data_usa['Poverty Percent Category'] == '50% - 75%']['duration (seconds)'],
                               ufo_data_usa[ufo_data_usa['Poverty Percent Category'] == 'Top 25%']['duration (seconds)'])

# Print results
print("Kruskal-Wallis H Statistic:", h_statistic)
print("Kruskal-Wallis p-value:", p_value)


# Based on the results of the Kruskal-Wallis test, it seems that there are statistically significant differences in the median durations of UFO sightings across the poverty percent categories in the USA.

# In[24]:


sightings_by_state = ufo_data.groupby('State')['datetime'].count().reset_index()
sightings_by_state.columns = ['State', 'Total Sightings']

# Merge sightings_by_fips with ufo_data_usa to get demographic variables
merged_data = pd.merge(sightings_by_state, ufo_data_usa, on='State', how='left')

# Select and append demographic variables to sightings_by_fips
demographic_variables = ['Poverty Estimate, All Ages', 'Poverty Percent, All Ages', 
                         'Median Household Income', 'Estimated Population 2022', 
                         'BIRTHS 2022', 'DEATHS 2022']
sightings_by_state[demographic_variables] = merged_data[demographic_variables]


sightings_by_state


# In[25]:


# Selecting relevant columns for correlation analysis
cm_variables = sightings_by_state[['Total Sightings', 'Poverty Estimate, All Ages', 'Poverty Percent, All Ages', 'Median Household Income', 'Estimated Population 2022', 'BIRTHS 2022', 'DEATHS 2022']]

# Calculating correlation matrix
correlation_matrix = cm_variables.corr()

# Displaying correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")  
plt.title('Correlation Matrix of Demographic Variables')
plt.show()


# In[26]:


from scipy.stats import pearsonr

# Calculate Pearson correlation coefficient and p-value for each demographic variable
correlation_results = {}

for column in cm_variables.columns[1:]:
    correlation_coefficient, p_value = pearsonr(cm_variables['Total Sightings'], cm_variables[column])
    correlation_results[column] = {'Correlation Coefficient': correlation_coefficient, 'P-Value': p_value}

# Display correlation results
for column, results in correlation_results.items():
    print(f"{column}:")
    print(f"Correlation Coefficient: {results['Correlation Coefficient']}")
    print(f"P-Value: {results['P-Value']}\n")


# ### Poverty Estimate, All Ages:
# 
# - Correlation Coefficient: -0.197
# - P-Value: 0.170
# - Interpretation: There is a weak negative correlation between "Total Sightings" and "Poverty Estimate, All Ages," but it is not statistically significant at the conventional significance level (p > 0.05).
# 
# ### Poverty Percent, All Ages:
# 
# - Correlation Coefficient: -0.013
# - P-Value: 0.930
# - Interpretation: There is a very weak negative correlation between "Total Sightings" and "Poverty Percent, All Ages," and it is not statistically significant.
# 
# ### Median Household Income:
# 
# - Correlation Coefficient: -0.018
# - P-Value: 0.901
# - Interpretation: There is a very weak negative correlation between "Total Sightings" and "Median Household Income," and it is not statistically significant.
# 
# ### Estimated Population 2022:
# 
# - Correlation Coefficient: -0.174
# - P-Value: 0.226
# - Interpretation: There is a weak negative correlation between "Total Sightings" and "Estimated Population 2022," but it is not statistically significant.
# 
# ### BIRTHS 2022:
# 
# - Correlation Coefficient: -0.165
# - P-Value: 0.253
# - Interpretation: There is a weak negative correlation between "Total Sightings" and "BIRTHS 2022," but it is not statistically significant.
# 
# ### DEATHS 2022:
# 
# - Correlation Coefficient: -0.178
# - P-Value: 0.216
# - Interpretation: There is a weak negative correlation between "Total Sightings" and "DEATHS 2022," but it is not statistically significant.
# 
# Overall, based on the p-values, none of the correlations between "Total Sightings" and the demographic variables are statistically significant at the conventional significance level (p > 0.05). Therefore, we cannot conclude that there is a significant linear relationship between "Total Sightings" and any of the demographic variables based on the Pearson correlation analysis.

# In[27]:


import statsmodels.api as sm

# Add a constant term to the predictor variables for the regression model
cm_variables_with_constant = sm.add_constant(cm_variables.drop(columns='Total Sightings'))

# Fit the linear regression model
model = sm.OLS(cm_variables['Total Sightings'], cm_variables_with_constant)
results = model.fit()

# Display regression results
print(results.summary())


# ### Overall Model Fit:
# 
# - R-squared: 0.108
# - Adjusted R-squared: -0.016
# 
# ### Significance of Coefficients:
# 
# - None of the coefficients for the demographic variables are statistically significant at the conventional significance level (p > 0.05).
# 
# ### Other Statistics:
# 
# - F-statistic: 0.8721
# - The p-value associated with the F-statistic is 0.523, indicating that the overall model is not statistically significant.
# 
# In summary, based on the linear regression analysis, the model does not provide strong evidence to support a significant linear relationship between "Total Sightings" and the demographic variables included in the analysis. 

# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select relevant columns for the pairplot (excluding State column)
selected_columns = ['Total Sightings', 'Poverty Estimate, All Ages', 'Poverty Percent, All Ages', 'Median Household Income', 'Estimated Population 2022', 'BIRTHS 2022', 'DEATHS 2022']
data_for_pairplot = sightings_by_state[selected_columns]

# Create pairplot
sns.pairplot(data_for_pairplot)
plt.show()


# Further evidence of the (lack of a) relationship between the number of sightings and socio-economic factors.

# # Where in the US are you most likely to see a UFO?

# In[29]:


# Group the data by state and count the number of sightings in each state
sightings_by_state = ufo_data_usa['State'].value_counts()

# Plotting the bar plot
plt.figure(figsize=(12, 8))
sightings_by_state.plot(kind='bar', color='green')
plt.title('Number of UFO Sightings by State')
plt.xlabel('State')
plt.ylabel('Number of Sightings')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[30]:


# Top 10 States
sightings_by_state.head(10)

# Top 9 states all have over 2000 sightings


# In[31]:


# Get the top 10 states with the highest number of sightings
top_states = sightings_by_state.head(10).index.tolist()

# Filter the data for sightings in the top 10 states
top_states_data = ufo_data[ufo_data['state'].isin(top_states)]

# Iterate over each state
for state in top_states:
    # Filter data for the current state
    state_data = top_states_data[top_states_data['state'] == state]
    
    # Group the data by city and count the number of sightings in each city
    sightings_by_city = state_data['city'].value_counts().head(10)
    
    # Plotting the bar plot for the current state
    plt.figure(figsize=(10, 6))
    plt.bar(sightings_by_city.index, sightings_by_city.values, color='skyblue')
    plt.title(f'Top Cities for UFO Sightings in {state}')
    plt.xlabel('City')
    plt.ylabel('Number of Sightings')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In[32]:


import matplotlib.pyplot as plt

# Group the data by state and count the number of sightings in each state
sightings_by_state = ufo_data_usa['State'].value_counts()

# Get the top 4 states with the highest number of sightings
top_states = sightings_by_state.head(4).index.tolist()

# Filter the data for sightings in the top 4 states
top_states_data = ufo_data[ufo_data['State'].isin(top_states)]

# Create subplots with a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Iterate over each state and corresponding subplot
for i, state in enumerate(top_states):
    # Filter data for the current state
    state_data = top_states_data[top_states_data['State'] == state]
    
    # Group the data by city and count the number of sightings in each city
    sightings_by_city = state_data['city'].value_counts().head(4)
    
    # Plotting the bar plot for the current state in the corresponding subplot
    ax = axs[i // 2, i % 2]  # Get the correct subplot
    ax.bar(sightings_by_city.index, sightings_by_city.values, color='skyblue')
    ax.set_title(f'Top Cities for UFO Sightings in {state}')
    ax.set_xlabel('City')
    ax.set_ylabel('Number of Sightings')
    ax.tick_params(axis='x', rotation=45)
    #ax.tight_layout()

# Adjust layout and display the subplots
plt.tight_layout()
plt.show()


# # When is the next UFO sighting likely to occur and where?

# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of UFO sightings over time
plt.figure(figsize=(12, 6))
sns.histplot(data=ufo_data_usa, x='datetime', bins=50, kde=True)
plt.title('Distribution of UFO Sightings Over Time')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.show()


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Set 'datetime' column as index
#ufo_data_usa.set_index('datetime', inplace=True)

# Compute the frequency of sightings per year
sightings_per_year = ufo_data_usa.resample('Y').size()

# Decompose the time series with annual frequency
decomposition = sm.tsa.seasonal_decompose(sightings_per_year, model='additive', period=1)  # Period = 1 for annual frequency
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the decomposed components
plt.figure(figsize=(12, 8))

plt.subplot(411)
plt.plot(sightings_per_year, label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

# Autocorrelation and Partial Autocorrelation Analysis
sm.graphics.tsa.plot_acf(sightings_per_year, lags=10)
plt.show()

sm.graphics.tsa.plot_pacf(sightings_per_year, lags=10)
plt.show()


# Identical 'Original' and 'Trend' plots implies that any systematic changes or movements in the data are minimal or negligible. This could indicate that the data is relatively stable over time without any significant upward or downward trends.
# 
# In the context of UFO sightings, it suggests that the frequency of sightings does not show a consistent increase or decrease over time within the analyzed period. This could have various interpretations, such as stable reporting patterns, no underlying changes in UFO activity, or limitations in the data collection process. 
# 
# The Seasonal and Residual plots indicate that the data does not exhibit any consistent periodic fluctuations or irregularities beyond what is captured by the trend component.
# 
# Possible interpretations of this finding could include:
# 
# - Lack of Seasonality: The data does not exhibit any regular seasonal patterns or cycles that repeat over time. This could imply that UFO sightings occur uniformly across different time periods without any specific seasonal trends.
# 
# - Stationarity: The data may be stationary, meaning that it does not exhibit any systematic changes or trends over time. In a stationary time series, the mean, variance, and autocorrelation structure remain constant over time.
# 
# - Data Limitations: The absence of discernible seasonality or residual patterns could also be attributed to limitations in the data collection process or the specific time period being analyzed. It's possible that the data does contain seasonality or residual patterns, but they are not captured effectively in the analysis.
# 
# Because UFO sightings are rare and seemingly sporadic, it makes sense that there is no "UFO season" where sightings typically occur.
# 
# The ACF plot shows a gradual decay and the PACF plot shows a sharp drop-off after a few lags, meaning that an autoregressive model may be suitable.

# In[16]:


import pandas as pd

# Assuming your DataFrame is named ufo_data_usa
# Convert 'datetime' column to datetime if it's not already in datetime format
ufo_data_usa['datetime'] = pd.to_datetime(ufo_data_usa['datetime'])

# Sort DataFrame by 'city' and 'datetime'
ufo_data_usa_sorted = ufo_data_usa.sort_values(by=['State', 'datetime'])

# Group DataFrame by 'city'
grouped_by_state = ufo_data_usa_sorted.groupby('State')

# Calculate time difference between consecutive sightings for each city
time_between_sightings = grouped_by_state['datetime'].diff()

# Convert timedelta to total days
time_between_sightings_seconds = time_between_sightings.dt.days

# Calculate average and standard deviation of time differences for each city
average_time_between_sightings = time_between_sightings_seconds.groupby(ufo_data_usa_sorted['State']).mean()
std_dev_time_between_sightings = time_between_sightings_seconds.groupby(ufo_data_usa_sorted['State']).std()

# Create a new DataFrame to store the calculated information
time_between_sightings_df = pd.DataFrame({
    'Average Time Between Sightings': average_time_between_sightings,
    'Standard Deviation of Time Between Sightings': std_dev_time_between_sightings
})

# Reset index to make 'city' a column instead of index
time_between_sightings_df.reset_index(inplace=True)

# Display the new DataFrame
print(time_between_sightings_df.head())
print()

# Print average time between sightings
print('Average time between sightings:', time_between_sightings_df['Average Time Between Sightings'].mean(),'days')


# In[20]:


# Find the index of the minimum value in the column 'Average Time Between Sightings'
min_avg_index = time_between_sightings_df['Average Time Between Sightings'].idxmin()

# Retrieve the corresponding state using the index
shortest_avg_time_state = time_between_sightings_df.loc[min_avg_index, 'State']

# Extract the corresponding information from time_between_sightings_df
shortest_avg_time_info = time_between_sightings_df.loc[min_avg_index]

print("State with the smallest average time between sightings:", shortest_avg_time_state)
print("Average Time Between Sightings (Days):", shortest_avg_time_info['Average Time Between Sightings'])
print()

# Find the index of the maximum value in the column 'Average Time Between Sightings'
max_avg_index = time_between_sightings_df[time_between_sightings_df['State'] != 'DC']['Average Time Between Sightings'].idxmax()

# Retrieve the corresponding state using the index
longest_avg_time_state = time_between_sightings_df.loc[max_avg_index, 'State']

# Extract the corresponding information from time_between_sightings_df
longest_avg_time_info = time_between_sightings_df.loc[max_avg_index]

print("State with the longest average time between sightings:", longest_avg_time_state)
print("Average Time Between Sightings (Days):", longest_avg_time_info['Average Time Between Sightings'])


# In[31]:


# Calculate time difference between consecutive sightings in days
ufo_data_usa['time_diff_days'] = ufo_data_usa['datetime'].diff().dt.days.fillna(0)

# Plotting the difference between consecutive sightings in days
plt.figure(figsize=(10, 6))
plt.plot(ufo_data_usa['datetime'], ufo_data_usa['time_diff_days'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Time Difference (days)')
plt.title('Time Difference Between Consecutive Sightings (Days)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[14]:


# Convert the datetime column to datetime format if it's not already
ufo_data_usa['datetime'] = pd.to_datetime(ufo_data_usa['datetime'])

# Group the data by datetime and count the number of sightings for each time period
sightings_over_time = ufo_data_usa.groupby(pd.Grouper(key='datetime', freq='M')).size()

# Plot the time series
plt.figure(figsize=(12, 6))
sightings_over_time.plot()
plt.title('Number of UFO Sightings Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Sightings')
plt.show()


# In[19]:


# Group the data by datetime and count the number of sightings for each time period
sightings = ufo_data_usa.groupby(pd.Grouper(key='datetime', freq='Y')).size().reset_index(name='sightings')

sightings.head()


# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dt
from pybats.analysis import analysis
from pybats.point_forecast import median

# Setting the Date as the index
sightings.set_index('datetime', inplace=True)

sightings_counts = sightings['sightings'].values


# In[29]:


forecast_horizon = 1 # predict one step ahead
start_forecast = 0 # begin forecast at time step 0
end_forecast = len(sightings) - 1 # conclude forecast a year after end of our data
model, samples_drawn = analysis(
    sightings_counts,
    family="poisson", # the family of the distribution to be employed
    forecast_start=start_forecast,
    forecast_end=end_forecast,
    k=forecast_horizon,
    nsamps=100, # number of samples drawn for each month
    prior_length=6, # number of data points defining the prior distribution
    rho=.9, # random effect extension
    deltrend=0.5, # discount factor for the trend component
    delregn=0.9 # discount factor for the regression component
)
predicted_forecast = median(samples_drawn)


# In[30]:


from pybats.plot import plot_data_forecast
from pybats.point_forecast import median
from pybats.loss_functions import MAPE

import matplotlib.pyplot as plt
import pandas as pd

# Visualization
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax = plot_data_forecast(fig, ax, sightings_counts, predicted_forecast, samples_drawn,
                        dates=sightings.index)
ax = ax_style(ax, ylabel='Sightings', xlabel='Time',
              legend=['Forecast', 'Sightings', 'Credible Interval'])


# Pretty good, but after 2000 it starts to lose accuracy

# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Decomposing the time series into trend, seasonal and residual components
decomposition = sm.tsa.seasonal_decompose(sightings['sightings'], model='additive')

# Plotting the decomposition
plt.figure(figsize=(14, 7))
decomposition.plot()
plt.show()


# In[34]:


# Rolling - Calculating a 7 and 30-day moving average for the sightings data
sightings['7-Day MA'] = sightings['sightings'].rolling(window=7).mean()
# Rolling - Calculating a 7 and 30-day moving average for the sightings data
sightings['30-Day MA'] = sightings['sightings'].rolling(window=30).mean()


# Plotting the original sales data with the 7-day moving average
plt.figure(figsize=(14, 7))
plt.plot(sightings['sightings'], label='Original')
plt.plot(sightings['7-Day MA'], color='red', label='7-Day Moving Average')
plt.plot(sightings['30-Day MA'], color='black', label='30-Day Moving Average')
plt.title('Sightings and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Sightings')
plt.legend()
plt.show()


# In[35]:


from statsmodels.tsa.stattools import adfuller

# Conducting the Augmented Dickey-Fuller test to check for stationarity
adf_test = adfuller(sightings['sightings'])

# Outputting the results
adf_output = pd.Series(adf_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in adf_test[4].items():
    adf_output[f'Critical Value ({key})'] = value

adf_output


# - The test statistic is lower than the critical values at all significance levels.
# - The p-value is very small (close to zero), indicating strong evidence against the null hypothesis.
# 
# Therefore, we reject the null hypothesis of non-stationarity and conclude that the time series is likely stationary.

# In[39]:


import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# ARIMA model fitting

# Determine the end date of your historical data
end_date = sightings.index[-1]

# Define the number of days you want to forecast into the future
forecast_days = 31  # Example: forecast for the next 31 days

# Generate future date range starting from tomorrow
start_date_future = datetime.now() + timedelta(days=1)
end_date_future = start_date_future + timedelta(days=forecast_days - 1)  # Adjusted to include the specified number of forecast days
future_dates = pd.date_range(start=start_date_future, end=end_date_future, freq='D')

# ARIMA model fitting

# Choosing ARIMA parameters (p, d, q)
# p: the number of lag observations included in the model (lag order)
# d: the number of times that the raw observations are differenced (degree of differencing)
# q: the size of the moving average window (order of moving average)

p = 0  # example value
d = 0  # since we've differenced the series once
q = 0  # example value

# Fitting the ARIMA model
model = ARIMA(sightings['sightings'], order=(p, d, q))
model_fit = model.fit()

# Forecasting
forecast = model_fit.forecast(steps=forecast_days)

# Converting forecast to a Series with future dates
forecast_series = pd.Series(forecast, index=future_dates)

# Plotting the forecast
plt.figure(figsize=(14, 7))
plt.plot(sightings['sightings'], label='Historical Sightings')
plt.plot(forecast_series, color='red', label='Forecasted Sightings')
plt.title('ARIMA Model Forecast')
plt.xlabel('Date')
plt.ylabel('Sightings')
plt.legend()
plt.show()

forecast_series


# In[62]:


# Determine the split point
split_point = int(0.8 * len(sightings))

# Split the DataFrame into training and testing sets
train_data = sightings.iloc[:split_point]
test_data = sightings.iloc[split_point:]

# SARIMA model fitting

# Define the order and seasonal_order parameters
order = (2, 2, 1)  # (p, d, q)
seasonal_order = (1, 0, 1, 60)  # (P, D, Q, S)

# Fitting the SARIMA model using training data
model = SARIMAX(train_data['sightings'], order=order, seasonal_order=seasonal_order)
model_fit = model.fit()

# Forecasting
forecast = model_fit.forecast(steps=len(test_data))

# Converting forecast to a Series with future dates
forecast_series = pd.Series(forecast, index=test_data.index)

# Plotting the forecast
plt.figure(figsize=(14, 7))
plt.plot(train_data['sightings'], label='Historical Sightings (Training)')
plt.plot(test_data['sightings'], label='Historical Sightings (Testing)')
plt.plot(forecast_series, color='red', label='Forecasted Sightings')
plt.title('SARIMA Model Forecast')
plt.xlabel('Date')
plt.ylabel('Sightings')
plt.legend()
plt.show()

# Evaluate the forecast against the testing set
mse = np.mean((forecast - test_data['sightings']) ** 2)
mae = np.mean(np.abs(forecast - test_data['sightings']))
print("Mean Squared Error (MSE):", mse)
print('Mean Absolute Error (MAE):', mae)


# In[63]:


# Calculate MAPE
def calculate_mape(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100

# Calculate RMSLE
def calculate_rmsle(actual, forecast):
    return np.sqrt(np.mean(np.square(np.log1p(actual) - np.log1p(forecast))))

# Assuming 'actual' is the actual values and 'forecast' is the forecasted values
mape = calculate_mape(test_data['sightings'], forecast_series)
rmsle = calculate_rmsle(test_data['sightings'], forecast_series)

print("Mean Absolute Percentage Error (MAPE):", mape)
print("Root Mean Squared Logarithmic Error (RMSLE):", rmsle)


# The Mean Absolute Percentage Error (MAPE) of approximately 25.95% indicates that, on average, the model's predictions deviate from the actual values by about 25.95% in terms of percentage. This means that the model's predictions, on average, are off by around 25.95% compared to the actual observed values.
# 
# The Root Mean Squared Logarithmic Error (RMSLE) of approximately 0.3306 measures the average difference between the logarithm of the predicted values and the logarithm of the actual values. It is a relative measure of accuracy and considers the ratio of predicted and actual values rather than their absolute difference.
# 
# In summary, a MAPE of 25.95% suggests that the model's predictions have a moderate level of accuracy, with some room for improvement. Similarly, an RMSLE of approximately 0.3306 indicates that the model's predictions have a reasonable level of accuracy, but there may still be room for refinement.

# In[69]:


# Forecast for 1 year
years = 1
pred_uc_1y = model_fit.get_forecast(steps=12 * years)
pred_ci_1y = pred_uc_1y.conf_int()

ax = sightings['sightings'].plot(label='Observed', figsize=(14, 7))
pred_uc_1y.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_1y.index,
                pred_ci_1y.iloc[:, 0],
                pred_ci_1y.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Sightings')
plt.legend()
# plt.savefig(current_dir + os.sep + 'SARIMA_FORECAST_1y.png')
plt.show()
plt.close()

# Forecast for 5 years
years = 5
pred_uc_5y = model_fit.get_forecast(steps=12 * years)
pred_ci_5y = pred_uc_5y.conf_int()

ax = sightings['sightings'].plot(label='Observed', figsize=(14, 7))
pred_uc_5y.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci_5y.index,
                pred_ci_5y.iloc[:, 0],
                pred_ci_5y.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Sightings')
plt.legend()
# plt.savefig(current_dir + os.sep + 'SARIMA_FORECAST_5y.png')
plt.show()
plt.close()


# Obviously, as the years go on, it's harder for the model to be accurate.

# In[82]:


from prophet import Prophet

# Rename the index to 'ds' and the 'sightings' column to 'y'
sightings_prophet = sightings.rename_axis('ds').reset_index()
sightings_prophet = sightings_prophet.rename(columns={'sightings': 'y'})

# Drop the last two columns
sightings_prophet = sightings_prophet.drop(columns=['7-Day MA', '30-Day MA'])


# Convert 'ds' column to datetime format
sightings_prophet['ds'] = pd.to_datetime(sightings_prophet['ds'])


print('Training a prophet...')
m = Prophet()  # Create a Prophet model instance
m.fit(sightings_prophet)  # Fit the model to the sightings data

# Forecast 1 year
years = 1
future = m.make_future_dataframe(periods=365 * years)
future.tail()

# Plot the forecast
fig1 = m.plot(forecast)
plt.show()  # Display the plot

# Plot forecast components
fig2 = m.plot_components(forecast)
plt.show()  # Display the plot


# In[84]:


# Drop the last two columns
sightings_prophet = sightings_prophet.drop(columns=['7-Day MA', '30-Day MA'])

sightings_prophet.head()


# In[85]:


# set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet(interval_width=0.95)

my_model.fit(sightings_prophet)


# In[86]:


future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
future_dates.tail()


# In[87]:


forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[88]:


my_model.plot(forecast,
              uncertainty=True)


# In[89]:


my_model.plot_components(forecast)


# In[95]:


# Group the data by datetime and count the number of sightings for each time period
sightings_daily = ufo_data_usa.groupby(pd.Grouper(key='datetime', freq='D')).size().reset_index(name='sightings')

sightings_daily = sightings_daily.rename(columns={'datetime': 'ds'})
sightings_daily = sightings_daily.rename(columns={'sightings': 'y'})

# Drop rows where sightings are zero
sightings_daily = sightings_daily[sightings_daily['y'] != 0]


sightings_daily.head()


# In[96]:


# set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet(interval_width=0.95)

my_model.fit(sightings_daily)


future_dates = my_model.make_future_dataframe(periods=60, freq='D')
future_dates.tail()


# In[97]:


forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[98]:


my_model.plot(forecast,
              uncertainty=True)


# In[101]:


#%% Forecast 1 year
years = 1
future = my_model.make_future_dataframe(periods=365*years)
future.tail()

fig2 = my_model.plot_components(forecast)
# plt.savefig(current_dir + os.sep + 'PROPHET_COMPONENTS.png')
plt.show()


# In[102]:


years = 5
future = my_model.make_future_dataframe(periods=365*years)
future.tail()

forecast = my_model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
# plt.savefig(current_dir + os.sep + 'PROPHET_FORECAST_5y.png')
plt.show()


# Altough each of the models above were able to produce some results, ultimately they weren't able to make any predictions into the actual future. Although dates 1-2 years beyond the data are technically in the future of the sightings, the data ends about 10 years ago in 2014. As such, predicting that the next sighting will happen sometime after 5-2014 and 1-2015 isn't actually a prediction.
# 
# ## Going forward, the next model is the one that was actually able to make predictions into the real future.

# In[21]:


# Group the data by datetime and count the number of sightings for each time period
sightings = ufo_data_usa.groupby(pd.Grouper(key='datetime', freq='M')).size().reset_index(name='sightings')

sightings.head()


# In[22]:


# Find the highest amount of sightings in a month
max_avg = sightings['sightings'].idxmax()

# Retrieve the corresponding date using the index
max_avg_date = sightings.loc[max_avg, 'datetime']

# Display results
print("Highest amount of  sightings in a month:", max_avg)
print("Month/Year:", max_avg_date)
print()

# Find the average amount of sightings in a month
avg = sightings['sightings'].mean()

# Display results
print("Average amount of  sightings in a month:", avg)
print()

# Find the lowest amount of sightings in a month
min_avg = sightings['sightings'].idxmin()

# Retrieve the corresponding date using the index
min_avg_date = sightings.loc[min_avg, 'datetime']

# Display results
print("Lowest amount of  sightings in a month:", min_avg)
print("Month/Year:", min_avg_date)


# In[23]:


# Elbow plot for clusters
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Rescale the data
scaler = StandardScaler()
sightings['sightings_scaled'] = scaler.fit_transform(sightings[['sightings']])

# Extract the scaled sightings count data
X = sightings[['sightings_scaled']].values

# Initialize an empty list to store the within-cluster sum of squares (inertia) for different values of k
wcss = []

# Define the range of k values to test
k_values = range(1, 11)  # You can adjust the range as needed

# Calculate the within-cluster sum of squares (inertia) for different values of k
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow plot
plt.plot(k_values, wcss, marker='o')
plt.title('Elbow Plot')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(np.arange(1, 11, 1))
plt.show()


# In[24]:


# Perform K-means clustering
kmeans = KMeans(n_clusters=3)  # Based on elbow plot
sightings['cluster'] = kmeans.fit_predict(sightings[['sightings_scaled']])

# Optional: Inverse transform the scaled values if needed
# monthly_sightings_df['sightings_count'] = scaler.inverse_transform(monthly_sightings_df[['sightings_count_scaled']])

# Display the DataFrame with the cluster assignments
print(sightings)


# In[25]:


# Plot the clusters
plt.figure(figsize=(10, 6))

k = 3 # number of clusters

for i in range(k):
    cluster_data = sightings[sightings['cluster'] == i]
    plt.scatter(cluster_data.index, cluster_data['sightings_scaled'], label=f'Cluster {i+1}')

plt.title('K-means Clustering of Monthly Sightings')
plt.xlabel('Month')
plt.ylabel('Scaled Sightings Count')
plt.legend()
plt.show()


# The clusters seem to follow the slow, boring period of 1930 ~ 1994, then the spike from 1995 to 2008, and then the huge boom from 2009 to 2014.

# In[26]:


from datetime import timedelta

# Choose the cluster for which you want to predict the next sighting
target_cluster = 2  

# Filter the data for sightings in the target cluster
cluster_data = sightings[sightings['cluster'] == target_cluster]

# Assuming the data is sorted by datetime, you can get the latest sighting date
latest_sighting_date = cluster_data['datetime'].max()

# Calculate the average time between sightings within this cluster
average_time_between_sightings = cluster_data['datetime'].diff().mean()

# Predict the next sighting date
predicted_next_sighting_date = latest_sighting_date + average_time_between_sightings

print("Predicted next sighting date:", predicted_next_sighting_date)
print()
print('Last recorded sighting in data frame:')
print(ufo_data_usa.iloc[-1, :3])


# In[28]:


from datetime import datetime, timedelta

# Assuming today's date is the current date
current_date = datetime.now()

# Predict the next sighting date starting from today
predicted_next_sighting_date = current_date + average_time_between_sightings

print("Predicted next sighting date (starting from today):", predicted_next_sighting_date)


# In[38]:


# Count the occurrences of each city in the original ufo_data_usa DataFrame
city_counts = ufo_data_usa['city'].value_counts()

# Find the most common city
most_common_city = city_counts.idxmax()

# Extract the corresponding State and County from the original data
most_common_state = ufo_data_usa.loc[ufo_data_usa['city'] == most_common_city, 'State'].iloc[0]
most_common_county = ufo_data_usa.loc[ufo_data_usa['city'] == most_common_city, 'County Name'].iloc[0]

# Get the total number of sightings for the most common city
total_sightings = city_counts[most_common_city]

# Print the most common city, corresponding state, county, and total number of sightings
print("Most common City:", most_common_city)
print("Corresponding State:", most_common_state)
print("Corresponding County:", most_common_county)
print("Total number of sightings:", total_sightings)


# In[43]:


# Group the data by datetime and FIPS code and count the number of sightings for each time period and FIPS code
sightings_by_coordinates = ufo_data_usa.groupby([pd.Grouper(key='datetime', freq='M'), 'latitude', 'longitude']).size().reset_index(name='sightings')

sightings_by_coordinates.head()


# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Prepare features and target variables
X = sightings_by_coordinates[['latitude', 'longitude']]  # Features
y_lat = sightings_by_coordinates['latitude']  # Target variable (latitude)
y_lon = sightings_by_coordinates['longitude']  # Target variable (longitude)

# Split the data into training and testing sets
X_train, X_test, y_lat_train, y_lat_test, y_lon_train, y_lon_test = train_test_split(X, y_lat, y_lon, test_size=0.2, random_state=42)

# Train linear regression models for latitude and longitude separately
reg_lat = LinearRegression()
reg_lat.fit(X_train, y_lat_train)

reg_lon = LinearRegression()
reg_lon.fit(X_train, y_lon_train)

# Extract features for the next sighting from your data
next_sighting_features = sightings_by_coordinates.iloc[-1][['latitude', 'longitude']].values.reshape(1, -1)

# Predict latitude and longitude for the next UFO sighting
next_lat_prediction = reg_lat.predict(next_sighting_features)
next_lon_prediction = reg_lon.predict(next_sighting_features)

print("Predicted latitude for the next sighting:", next_lat_prediction)
print("Predicted longitude for the next sighting:", next_lon_prediction)

# Define a threshold for matching latitudes and longitudes
threshold = 0.1  # Adjust as needed based on your data and model accuracy

# Find matching rows in ufo_data_usa
matching_rows = ufo_data_usa[
    (abs(ufo_data_usa['latitude'] - next_lat_prediction) <= threshold) &
    (abs(ufo_data_usa['longitude'] - next_lon_prediction) <= threshold)
]
# Select just one matching row (the first one, for example)
matching_row = matching_rows.iloc[0]

# Print the matching city and state
matching_city = matching_row['city']
matching_state = matching_row['state']

print("Location:", matching_city, matching_state)


# Evaluate the model
y_lat_pred = reg_lat.predict(X_test)
y_lon_pred = reg_lon.predict(X_test)

mse_lat = mean_squared_error(y_lat_test, y_lat_pred)
mse_lon = mean_squared_error(y_lon_test, y_lon_pred)

print("Mean Squared Error (Latitude):", mse_lat)
print("Mean Squared Error (Longitude):", mse_lon)


# In[45]:


import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd

# Load the UFO sightings data and create a GeoDataFrame
geometry = [Point(xy) for xy in zip(ufo_data['longitude'], ufo_data['latitude'])]
gdf = gpd.GeoDataFrame(ufo_data, geometry=geometry)

# Load the state boundaries Shapefile
states_shapefile = gpd.read_file("./ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp")

# Create a GeoDataFrame for the predicted location
predicted_location = gpd.GeoDataFrame({'City': ['Ellensburg, WA'],
                                        'geometry': [Point(-120.5466667, 46.9966667)]})

# Plot the state boundaries and UFO sightings on the same map
fig, ax = plt.subplots(figsize=(12, 10))

# Plot state boundaries
states_shapefile.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5)

# Plot the UFO sightings
gdf.plot(ax=ax, marker='o', column='state', cmap='tab20', markersize=1, alpha=0.6, label='Reported UFO Sightings')

# Plot the predicted location with a star marker
predicted_location.plot(ax=ax, marker='*', color='red', markersize=100, label='Predicted Location')

# Set the limits to define the area of interest (United States)
xlim = (-130, -65)  # Longitude limits for the US
ylim = (25, 50)     # Latitude limits for the US
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_title('Geographical Distribution of UFO Sightings in the United States 1930-2014')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend()

plt.show()


# ## NLP Investigation

# In[48]:


from wordcloud import WordCloud
from textblob import TextBlob

ufo_data_usa['comments'] = ufo_data_usa['comments'].str.replace('[^a-zA-Z\s]', '')
comments = ufo_data_usa['comments']
sample = comments.iloc[1:150]
# Step 2: Preprocess the text data
# You can add more preprocessing steps if needed
text = ' '.join(sample)  # Combine all strings into one

# Step 3: Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Step 4: Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[49]:


ufo_data_usa['shape'] = ufo_data_usa['shape'].str.replace('[^a-zA-Z\s]', '')
shape = ufo_data_usa['shape']
sample = comments.iloc[1:150]
# Step 2: Preprocess the text data
# You can add more preprocessing steps if needed
text = ' '.join(sample)  # Combine all strings into one

# Step 3: Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Step 4: Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[50]:


ufo_data_usa['shape'] = ufo_data_usa['shape'].str.replace('[^a-zA-Z\s]', '')
shape = ufo_data_usa['shape']
sample = comments.iloc[1:1000]
# Step 2: Preprocess the text data
# You can add more preprocessing steps if needed
text = ' '.join(sample)  # Combine all strings into one

# Step 3: Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Step 4: Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[51]:


# Perform sentiment analysis on each string in the 'text_column'
sentiment_scores = ufo_data_usa['comments'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Assign sentiment labels based on the sentiment scores
sentiment_labels = sentiment_scores.apply(lambda score: 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral')

# Add sentiment scores and labels to the DataFrame
ufo_data_usa['sentiment_score'] = sentiment_scores
ufo_data_usa['sentiment_label'] = sentiment_labels

# Display the DataFrame with sentiment scores and labels
ufo_data_usa.head()


# In[52]:


# Count the frequency of each sentiment label
sentiment_counts = ufo_data_usa['sentiment_label'].value_counts()

# Plot the sentiment labels
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Label')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# ## Sightings x Airports

# In[53]:


# Load the state boundaries Shapefile
airports_shp = gpd.read_file("./USA_Airports/USA_Airports.shp")

# Display the first few rows of the dataframe
airports_shp.head()


# In[61]:


# Getting latitude and longitude coordinates
airports_shp['lon'] = airports_shp.geometry.x
airports_shp['lat'] = airports_shp.geometry.y

airports_shp.head()


# In[63]:


import plotly.graph_objects as go

# create a new Plotly figure
fig = go.Figure()

# Adding the scatter_geo plot for airports
fig.add_trace(
    go.Scattergeo(
        lon = airports_shp['lon'],
        lat = airports_shp['lat'],
        text = airports_shp['FAA_ID'],
        mode = 'markers+text',
        marker = dict(size=5, color='blue', line=dict(width=1, color='black')),
        textposition='top center'
    )
)

# Update the layout to add state boundaries and tweak map appearance
fig.update_geos(
    visible=True, # turn off the base map features
    resolution=50, # Set resolution to 50 for state boundaries
    showcountries=True, countrycolor="RebeccaPurple",
    showsubunits=True, subunitcolor="Gray"
)

# set the map scope to usa
fig.update_layout(
    title=dict(text="Airports in the USA", xanchor='center', x=0.5),
    geo_scope='usa', # limit map scope to USA
)

# Show figure
fig.show()


# In[64]:


# Create a scatter_geo plot for the UFO sightings
fig.add_trace(
    go.Scattergeo(
        lon = ufo_data_usa['longitude'],
        lat = ufo_data_usa['latitude'],
        mode = 'markers',
        marker = dict(size=4, color='yellow', line=dict(width=0)),
        name='UFO Sightings'
    )
)

# Show figure
fig.show()


# In[65]:


# Adjust the size of the map and the marker sizes 
fig.update_layout(
    title=dict(text="Airports and UFO Sightings in the USA", xanchor='center', x=0.5),
    geo_scope='usa', # limit map scope to USA
    width=1000,  # Width of the map
    height=600,  # Height of the map
    margin={"r":0,"t":0,"l":0,"b":0}  # Reducing the white margins
)

# Adjust the UFO sightings markers
fig.data[1].marker.size = 2

# Show figure
fig.show()


# In[66]:


# Update map background 
fig.update_geos(
    landcolor='lightgray',
)

# Bring airport markers to the front 
fig.data = [fig.data[1], fig.data[0]]  # switches the layered order of UFO sightings and airports

# Update the marker size and label for airports
fig.data[1].marker.size = 7
fig.data[1].name = 'Airports'  

# Adjust the legend 
fig.update_layout(
    legend=dict(
        title='',  # Remove the legend title
        traceorder='normal',
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
    )
)

# Show the updated figure
fig.show()


# In[67]:


from scipy.spatial.distance import cdist
import numpy as np

# Function to calculate haversine distance between two sets of (lon, lat) arrays
def haversine_distances(airports, ufos):
    """
    airports: array-like of shape (n_airports, 2) - (longitude, latitude) for airports
    ufos: array-like of shape (n_ufos, 2) - (longitude, latitude) for ufo sightings
    """
    # Earth radius in miles
    R = 3959.87433

    # Convert latitude and longitude from degrees to radians
    m_airports = np.radians(airports)
    m_ufos = np.radians(ufos)

    # Haversine distance formula
    dlon = m_ufos[:, np.newaxis, 0] - m_airports[np.newaxis, :, 0]
    dlat = m_ufos[:, np.newaxis, 1] - m_airports[np.newaxis, :, 1]
    a = np.sin(dlat / 2.0)**2 + np.cos(m_airports[np.newaxis, :, 1]) * np.cos(m_ufos[:, np.newaxis, 1]) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    d = R * c

    return d

# Extract coordinates for airports and ufo sightings as (lon, lat)
airport_coords = airports_shp[['lon', 'lat']].to_numpy()
ufo_coords = ufo_data_usa[['longitude', 'latitude']].to_numpy()

# Calculate the haversine distances from each UFO sighting to each airport
distances = haversine_distances(airport_coords, ufo_coords)

# Get the minimum distance to the nearest airport for each UFO sighting
min_distances = np.min(distances, axis=1)

# Let's add this minimum distance data to our ufo dataframe
ufo_data_usa['min_distance_to_airport'] = min_distances

# Check the first few entries to verify
ufo_data_usa[['datetime', 'city', 'state', 'min_distance_to_airport']].head()


# In[70]:


# Constants
total_us_area = 3119885  # The approximate area of the contiguous US in square miles
radius = 10  # 10-mile radius
area_per_airport = np.pi * (radius ** 2)  # Area within a 10-mile radius of an airport

# Number of sightings within 10 miles of an airport
num_sightings_near_airport = (ufo_data_usa['min_distance_to_airport'] <= 10).sum()

# Total number of sightings
total_num_sightings = len(ufo_data_usa)

# using the count of airports to estimate the combined area of influence
total_airport_influence_area = area_per_airport * len(airports_shp)

# Expected number of sightings within 10 miles of an airport if uniformly distributed
expected_num_sightings_near_airport = (total_airport_influence_area / total_us_area) * total_num_sightings

# Times more likely to see a UFO within 10 miles of an airport
times_more_likely = num_sightings_near_airport / expected_num_sightings_near_airport

# Calculate the percentage more likely
percentage_more_likely = (num_sightings_near_airport - expected_num_sightings_near_airport) / expected_num_sightings_near_airport * 100

num_sightings_near_airport, expected_num_sightings_near_airport, times_more_likely, percentage_more_likely


# Based on the analysis:
# 
# There are 13,577 UFO sightings within 10 miles of an airport. If UFO sightings were uniformly distributed across the contiguous United States, we would expect approximately 584 sightings within 10 miles of an airport, given the total number of sightings and the combined area of influence of all airports. UFO sightings are therefore about 23.26 times more likely to be reported within 10 miles of an airport than if they were uniformly distributed. This is equivalent to a 2226% higher likelihood of a sighting near an airport compared to a random location in the US.
# 
# We can include a statement like this in our analysis:
# 
# "A UFO sighting is approximately 23 times more likely to occur within 10 miles of a major airport, or there is a 2226% higher likelihood of a sighting in these areas compared to the overall region."

# In[ ]:


# Load the missing persons report data
missing_persons_path = 'C:/Users/benjh/Desktop/School/IST 718/UFO Project/Data/Missing_Persons_clean.csv'

# Try reading the CSV file with 'ISO-8859-1' encoding
try:
    missing_persons_df = pd.read_csv(missing_persons_path, encoding='ISO-8859-1')
    # Display the first few rows of the dataframe to understand its structure
    display(missing_persons_df.head())
except Exception as e:
    print(f"An error occurred: {e}")


# In[ ]:


# plot all the data, but for very large datasets
missing_persons_df['latitude'] = missing_persons_df['Lat']
missing_persons_df['longitude'] = missing_persons_df['Long']

# Add the missing persons data to the map
fig.add_trace(
    go.Scattergeo(
        lon = missing_persons_df['longitude'],
        lat = missing_persons_df['latitude'],
        mode = 'markers',
        marker = dict(size=2, color='red', line=dict(width=0)),
        name='Missing Persons'
    )
)

# Update layout to make sure everything is still visible
fig.update_layout(
    margin={"r":0,"t":0,"l":0,"b":0}  # Reduce white space margins
)

# Update the map to bring airports (now trace 2) to the front
fig.data = [fig.data[0], fig.data[2], fig.data[1]]  # UFO sightings, Missing Persons, Airports

# Show figure
fig.show()


# In[ ]:


from geopandas import GeoDataFrame, points_from_xy

missing_persons_df = missing_persons_df.dropna(subset=['Lat', 'Long'])

# Convert the missing persons DataFrame to a GeoDataFrame
gdf_missing_persons = GeoDataFrame(missing_persons_df, 
                                   geometry=points_from_xy(missing_persons_df['Long'], missing_persons_df['Lat']))

# Set the CRS for missing persons GeoDataFrame to WGS84 (to match the airports GeoDataFrame)
gdf_missing_persons.crs = airports_geojson.crs

# Spatial join the missing persons points with the US boundaries (assuming airports_geojson has the US boundaries)
gdf_missing_persons_us = gpd.sjoin(gdf_missing_persons, airports_geojson, how="inner", op='intersects')

# Now gdf_missing_persons_us contains only the missing persons within the US boundaries
# We'll update the map using this cleaned GeoDataFrame

# Create a scatter_geo plot for the cleaned missing persons data
fig.add_trace(
    go.Scattergeo(
        lon = gdf_missing_persons_us['Long'],
        lat = gdf_missing_persons_us['Lat'],
        mode = 'markers',
        marker = dict(size=2, color='red', line=dict(width=0)),
        name='Missing Persons (US only)'
    )
)

# Remove the previous missing persons layer (which had missing persons outside the US)
fig.data = [fig.data[0], fig.data[2], fig.data[1], fig.data[3]]  # UFO sightings, Missing Persons (US only), Airports

# Show the updated figure
fig.show()

