#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all the required libraries
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
from scipy import stats
get_ipython().system('pip install nltk')
get_ipython().system('pip install textblob')
import re


# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import geopandas as gpd

import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download required resources
nltk.download('stopwords')
nltk.download('vader_lexicon')


# ## Data Integration

# In[2]:


# The type of data integration used is horizontal integration
# Read the three datasets
df1 = pd.read_csv('names.csv')
df2 = pd.read_csv('hospitalisationdetails.csv')
df3 = pd.read_csv('medicalexaminations.csv')

# Merge the first two datasets on customer id
merged_df = pd.merge(df1, df2, on='Customer ID', how='inner')

# Merge the third dataset with the merged DataFrame on customer id
final_df = pd.merge(merged_df, df3, on='Customer ID', how='inner')

# Print the final DataFrame with only the rows having customer id available in all three datasets
print(final_df)

final_df.info()

# This code exemplifies the process of data integration by combining multiple datasets based on common identifiers. 
# The merging operation helps bring together related information from different sources into a single dataset, providing a better view of the data


# In[27]:


# printing out all the column names
print(final_df.columns.values)


# In[4]:


#to show the number of rows and columns
final_df.shape


# ## b) Data Quality Assessment

# In[5]:


# Profiling the data
# Get basic information about the dataset
print(final_df.head(10))  # Preview the first 10 rows
print(final_df.info())  # Summary of columns, data types, and non-null values
print(final_df.describe())  # Summary statistics
final_df.tail(10) #display the last 10 data

# Profile numerical columns
numerical_columns = ['charges', 'BMI']
numerical_profile = final_df[numerical_columns].describe()

# Profile categorical columns
categorical_columns = ['State ID', 'Hospital tier']
categorical_profile = final_df[categorical_columns].value_counts()

print(numerical_profile)
print(categorical_profile)


# In[6]:


# Parse and standardize text data

# Remove spaces in columns and replace them with underscore
final_df.columns = final_df.columns.str.replace(" ", "_")

# Set the name of the columns to all lowercase
final_df.columns = map(str.lower, final_df.columns)

final_df.rename(columns={'date': 'day'}, inplace=True)


# Rename multiple columns
final_df.rename(columns={'smoker': 'smoker_status', 'hba1c': 'hba1c_level', 'numberofmajorsurgeries':'no_of_major_surgeries', 'children': 'no_of_children'}, 
                inplace=True)


# Print the DataFrame with the renamed column
print(final_df)

# Handling inconsistent or erroneous data
final_df['hospital_tier'] = final_df['hospital_tier'].replace('tier - 1', 'Tier 1')  # Replacing inconsistent tier labels
final_df['hospital_tier'] = final_df['hospital_tier'].replace('tier - 2', 'Tier 2')
final_df['hospital_tier'] = final_df['hospital_tier'].replace('tier - 3', 'Tier 3')
final_df['city_tier'] = final_df['hospital_tier'].replace('tier - 1', 'Tier 1')
final_df['city_tier'] = final_df['hospital_tier'].replace('tier - 2', 'Tier 2')
final_df['city_tier'] = final_df['hospital_tier'].replace('tier - 3', 'Tier 3')

# Recheck columns and values
print(final_df.info())
print(final_df.head())


# In[7]:


#Generalized Cleansing

# Replace "?" values to NaN values from the dataframe
final_df.replace("?", pd.NaT, inplace=True)

# Remove the rows with NaN values
final_df.dropna(inplace = True)

# remove "?" values from day, month, year columns
final_df = final_df[final_df['day'] != "?"]
final_df = final_df[final_df['month'] != "?"]
final_df = final_df[final_df['year'] != "?"]

# Handling missing values by filling with appropriate values
final_df['no_of_children'].fillna(0, inplace=True)  # Fill missing children with 0

# Replace 'No' with 0 in the 'no_of_major_surgeries' column
final_df['no_of_major_surgeries'].replace({'No major surgery': 0}, inplace=True)

# Replace 'yes' with 'Yes' for smoker_status column
final_df['smoker_status'].replace({'yes': 'Yes'}, inplace=True)

# Replace 'yes' with 'Yes' in the heart_issues column
final_df['heart_issues'].replace({'yes': 'Yes'}, inplace=True)


# Visualize missing values
sns.heatmap(final_df.isnull(), cbar=False)

final_df


# In[8]:


# Matching


# In[9]:


# Monitoring
# Check for missing values in specific columns
missing_values = final_df[['customer_id', 'charges', 'bmi']].isnull().sum()
print(missing_values)


# ## Problem Resolution

# ### Format Checks & Completeness Checks

# In[10]:


# Format Checks

# Create a new column 'dateofbirth' by combining 'day', 'month', and 'year' columns
final_df['date_of_birth'] = final_df['day'].astype(str) + '-' + final_df['month'].astype(str) + '-' + final_df['year'].astype(str)

# Convert 'dateofbirth' column to date format
final_df['date_of_birth'] = pd.to_datetime(final_df['date_of_birth'], format='%d-%b-%Y')

# Remove unnecessary columns 
final_df = final_df.drop(columns=['month','day', 'state_id'])
final_df.columns

# Print the updated DataFrame with the readable, combined date format
print(final_df)

# Completeness Checks
# Check for missing values in the dataset
missing_values = final_df.isnull().sum()
print("Missing Values:\n", missing_values)


# ### Reasonable Checks & Limit Checks

# In[11]:


# Reasonableness Checks
# Check if values fall within expected ranges
def check_value_ranges(final_df):
    # Check for invalid values in the 'charges' column
    invalid_charges = final_df[(final_df['charges'] < 0)]
    
    # Check for invalid values in the 'bmi' column
    invalid_bmi = final_df[(final_df['bmi'] < 10) | (final_df['bmi'] > 100)]
    
    # Combine the invalid values from both columns
    invalid_values = pd.concat([invalid_charges, invalid_bmi])
    
    return invalid_values
# no invalid values

# Call the function to check value ranges
invalid_values = check_value_ranges(final_df)

# Print the invalid values
print(invalid_values)

# Identify and limit checks for numeric columns
final_df['charges'] = np.where(final_df['charges'] < 0, np.nan, final_df['charges'])
final_df['bmi'] = np.where((final_df['bmi'] < 10) | (final_df['bmi'] > 100), np.nan, final_df['bmi'])

# Review of the data to identify outliers
# Perform statistical outlier detection on the 'charges' column
z_scores = np.abs(stats.zscore(final_df['charges']))
outliers = final_df[z_scores > 3]
print("Outliers:\n", outliers)


# ### Missing value handling & Remove outliers

# In[12]:


# Missing Values
# Impute missing values with mean
def impute_missing_values(final_df):
    data_filled = final_df.fillna(final_df.mean())
    return data_filled

data_filled = impute_missing_values(final_df)
print(data_filled)

# Identify or Remove Outliers
# Remove outliers from the dataset based on domain knowledge or statistical methods
final_df = final_df[(final_df['charges'] >= 0) & (final_df['charges'] <= final_df['charges'].quantile(0.95))]


# ## Data Visualisation For Problem Solution

# In[13]:


# Does our data have a spatial or geographic component?
# Not sure


# In[14]:


# Group the data by the number of children and calculate the total healthcare costs
total_costs_by_children = final_df.groupby('no_of_children')['charges'].sum()

# Create a stacked bar chart
plt.figure(figsize=(8, 6))
total_costs_by_children.plot(kind='bar', stacked=True)
plt.xlabel('Number of Children')
plt.ylabel('Total Healthcare Costs')
plt.title('Healthcare Costs by Number of Children')
plt.show()


# In[15]:


# Does our data have a temporal component, showing change over time?
# No


# In[16]:


# How many variables are we trying to represent? 2 variables
# Create a stacked bar chart
sns.barplot(x='hospital_tier', y='charges', data=final_df, errorbar=None)
plt.xlabel('Hospital Tier')
plt.ylabel('Healthcare Costs')
plt.title('Healthcare Costs by Hospital Tier')
plt.show()


# In[17]:


# Who is the audience we are trying to reach?
# General Audience 
# Create a stacked bar chart to compare healthcare costs by hospital tier and city tier
plt.figure(figsize=(10, 6))
final_df.groupby(['hospital_tier', 'city_tier'])['charges'].mean().unstack().plot(kind='bar', stacked=True)
plt.xlabel('Hospital Tier')
plt.ylabel('Average Healthcare Costs')
plt.title('Average Healthcare Costs by Hospital Tier and City Tier')
plt.legend(title='City Tier', loc='upper right')
plt.show()


# In[18]:


# Academic Audience
# Create a scatter plot with regression line to examine the relationship between BMI and healthcare costs
plt.figure(figsize=(8, 6))
sns.regplot(x='bmi', y='charges', data=final_df, scatter_kws={'alpha':0.5})
plt.xlabel('BMI')
plt.ylabel('Healthcare Costs')
plt.title('Relationship between BMI and Healthcare Costs')
plt.show()
#the higher the BMI, the higher the healthcare cost one have to spend


# In[19]:


# Group the data by year and heart issues, and calculate the count of records
grouped_data = final_df.groupby(['year', 'heart_issues']).size().unstack()

# Create a stacked bar chart
plt.figure(figsize=(10, 6))
grouped_data.plot(kind='bar', stacked=True)
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Heart Issues by Year')
plt.legend(title='Heart Issues')
plt.show()


# In[20]:


# Count the occurrences of cancer history categories
cancer_counts = final_df['cancer_history'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 6))
plt.pie(cancer_counts, labels=cancer_counts.index, autopct='%1.1f%%')
plt.title('Proportion of People With Cancer History')
plt.show()


# In[29]:


# Count the frequency of each year
year_counts = final_df['year'].value_counts().sort_index()

# Create a bar chart
plt.figure(figsize=(8, 6))
year_counts.plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.title('Count of Patients Who Born On Different Years')
plt.show()

# This visualization shows the distribution of data across different years, 
# providing insights into the frequency or counts of people who born on different years.


# In[22]:


# Count the number of customers who smoke
smoker_counts = final_df['smoker_status'].value_counts()
total_smokers = smoker_counts['Yes']

# Create a bar chart to visualize the total number of smokers
plt.figure(figsize=(4, 2))
plt.bar(['Smokers'], [total_smokers])
plt.xlabel('Customer Group')
plt.ylabel('Total Number of Customers')
plt.title('Total Number of Customers: Smokers')
plt.show()


# In[23]:


# Count the number of customers who have heart issues
heart_issues_counts = final_df['heart_issues'].value_counts()
total_heart_issues = heart_issues_counts['Yes']

# Create a bar chart to visualize the total number of customers with heart issues
plt.figure(figsize=(6, 4))
plt.bar(['Heart Issues'], [total_heart_issues])
plt.xlabel('Customer Group')
plt.ylabel('Total Number of Customers')
plt.title('Total Number of Customers: Heart Issues')
plt.show()


# In[24]:


# Filter the DataFrame for customers who both smoke and have heart issues
filtered_df = final_df[(final_df['smoker_status'] == 'Yes') & (final_df['heart_issues'] == 'Yes')]

# Count the total number of customers
total_customers = len(filtered_df)

# Create a bar chart to visualize the total number of customers
plt.figure(figsize=(5, 3))
plt.bar(['Smoker with Heart Issues'], [total_customers])
plt.xlabel('Customer Group')
plt.ylabel('Total Number of Customers')
plt.title('Total Number of Customers that are Smokers with Heart Issues')
plt.show()


# In[25]:


# Group the data by both 'smoker_status' and 'heart_issues', and calculate the average healthcare costs
grouped_costs = final_df.groupby(['smoker_status', 'heart_issues'])['charges'].mean().unstack()

# Create a grouped bar chart
plt.figure(figsize=(10, 6))
grouped_costs.plot(kind='bar')
plt.xlabel('Smoker Status')
plt.ylabel('Average Healthcare Costs')
plt.title('Average Healthcare Costs by Smoker Status and Heart Issues')
plt.legend(title='Heart Issues')
plt.show()


# In[26]:


# Calculate the average healthcare costs by smoker status
smoker_costs = final_df.groupby('smoker_status')['charges'].mean()

# Calculate the average healthcare costs by heart issues
heart_issues_costs = final_df.groupby('heart_issues')['charges'].mean()

# Create individual bar charts for smoker status and heart issues
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
smoker_costs.plot(kind='bar')
plt.xlabel('Smoker Status')
plt.ylabel('Average Healthcare Costs')
plt.title('Average Healthcare Costs by Smoker Status')

plt.subplot(1, 2, 2)
heart_issues_costs.plot(kind='bar')
plt.xlabel('Heart Issues')
plt.ylabel('Average Healthcare Costs')
plt.title('Average Healthcare Costs by Heart Issues')

plt.tight_layout()
plt.show()


# In[ ]:




