# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

# Loading the data
sales_data = pd.read_csv('./Files/AusApparalSales4thQrt2020.csv')

# Inspecting the data
print(f"Sales Head:\n{sales_data.head()}")
print(f"Sales Tail:\n{sales_data.head()}")
print()

# Printing a summary of the data
print(f"Sales data summary:\n{sales_data.info}")
print()

# Checking for missing values
print(f"Count of missing values:\n{sales_data.isna().sum()}")
print(f"Count of null values:\n{sales_data.isnull().sum()}")
print()

# MARKDOWN
# ## Initial Assessment
# * There are no missing or null values on the DataFrme
# 8 The DataFrame has 7560 rows and 6 columns

# Data Wrangling
# Converting the date field to a datetime format
sales_data['Date'] = pd.to_datetime(sales_data['Date'], dayfirst=True)
print(f"Date type column is now {sales_data['Date'].dtype}")
print()

# Checking the time, state and group fields for unique items
print(f"Unique items in Time column: {sales_data['Time'].unique()}")
print(f"Unique items in State column: {sales_data['State'].unique()}")
print(f"Unique items in Group column: {sales_data['Group'].unique()}")
print()

# Removing the leading space from the values
sales_data['Time'] = sales_data['Time'].str.strip()
sales_data['State'] = sales_data['State'].str.strip()
sales_data['Group'] = sales_data['Group'].str.strip()

# Checking the time, state and group fields for unique items
print(f"Removed the leading space from the values:")
print(f"Unique items in Time column: {sales_data['Time'].unique()}")
print(f"Unique items in State column: {sales_data['State'].unique()}")
print(f"Unique items in Group column: {sales_data['Group'].unique()}")
print()

# Descriptive Statistical Analysis
print(f"Descriptive Unite Statistics:")
print(f"{sales_data.describe()}")
print()

# Plotting histograms from Unit and Sales
plt.figure(figsize=(10,5)) # 10in wide, 5in tall
plt.subplot(1,2,1) # 1 Row, 2 Columns, this is the 1st
plt.hist(sales_data['Unit'], bins=20) # 20 columns
plt.title("Unit Sales Distribution")
plt.xlabel("Units")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2) # 1 Row, 2 Columns, this is the 2nd
plt.hist(sales_data['Sales'], bins=20) # 20 columns
plt.title('Sales Amount Distribution')
plt.xlabel('Sales Amount ($)')
plt.ylabel('Frequency')
#plt.show()

# MARKDOWN
# ## Data Analysis - Units
# * There is a moderate amount of variability.
# * There is a wide range of sales volumes.
# * The distribution of unit sales appears to be skewed, with a longer tail on the right side, towards higher sales.
# ## Data Analysis - Sales
# * There is a significant amount of variability in sales.
# * There is a wide range of sales amounts.
# ## Data Analysis - Comparison
# * The skewness of the sales distribution is more pronounced thant the skewness of the unit distribution, suggesting that there are more extreme sales amounts.

# Group Analysis
sales_by_group = sales_data.groupby('Group')['Sales'].sum()
print(f"Sales by Group:\n{sales_by_group}")
print()

# Highest 
highest_group = sales_by_group.idxmax()
highest_group_amount = sales_by_group.max()
print(f"Group with highest sales: {highest_group} with ${highest_group_amount}")

#Lowest 
lowest_group = sales_by_group.idxmin()
lowest_group_amount = sales_by_group.min()
print(f"Group with lowest sales: {lowest_group} with ${lowest_group_amount}")

# MARKDOWN
# ## Data Analysis - Groups
# * Men have the highest sales.
# * Seniors have the lowest sales.
# * Sales amounts are relatively close.

# Data Analysis by datetime
sales_data['Week'] = sales_data['Date'].dt.isocalendar().week
sales_data['Month'] = sales_data['Date'].dt.month
sales_data['Quarter'] = sales_data['Date'].dt.quarter

print(f"Sales data summary:\n{sales_data.info}")
print()

# Calculating total sales per period
sales_by_week = sales_data.groupby('Week')['Sales'].sum()
sales_by_month = sales_data.groupby('Month')['Sales'].sum()
sales_by_quarter = sales_data.groupby('Quarter')['Sales'].sum()

print(f"Weekly Sales Report:\n{sales_by_week}")
print()
print(f"Monthly Sales Report:\n{sales_by_month}")
print()
print(f"Quarterly Sales Report:\n{sales_by_quarter}")
print()

# Plotting line charts
plt.figure(figsize=(10,5)) # 10in wide, 5in tall
plt.subplot(1,2,1) # 1 Row, 2 Columns, this is the 1st
sales_by_week.plot(kind="line")
plt.title("Weekly Sales")
plt.xlabel("Week")
plt.ylabel("Sales (millions of $)")
# updates the axis labels to show from the minimum week number to the maximum week number (inclusive)
ticks = range(sales_by_week.index.min(), sales_by_week.index.max() +1)
plt.xticks(ticks, ticks, rotation=45)

# Adding a trendline to the Weekly Graph
z = np.polyfit(sales_by_week.index, sales_by_week.values, 1)
p = np.poly1d(z)
plt.plot(sales_by_week.index, p(sales_by_week.index), "r--") # r-- means red dashed line

plt.subplot(1, 2, 2) # 1 Row, 2 Columns, this is the 2nd
sales_by_month.plot(kind='line')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales (millions of $)')
# updates the axis labels to show from the minimum month number to the maximum month number (inclusive)
ticks = range(sales_by_month.index.min(), sales_by_month.index.max() +1)
plt.xticks(ticks, ticks, rotation=45)
#plt.show()

# MARKDOWN
# ## Data Analysis - Date
# * The sales data show a general increasing trend, with some fluctuations throughout the weeks and months
# * The peak sales period is towards the end of the year

# State Analysis
sales_by_state = sales_data.groupby('State')['Sales'].sum()
units_by_state = sales_data.groupby('State')['Unit'].sum()

# Plot bar charts for total sales and units by state
plt.figure(figsize=(10, 5)) # 10in wide, 5in tall
plt.subplot(1, 2, 1) # 1 Row, 2 Columns, this is the 1st
sales_by_state.plot(kind='bar')
plt.title('Total Sales by State')
plt.xlabel('State')
plt.ylabel('Sales (millions of $)')

plt.subplot(1, 2, 2) # 1 Row, 2 Columns, this is the 2nd
units_by_state.plot(kind='bar')
plt.title('Total Units by State')
plt.xlabel('State')
plt.ylabel('Units')
# plt.show()

# Total sales by state AND group
sales_by_state_and_group = sales_data.groupby(['State', 'Group'])['Sales'].sum().unstack()

# Plot Grouped Bar for total sales by state and group
plt.figure(figsize=(10, 5))
sales_by_state_and_group.plot(kind='bar')
plt.title('Total Sales by State and Group')
plt.xlabel('State')
plt.ylabel('Sales (millions of $)')
plt.legend(title='Group')
# plt.show()

# MARKDOWN
# ## Data Analysis - State
# * VIC is the state with highest total sales and units.
# * NSW and SA are also strong performers from sales and units perspective.
# * NT and TAShave the lower total sales and units.
# * The women group is the strongest performer in terms of total sales across most states.

# State vs demographic groups analysis
plt.figure(figsize=(12, 6))
seaborn.barplot(x='State', y='Sales', hue='Group', data=sales_data)
plt.title('State vs Demographic Groups')
plt.xlabel('State')
plt.ylabel('Sales')
# plt.show()

# Time of Day Analysis
sales_by_time_of_day = sales_data['Time'].value_counts()

# Bar Chart
plt.figure(figsize=(10, 5)) # 10in wide, 5in tall
sales_by_time_of_day.plot(kind='bar')
plt.title('Total Sales by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Sales')
# plt.show()

# Dashboard
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Daily sales
axs[0, 0].plot(sales_data['Date'], sales_data['Sales'])
axs[0, 0].set_title('Daily Sales')
axs[0, 0].set_xlabel('Date')
axs[0, 0].set_ylabel('Sales')

# Weekly sales
axs[0, 1].plot(sales_by_week.index, sales_by_week.values)
axs[0, 1].set_title('Weekly Sales')
axs[0, 1].set_xlabel('Week')
axs[0, 1].set_ylabel('Sales')

# Monthly sales
axs[1, 0].plot(sales_by_month.index, sales_by_month.values)
axs[1, 0].set_title('Monthly Sales')
axs[1, 0].set_xlabel('Month')
axs[1, 0].set_ylabel('Sales')

# Quarterly sales
axs[1, 1].plot(sales_by_quarter.index, sales_by_quarter.values)
axs[1, 1].set_title('Quarterly Sales')
axs[1, 1].set_xlabel('Quarter')
axs[1, 1].set_ylabel('Sales')

fig.tight_layout()
plt.show()