import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the data and explore its structure
data = pd.read_csv("Suicides_India.csv")

# Display basic information about the dataset
print(data.info())

# Step 2: Data cleaning and preprocessing (if required)
# Check for missing values
print(data.isnull().sum())

# Handle missing values if any
# For example, you can fill missing values in a particular column with its mean or mode.

# Step 3: Exploratory Data Analysis (EDA)
# Here, you can explore the dataset to understand its content better.
# For example, check the columns, their data types, unique values, etc.
print(data.head())

total_suicides = len(data)
print("Total number of suicide cases:", total_suicides)

common_reasons = data['Type'].value_counts().head(5)  # Change 5 to any desired number
print("Most common reasons for suicide:\n", common_reasons)

# Calculate suicide rates by state (using raw counts)
state_suicides = data['State'].value_counts()

highest_suicide_states = state_suicides.nlargest(5)  # Change 5 to any desired number
lowest_suicide_states = state_suicides.nsmallest(5)   # Change 5 to any desired number

print("States with highest suicide rates:\n", highest_suicide_states)
print("\nStates with lowest suicide rates:\n", lowest_suicide_states)

# Create a pivot table for number of suicides across age and gender
pivot_age_gender = data.pivot_table(index='Age_group', columns='Gender', aggfunc='size')

# Plot the pivot table as a bar plot
pivot_age_gender.plot(kind='bar', figsize=(10, 6))
plt.title("Number of Suicides Across Age Groups and Gender")
plt.xlabel("Age Group")
plt.ylabel("Number of Suicides")
plt.show()
