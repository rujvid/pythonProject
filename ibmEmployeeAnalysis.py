import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency


# Loading and exploring data
# Read the dataset
data = pd.read_csv("IBM employee data.csv")

# Display basic info about the dataset
print(data.info())

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Data Processing
# Encode categorical variables using one-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Summary statistics
print(data_encoded.describe())



# Data Analysis and visualization
# Attrition Analysis
# What is the overall attrition rate in the company?
# Are there any specific departments or job roles with higher attrition rates?
# How does attrition vary based on education levels and fields?

# Calculate overall attrition rate
attrition_rate = data['Attrition'].value_counts(normalize=True) * 100
print("Overall Attrition Rate:\n", attrition_rate)

# Attrition by department
dept_attrition = data.groupby('Department')['Attrition'].value_counts(normalize=True).unstack() * 100
print("Attrition by Department:\n", dept_attrition)

# Attrition by job role
role_attrition = data.groupby('JobRole')['Attrition'].value_counts(normalize=True).unstack() * 100
print("Attrition by Job Role:\n", role_attrition)

# Attrition by education level
education_attrition = data.groupby('EducationField')['Attrition'].value_counts(normalize=True).unstack() * 100
print("Attrition by Education Field:\n", education_attrition)

# Visualization of overall attrition rate
plt.figure(figsize=(6, 4))
sns.countplot(x='Attrition', data=data)
plt.title("Overall Attrition Rate")
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.show()

# Visualization of attrition by department
plt.figure(figsize=(10, 6))
sns.countplot(x='Department', hue='Attrition', data=data)
plt.title("Attrition by Department")
plt.xlabel("Department")
plt.ylabel("Count")
plt.legend(title="Attrition")
plt.show()

# Visualization of attrition by job role
plt.figure(figsize=(12, 8))
sns.countplot(y='JobRole', hue='Attrition', data=data)
plt.title("Attrition by Job Role")
plt.xlabel("Count")
plt.ylabel("Job Role")
plt.legend(title="Attrition")
plt.show()

# Visualization of attrition by education field
plt.figure(figsize=(10, 6))
sns.countplot(y='EducationField', hue='Attrition', data=data)
plt.title("Attrition by Education Field")
plt.xlabel("Count")
plt.ylabel("Education Field")
plt.legend(title="Attrition")
plt.show()


# Is there a correlation between age and attrition? Do younger or older employees tend to leave more?
# How does the average age differ between employees who left and those who stayed?
# Correlation between age and attrition
# Average age comparison
average_age_left = data[data['Attrition'] == 'Yes']['Age'].mean()
average_age_stayed = data[data['Attrition'] == 'No']['Age'].mean()

print("Average age of employees who left:", average_age_left)
print("Average age of employees who stayed:", average_age_stayed)

# Visualization of correlation between age and attrition
sns.boxplot(x='Attrition', y='Age', data=data)
plt.title("Correlation between Age and Attrition")
plt.xlabel("Attrition")
plt.ylabel("Age")
plt.show()



# Is there a correlation between job satisfaction and attrition?
# Do employees with lower job satisfaction tend to leave more?

# Calculate the correlation coefficient
correlation = data_encoded['JobSatisfaction'].corr(data_encoded['Attrition_Yes'])

print("Correlation between job satisfaction and attrition:", correlation)

# Box plot of job satisfaction by attrition
plt.figure(figsize=(8, 6))
sns.boxplot(x='Attrition_Yes', y='JobSatisfaction', data=data_encoded)
plt.title("Job Satisfaction vs. Attrition")
plt.xlabel("Attrition")
plt.ylabel("Job Satisfaction")
plt.show()


# Does monthly income have an impact on attrition?
# Do employees with lower income leave more frequently?

# Remove unnecessary columns
data = data.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1)

# Encode categorical variables using one-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Box plot of MonthlyIncome vs Attrition
plt.figure(figsize=(10, 6))
sns.boxplot(x='Attrition_Yes', y='MonthlyIncome', data=data_encoded)
plt.title("Monthly Income vs Attrition")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")
plt.show()

# Perform t-test to compare MonthlyIncome for employees who left and stayed
# The t-test helps us determine whether the difference in means is statistically significant
# or could have occurred by chance. The t-statistic and p-value from the test provide insights
# into the impact of monthly income on attrition.
attrition_yes = data_encoded[data_encoded['Attrition_Yes'] == 1]['MonthlyIncome']
attrition_no = data_encoded[data_encoded['Attrition_Yes'] == 0]['MonthlyIncome']
t_statistic, p_value = ttest_ind(attrition_yes, attrition_no)

print("T-Statistic:", t_statistic)
print("P-Value:", p_value)



# Is there a correlation between work-life balance and attrition?
# Do employees with poor work-life balance leave more often?
#try2