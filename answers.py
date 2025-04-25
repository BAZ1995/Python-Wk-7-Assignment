# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Enable Seaborn styling
sns.set(style="whitegrid")

# ------------------------------
# Task 1: Load and Explore the Dataset
# ------------------------------
try:
    # Load Iris dataset from sklearn and convert to DataFrame
    iris_data = load_iris()
    df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target
    df['species'] = df['species'].apply(lambda x: iris_data.target_names[x])

    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check data types and missing values
    print("\nData types and missing values:")
    print(df.info())

    print("\nMissing values in each column:")
    print(df.isnull().sum())

    # Clean dataset (no missing values in this dataset, but here's the line in case)
    df.dropna(inplace=True)

except FileNotFoundError:
    print("Error: File not found.")
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")

# ------------------------------
# Task 2: Basic Data Analysis
# ------------------------------

# Descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# Grouping: Average of features per species
grouped_means = df.groupby("species").mean()
print("\nMean values grouped by species:")
print(grouped_means)

# Observations:
print("\nInteresting pattern:")
print("- Virginica species generally has the highest mean values for most features.")
print("- Setosa species has the smallest petal length and width on average.")

# ------------------------------
# Task 3: Data Visualization
# ------------------------------

# Line Chart: Simulated trend (cumulative petal length as a time series)
plt.figure(figsize=(10, 5))
df_sorted = df.sort_values(by='petal length (cm)').reset_index()
df_sorted['cumulative_petal_length'] = df_sorted['petal length (cm)'].cumsum()
plt.plot(df_sorted.index, df_sorted['cumulative_petal_length'], label='Cumulative Petal Length', color='teal')
plt.title('Line Chart: Cumulative Petal Length')
plt.xlabel('Index')
plt.ylabel('Cumulative Petal Length (cm)')
plt.legend()
plt.show()

# Bar Chart: Average Petal Length by Species
plt.figure(figsize=(8, 5))
sns.barplot(x=grouped_means.index, y=grouped_means['petal length (cm)'], palette='Set2')
plt.title('Bar Chart: Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# Histogram: Distribution of Sepal Width
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True, color='coral')
plt.title('Histogram: Sepal Width Distribution')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='Dark2')
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
