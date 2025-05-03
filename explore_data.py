import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Save correlation matrix plot
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Heart Disease Dataset')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# Count of target classes
plt.figure(figsize=(8, 6))
df['target'].value_counts().plot(kind='bar')
plt.title('Distribution of Heart Disease Classes')
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.xticks([0, 1], ['No Disease (0)', 'Disease (1)'])
plt.savefig('target_distribution.png')

print("\nAnalysis complete. Check the generated plots for visualization.") 