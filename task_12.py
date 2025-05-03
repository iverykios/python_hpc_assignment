import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('HPC/all_reults.csv')
df.columns = df.columns.str.strip()  # Strip leading/trailing spaces from column names
print("Sanitized columns:", df.columns.tolist())

plt.figure(figsize=(8, 5))
sns.histplot(df['mean_temp'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Mean Temperatures')
plt.xlabel('Mean Temperature (ºC)')
plt.ylabel('Number of Buildings')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.boxplot(x=df['mean_temp'], color='lightgreen')
plt.title('Boxplot of Mean Temperatures')
plt.xlabel('Mean Temperature (ºC)')
plt.tight_layout()
plt.show()

avg_mean_temp = df['mean_temp'].mean()
print(f"Average mean temperature: {avg_mean_temp:.2f} ºC")

avg_std_temp = df['std_temp'].mean()
print(f"Average temperature standard deviation: {avg_std_temp:.2f} ºC")

count_above_18 = (df['pct_above_18'] >= 50).sum()
count_below_15 = (df['pct_below_15'] >= 50).sum()
print(f"Number of buildings with ≥50% area above 18ºC: {count_above_18}")
print(f"Number of buildings with ≥50% area below 15ºC: {count_below_15}")


plt.figure(figsize=(6, 5))
sns.barplot(
    x=['≥50% above 18ºC', '≥50% below 15ºC'],
    y=[count_above_18, count_below_15],
    palette='pastel'
)
plt.title('Number of Buildings by Temperature Thresholds')
plt.ylabel('Number of Buildings')
plt.tight_layout()
plt.show()
