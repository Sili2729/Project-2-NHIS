import pandas as pd
import numpy as np

file_path = r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Uncleaned data\dataset_1.xlsx"
xls = pd.ExcelFile(file_path)

xls.sheet_names

df = pd.read_excel(xls, sheet_name="dataset_1")

# Display basic information and first few rows
df.info(), df.head()


df_cleaned = df.copy()

# Convert categorical columns
df_cleaned['yr'] = df_cleaned['yr'].astype('category')
df_cleaned['mnth'] = df_cleaned['mnth'].astype('category')
df_cleaned['hr'] = df_cleaned['hr'].astype('category')

# Verify changes
df_cleaned.info(), df_cleaned.head()


# Load the second dataset
file_path_2 = r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Uncleaned data\dataset_2.xlsx"

# Check sheet names
xls_2 = pd.ExcelFile(file_path_2)
xls_2.sheet_names


# Load the data from the first sheet
df2 = pd.read_excel(file_path_2, sheet_name="dataset_2")

# Display basic info and the first few rows
df2.info(), df2.head()

# Drop the 'Unnamed: 0' column
df2_cleaned = df2.drop(columns=['Unnamed: 0'])

# Fill missing values in 'atemp' with the column mean
df2_cleaned['atemp'].fillna(df2_cleaned['atemp'].mean(), inplace=True)

# Verify changes
df2_cleaned.info(), df2_cleaned.head()


# Define file paths for the cleaned datasets
cleaned_file_path_1 = r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Cleaned data\cleaned dataset-1.xlsx"
cleaned_file_path_2 = r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Cleaned data\cleaned dataset-2.xlsx"

# Save cleaned datasets to Excel files
with pd.ExcelWriter(cleaned_file_path_1) as writer:
    df_cleaned.to_excel(writer, sheet_name="cleaned_dataset_1", index=False)

with pd.ExcelWriter(cleaned_file_path_2) as writer:
    df2_cleaned.to_excel(writer, sheet_name="cleaned_dataset_2", index=False)

# Provide download links
cleaned_file_path_1, cleaned_file_path_2


# Load both cleaned datasets
file1 = r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Cleaned data\cleaned dataset-1.xlsx"
file2 = r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Cleaned data\cleaned dataset-2.xlsx"
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# Identify common columns for merging
common_cols = list(set(df1.columns) & set(df2.columns))
print("Common columns:", common_cols)

# Merge on the first common column (modify if needed)
if common_cols:
    df_merged = pd.merge(df1, df2, on=common_cols[0], how="inner")
else:
    raise ValueError("No common column found for merging.")

# Drop duplicates
df_merged = df_merged.drop_duplicates()

# Drop columns with too many missing values (threshold: 50%)
df_merged = df_merged.dropna(thresh=len(df_merged) * 0.5, axis=1)

# Fill remaining missing values with column mean
df_merged.fillna(df_merged.mean(numeric_only=True), inplace=True)

# Save the cleaned merged dataset
merged_file_path = r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Cleaned data\New folder\merged dataset 1&2.xlsx"
df_merged.to_excel(merged_file_path, index=False)

merged_file_path


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Drop constant columns
df_cleaned = df.drop(columns=['season', 'yr', 'mnth'])

# Save the cleaned dataset
cleaned_file_path =  r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Cleaned data\New folder\merged dataset 1&2.xlsx"
df_cleaned.to_excel(cleaned_file_path, index=False)

# Confirm dropped columns
df_cleaned.head(), cleaned_file_path


# Set visualization style
sns.set_style("whitegrid")

# Plot correlation heatmap without gaps
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5, square=True)

# Title
plt.title("Feature Correlation Heatmap")


# Load the merged dataset
file_path = r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Cleaned data\New folder\merged dataset 1&2.xlsx"
df = pd.read_excel(file_path)
# Set visualization style
sns.set_style("whitegrid")

# Create figure for the line chart
fig, ax = plt.subplots(figsize=(12, 6))

# Line Chart - Trend of bike rentals over time
sns.lineplot(data=df, x="dteday", y="cnt", ax=ax, color="green", marker='o')
ax.set_title("Trend of Bike Rentals Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Total Rentals")
ax.tick_params(axis='x', rotation=30)

# Show plot
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Cleaned data\New folder\merged dataset 1&2.xlsx"
df = pd.read_excel(file_path)

# Set visualization style
sns.set_style("whitegrid")

# Create scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="temp", y="cnt", color="red", alpha=0.6)

# Add labels and title
plt.title("Scatter Plot of Temperature vs. Bike Rentals")
plt.xlabel("Temperature")
plt.ylabel("Total Rentals")

# Show plot
plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Cleaned data\New folder\merged dataset 1&2.xlsx"
df = pd.read_excel(file_path)

# Set visualization style
sns.set_style("whitegrid")

# Create box plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, y="cnt", color="skyblue")

# Add title and labels
plt.title("Box Plot of Total Bike Rentals")
plt.ylabel("Total Rentals")

# Show plot
plt.show()


import pandas as pd

# Load datasets
file1 = r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Cleaned data\New folder\merged dataset 1&2.xlsx"
file2 = r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Cleaned data\cleaned dataset-3.xlsx"

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# Display column names
print("Columns in df1:", df1.columns)
print("Columns in df2:", df2.columns)

# Find common columns
common_cols = list(set(df1.columns) & set(df2.columns))
print("Common columns:", common_cols)

# Merge datasets on common columns (modify if needed)
if common_cols:
    df_merged = pd.merge(df1, df2, on=common_cols, how="outer")  # Use "inner" if you only want matching rows
else:
    raise ValueError("No common columns found for merging!")

# Drop duplicate rows
df_merged = df_merged.drop_duplicates()

# Drop columns with more than 50% missing values
df_merged = df_merged.dropna(thresh=len(df_merged) * 0.5, axis=1)

# Fill remaining missing values with column mean (for numerical data)
df_merged.fillna(df_merged.mean(numeric_only=True), inplace=True)

# Save the merged and cleaned dataset
output_path = r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Cleaned data\New folder\final merged data.xlsx"
df_merged.to_excel(output_path, index=False)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the merged dataset
file_path = r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Cleaned data\New folder\final merged data.xlsx"
df = pd.read_excel(file_path)


# Convert date column to datetime format
df["dteday"] = pd.to_datetime(df["dteday"])

# Set style
sns.set_style("whitegrid")

# Create a figure with subplots (adjust size for better readability)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # 2x2 layout for 4 charts

 #**Heatmap - Feature Correlation**
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5, ax=axes[0, 0])
axes[0, 0].set_title("Feature Correlation Heatmap")

#**Line Chart - Trend of Bike Rentals Over Time**
sns.lineplot(data=df, x="dteday", y="cnt", ax=axes[0, 1], color="green", marker='o')
axes[0, 1].set_title("Trend of Bike Rentals Over Time")
axes[0, 1].set_xlabel("Date")
axes[0, 1].set_ylabel("Total Rentals")
axes[0, 1].tick_params(axis='x', rotation=30)

#**Box Plot - Checking for Outliers in Total Bike Rentals**
sns.boxplot(data=df, y="cnt", ax=axes[1, 0], color="skyblue")
axes[1, 0].set_title("Box Plot of Total Bike Rentals")
axes[1, 0].set_ylabel("Total Rentals")

#**Scatter Plot - Temperature vs. Bike Rentals**
sns.scatterplot(data=df, x="temp", y="cnt", ax=axes[1, 1], color="red", alpha=0.6)
axes[1, 1].set_title("Scatter Plot of Temperature vs. Bike Rentals")
axes[1, 1].set_xlabel("Temperature")
axes[1, 1].set_ylabel("Total Rentals")

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the merged dataset
file_path =  r"C:\Users\Hitansh\Desktop\PROJECTS\PYTHON PROJECTS\Nexthire projects\Project-2\Cleaned data\New folder\final merged data.xlsx"
df = pd.read_excel(file_path)

# Set visualization style
sns.set_style("whitegrid")

# Ensure datetime format for proper sorting
df["dteday"] = pd.to_datetime(df["dteday"])

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 16))

# Hourly Bike Rentals**
sns.barplot(data=df, x="hr", y="cnt", ax=axes[0], palette="viridis", ci=None)
axes[0].set_title("Hourly Distribution of Bike Rentals")
axes[0].set_xlabel("Hour of the Day")
axes[0].set_ylabel("Total Rentals")

#Impact of Weather on Bike Rentals (Box Plot)**
sns.boxplot(data=df, x="weathersit", y="cnt", ax=axes[1], palette="viridis")
axes[1].set_title("Impact of Weather on Bike Rentals")
axes[1].set_xlabel("Weather Situation (1=Clear, 2=Cloudy, 3=Rain/Snow)")
axes[1].set_ylabel("Total Rentals")

# Show plots
plt.tight_layout()
plt.show()


