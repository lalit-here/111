# Daily Data Work: NumPy, Pandas, Matplotlib, Seaborn

# --------------------- NUMPY ---------------------
import numpy as np

# Create arrays
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])

# Array properties
a.ndim
b.shape
b.size
b.dtype

# Creation shortcuts
np.zeros((2,2))
np.ones((3,3))
np.full((2,3), 7)
np.eye(3)
np.random.rand(3,2)
np.random.randint(0, 10, (2, 3))
np.random.normal(0, 1, 1000)

# Broadcasting and operations
a + 10
a * np.array([10, 100, 1000])

# Math
np.mean(a)
np.std(a)
np.max(a)
np.argmin(a)
np.exp(a)
np.log(a)
np.sqrt(a)


# --------------------- PANDAS ---------------------
import pandas as pd

# Load & Save Data
df = pd.read_csv("file.csv")
df.to_csv("output.csv", index=False)

# Inspection
df.head()
df.tail()
df.sample(5)
df.info()
df.describe()
df.shape
df['column'].unique()
df['column'].value_counts(normalize=True)

# Filtering
df[df['Age'] > 25]
df[(df['Age'] > 25) & (df['Gender'] == 'Male')]
df.query('Age > 25 and Gender == "Male"')

# Transform Columns
df['New'] = df['Old'] * 2
df['Name'] = df['First'] + " " + df['Last']
df['col'] = df['col'].str.lower()

# Groupby
df.groupby('Department')['Salary'].mean()
df.groupby(['Dept', 'Gender']).agg({
    'Salary': ['mean', 'max'],
    'Age': 'median'
})

# Missing Data
df.isnull().sum()
df.dropna()
df.fillna({'Age': 25})

# Categorical for memory
df['city'] = df['city'].astype('category')

# Dates
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.day_name()


# --------------------- MATPLOTLIB ---------------------
import matplotlib.pyplot as plt

# Line plot
plt.figure(figsize=(8,5))
plt.plot([1, 2, 3], [4, 5, 6], color='red', marker='o', linestyle='--')
plt.title("Line Plot")
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.grid(True)
plt.show()

# Bar Plot
plt.bar(['A', 'B', 'C'], [10, 20, 15], color='skyblue')
plt.title("Bar Plot")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.show()

# Horizontal Bar Plot
plt.barh(['A', 'B', 'C'], [10, 20, 15], color='salmon')
plt.title("Horizontal Bar Plot")
plt.show()

# Histogram
plt.hist([1, 1, 2, 2, 2, 3, 4, 4, 5], bins=5, color='lightgreen', edgecolor='black')
plt.title("Histogram")
plt.xlabel("Bins")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
plt.scatter(x, y, color='purple')
plt.title("Scatter Plot")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()

# Pie Chart
labels = ['A', 'B', 'C']
sizes = [50, 30, 20]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Pie Chart")
plt.axis('equal')
plt.show()

# Box Plot
data = [7, 15, 13, 20, 22, 25, 30, 32]
plt.boxplot(data)
plt.title("Box Plot")
plt.show()

# Subplots
fig, axs = plt.subplots(1, 2, figsize=(12,5))
axs[0].bar(['A', 'B', 'C'], [10, 20, 15])
axs[1].plot([1, 2, 3], [5, 3, 6])
plt.tight_layout()
plt.show()


# --------------------- SEABORN ---------------------
import seaborn as sns

# Theme
sns.set_theme(style="whitegrid")

# Boxplot, Violin, Count
sns.boxplot(x='Gender', y='Salary', data=df)
sns.violinplot(x='Category', y='Value', data=df)
sns.countplot(x='Product', data=df)

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Pairplot
sns.pairplot(df[['Age', 'Salary', 'Score']], hue='Gender')

# Regression and KDE
sns.regplot(x='Age', y='Salary', data=df)
sns.kdeplot(df['Score'], fill=True)


# --------------------- BONUS ---------------------

# Convert between pandas & numpy
df.values                      # to numpy
pd.DataFrame(a)               # from numpy

# Pickle save/load
df.to_pickle('mydf.pkl')
df = pd.read_pickle('mydf.pkl')

# Method chaining with assign and pipe
df = (
    pd.read_csv("data.csv")
      .dropna()
      .assign(ratio = lambda d: d['score'] / d['time'])
      .pipe(lambda d: d[d['ratio'] > 1])
)

# Inline plots for Jupyter
# %matplotlib inline
