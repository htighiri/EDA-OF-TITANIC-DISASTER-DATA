## import lib

# In [1]
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

## read data

# In [2]
raw_df = pd.read_csv('train.csv')
print(raw_df.to_string())

## explore data

# In [3]
raw_df.info()

# In [4]
raw_df.describe()

# In [5]
print(raw_df.columns.values)

# In [6]
raw_df.head()

# In [7]
raw_df.tail()

### DATA CLEANING

## In order to analysis the age column, since there are missing data in the AGE column, I will be replacing the missing values with the median value of the AGE values.

# In [8]
raw_df

# In [9]
raw_df.info()

# In [10]
raw_df.describe()

# In [11]
raw_df.describe(include=['O'])

## Check how many NaN values are there

# In [12]
raw_df.isnull().sum()

# In [13]
sns.heatmap(raw_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

## copy data for cleaning

# In [14]
clean_df = raw_df.copy()
clean_df

## Replacing missing Age values with the median

# Find the median of the Age Column

# In [15]
median_Age = clean_df['Age'].median()
median_Age

## Replace the missing AGE values with the median value 

# In [16]
clean_df['Age'].fillna(median_Age, inplace=True)
clean_df


## Replacing missing Embarked values with the mode since it is a categorical data

# Find the meode of the Embarked Column

# In [17]
mode_Embarked = clean_df['Embarked'].mode()[0]
mode_Embarked

## Check the data to confirm the actions

# In [18]
clean_df.info()


### DATA ANALYSIS


## Analysing the data to check for what categories of passengers were most likely to survive the Titanic disaster.

## Data Aggregation and Grouping 

# In [19]
Pclass_df = clean_df
Pclass_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# In [20]
Sex_df = clean_df
Sex_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# In [21]
SibSp_df = clean_df
SibSp_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# In [22]
Parch_df = clean_df
Parch_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

## Data Visualization

# In [23]
Age_df = clean_df
g = sns.FacetGrid(Age_df, col='Survived')
g.map(plt.hist, 'Age', bins=10)

# In [24]
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = clean_df[clean_df['Sex']=='female']
men = clean_df[clean_df['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')

# In [25]
grid = sns.FacetGrid(clean_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

# In [26]
sns.barplot(x='Pclass', y='Survived', data=clean_df)

# In [27]
FacetGrid = sns.FacetGrid(clean_df, row='Embarked', height=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()


### Engineering new attributes


# I want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival.

# Based on my assumptions and decisions I want to drop the Cabin and Ticket features.

# In [28]
clean_df.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
clean_df

# In [29]
print(clean_df.columns.values)

# In [30]
clean_df.info()

## Creating new attribute (Title)

# In [31]
modified_df = clean_df.copy()
modified_df['Title'] = modified_df['Name'].apply(lambda x : x.split()[1].strip())
modified_df

## Raplacing the Titles for easy analysis

# In [32]
modified_df['Title'] = modified_df['Title'].replace('Mlle.', 'Miss')
modified_df['Title'] = modified_df['Title'].replace('Ms.', 'Miss')
modified_df['Title'] = modified_df['Title'].replace('Mme.', 'Mrs')
modified_df['Title'] = modified_df['Title'].replace('Mr.', 'Mr')
modified_df['Title'] = modified_df['Title'].replace('Master.', 'Master')
modified_df['Title'] = modified_df['Title'].replace('Mrs.', 'Mrs')
modified_df['Title'] = modified_df['Title'].replace('Miss.', 'Miss')
modified_df['Title'] = modified_df['Title'].replace(['Melkebeke,', 'Messemaeker,', 'Mulder,', 'Pelsmaeker,','Planke,', 'Shawah,', 'Steen,', 'Velde,', 'Walle,', 'der', 'the', 'y', 'Gordon,', 'Impe,', 'Jonkheer.', 'Billiard,', 'Carlo,', 'Cruyssen,', 'Lady', 'Countess','Capt.', 'Col.', 'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer', 'Dona'], 'Rare')
modified_df

# In [33]
modified_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
