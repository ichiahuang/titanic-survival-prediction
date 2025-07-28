import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
def preprocess_data(df):
  df['Name Title'] = df['Name'].str.extract(r'\b(Mr\.?|Mrs\.?|Miss\.?|Master\.?)\b')
  df['Name Title'] = df['Name Title'].fillna(value = 'Other')
  group_means = df.groupby('Name Title')['Age'].transform('mean')
  df['Age Filled'] = df['Age'].fillna(group_means)
  mean = df[df['Age Filled'].notnull()]['Age Filled'].mean()
  df['Age Filled'] = df['Age Filled'].fillna(value = mean)
  mean = df[df['Fare'].notnull()]['Fare'].mean()
  df['Fare'] = df['Fare'].fillna(value = mean)

  one_hot_Embarked = pd.get_dummies(df['Embarked'], prefix = 'Embarked').astype(int)
  df = pd.concat([df, one_hot_Embarked], axis = 1)
  one_hot_Sex = pd.get_dummies(df['Sex']).astype(int)
  df = pd.concat([df, one_hot_Sex], axis = 1)
  one_hot_Name_Title = pd.get_dummies(df['Name Title']).astype(int)
  df = pd.concat([df, one_hot_Name_Title], axis = 1)
  df['Cabin_Deck'] = df['Cabin'].str[0].fillna('Unknown')

  features = [
    'Pclass', 'Age Filled', 'SibSp', 'Parch', 'Fare',
    'Embarked_C', 'Embarked_Q', 'Embarked_S',
    'female', 'male',
    'Master', 'Miss', 'Mr', 'Mrs', 'Other',
    'Cabin_Deck'
    ]
  X = df[features]
  # y = df['Survived']

  one_hot_Cabin_Deck = pd.get_dummies(df['Cabin_Deck']).astype(int)
  X = pd.concat([X, one_hot_Cabin_Deck], axis = 1)
  X = X.drop(columns = 'Cabin_Deck')

  scaler = StandardScaler()
  X[['Age Filled', 'Fare']] = scaler.fit_transform(X[['Age Filled', 'Fare']])

  return X