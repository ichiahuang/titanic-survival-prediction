# Titanic Survival Prediction
[![Kaggle](https://img.shields.io/badge/Data-Kaggle_Titanic-20BEFF)](https://www.kaggle.com/competitions/titanic)
**Predict passenger survival** using machine learning with:  
- **Exploratory Data Analysis** (EDA) to uncover patterns
- **Feature engineering** (handling missing values, categorical encoding) 
- **Model comparison** (Logistic Regression vs XGBoost)

## Data Preprocess
### Extract Title form Name
I extracted titles from the Name column (`Master`, `Miss`, `Mr`, `Mrs`, `Other`).
```python
df['Name Title'] = df['Name'].str.extract(r'\b(Mr\.?|Mrs\.?|Miss\.?|Master\.?)\b')
df['Name Title'] = df['Name Title'].fillna(value = 'Other')
```

### Fill missing values

I filled missing `Age` using the average age of each title group and filled missing `Cabin` with Unknown.
```python
group_means = df.groupby('Name Title')['Age'].transform('mean')
df['Age Filled'] = df['Age'].fillna(group_means)
```

### Encode Categorical Features
I used **One-hot encoding** for features like:
- Sex
- Cabin
- Name Title
- Embarked

### Standardize Features
I used **StandardScaler** to scale features like:
- Age
- Fare

## EDA: Key Survival Patterns
### Sex & Class

<p align="center">
<img width="450" alt="Image" src="https://github.com/user-attachments/assets/dce670a4-c0b9-40d2-a240-8ce38de82649" />
</p>

We can observe that females had a higher survival rate than males, especially in first class.
 
```python
sns.barplot(data = df, x = 'Pclass', y = 'Survived', hue = 'Sex')
plt.title('Survival Rate by Pclass and Sex')
plt.show()
```

### Age & Sex

<p align="center">
<img width="450" alt="Image" src="https://github.com/user-attachments/assets/97faddab-c376-4f1d-b539-61f7acd1414a" />
</p>

Children and Elderly, especially females, also had a higher survival rate compared to adults passengers.

```python
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Elderly'])
sns.barplot(data=df, x='AgeGroup', y='Survived', hue='Sex')
plt.title('Survival Rate by Age Group and Sex')
plt.show()
```

### Summary of Key EDA Findings
- **Females** had higher survival rate than males across all classes.
- **First-class** passengers were more likely to survive than second or third class.
- **Children and Elderly**, especially girls, also had higher survival chances compared to adults.

## Model Building & Evaluation
I trained and compared two classification models to predict Titanic survival:
- **Logistic Regression**
- **XGBoost**

### Hyperparameter Tuning
I performed hyperparameter tuning using for loops:
- For Logistic Regression, I varied **C values** (inverse of regularization strength).
- For XGBoost, I tested different combinations of `max_depth`, `n_estimators`, and `reg_lambda`.
I plotted **validation accuracy** against hyperparameters to find the best setting:

<p align="center">
<img width="450" alt="Image" src="https://github.com/user-attachments/assets/94c4e725-4e01-432e-bd85-70ec0b4cd993" />
</p>
The best C value:1.011, best validation accuracy:0.820

<p align="center">
<img width="450" alt="Image" src="https://github.com/user-attachments/assets/7a9b6c55-25a9-42cc-8403-fc2311f45705" />
</p>
The best lambda: 57.576, best validation accuracy:0.871, best round:118

### Learning Curves
I also plotted learning curves to see how the models learn:

<p align="center">
<img width="450" alt="Image" src="https://github.com/user-attachments/assets/b594c15f-ef00-4ef6-82cc-eb46a49911fa" />
</p>

For **Logisitic Regression**, the training and validation accuracy are close. This means the model is not overfitting, and this model learns in a stable way. But the accuracy is only 82%, so this model is not a very good model.

<p align="center">
<img width="450" alt="Image" src="https://github.com/user-attachments/assets/470cceb7-80a0-4d66-ad8b-9ed8838cc074" />
</p>

For **XGBoost**, the training accuracy is very high, but the validation accuracy is quite lower. This means the model is quite overfitting that it learns training data very well, but it learns not that well on new data. Even so, XGBoost got higher scores in final test and on Kaggle competition.

### Model Performance

| Model              | Validation Accuracy | Final Accuracy | Precision score | Recall score | F1 score | Kaggle Score |
|--------------------|-----------|------------|-----------|-----------|-----------|-----------|
| Logistic Regression| 0.820     | 0.816      | 0.773     | 0.784     | 0.779     | 0.76315   |
| XGBoost            | 0.871     | 0.838      | 0.800     | 0.757     | 0.778     | 0.77511   |

### Top 3 Important Features
| Rank | Logistic Regression | Coefficient | XGBoost | Gain |
|------|---------------------|-------------|-----------------|--------|
| 1    | Master (Name Title) | +1.638      | Mr (Name Title) | 21.284 |
| 2    | female (Sex)        | +1.430      | female (Sex)    | 2.657  |
| 3    | E (Cabin Title)     | +1.212      | Pclass          | 1.850  |

This difference shows how linear and tree-based models may focus on different aspects of the same data.








