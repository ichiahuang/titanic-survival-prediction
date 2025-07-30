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

For **Logisitic Regression**, the training and validation accuracy are close. This means the model is not overfitting, and this model learns in a stable way. However, with only 82% accuracy, this model is not very competitive for this task.

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

## Kaggle competition result

<img width="1300" height="279" alt="Image" src="https://github.com/user-attachments/assets/05003790-f17f-4f64-b5f4-6b96b3022e25" />
<img width="1300" height="171" alt="Image" src="https://github.com/user-attachments/assets/6d8d8c51-2984-43af-96a9-17ba8a3b3302" />


## How to run
You can click the code files below to check each step:
- [preprocess code](./code/preprocess.py)
- Clean data and do feature engineering
- [logistic model code](./code/logistic_model.py)
- Train logistic regression and tune C value
- [XGBoost model code](./code/XGBoost_model)
-  Train XGBoost and tune hyperparameters
- [logistic predict code](./code/logistic_predict)
- Predict and output result with logistic model
- [XGBoost predict code](./code/XGBoost_predict)
- Predict and output result with XGBoost model

## Conclusion

This project analysed the Titanic passenger dataset and used machine learning model to predict survival outcomes.

- Through feature engineering and hyperparameter tuning, XGBoost model has the highest performance, with a Kaggle score of 0.77511.
- Key features that contributed to the model’s performance included:
  - Name Title
  - Sex
  - Class
  - Cabin Title
- Logistic Regression model offered a stable baseline, while XGBoost model showed better generalization on the test set.

This project shows my ability to handle data preprocessing, model building and result interpretation using **Python**, **Scikit-learn** and **XGBoost**.

## Reflection
This was my first machine learning project and I learned a lot from it.

- I try feature engineering like one-hot and extract title from `Name`, It's very fun but not always easy.
- I also noticed model type and tuning can change the result a lot.
- Sometimes I am not sure how to do things like learning curve or how to see feature importance. But after I try and google some things, I feel better with it.

In the future, I hope to explore deep learning and more advanced data problems. I think this project really helps me understand machine learning more.


*Note: English is not my first language, This README is written by myself with some grammar checkingg tools.*








