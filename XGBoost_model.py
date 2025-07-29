import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from preprocess import preprocess_data

# load data
df = pd.read_csv(r"./data/train.csv") 
X = preprocess_data(df)
y = df['Survived']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25, random_state=42)
columns_used = X_train.columns.tolist()

# build XGBoost model
# choose best lambda
lambdas = np.linspace(0.001, 100, 100)
val_accuracy = []
val_logloss = []
X_train = np.array(X_train)  # 比 np.array() 更安全，不會複製已為array的數據
X_val = np.array(X_val)
y_train = np.array(y_train).ravel()  # 確保y是1D array
y_val = np.array(y_val).ravel()
for i in lambdas:
  xgb_model = XGBClassifier(
      n_estimators = 500,
      max_depth=5,
      learning_rate = 0.1,
      reg_lambda = i,
      early_stopping_rounds = 10,
      random_state=42
  )
  xgb_model.fit(
      X_train, y_train,
      eval_set = [(X_val, y_val)],
      verbose=False
  )

  best_round = xgb_model.best_iteration
  y_val_pred = xgb_model.predict(X_val, iteration_range=(0, best_round + 1))
  accuracy = accuracy_score(y_val, y_val_pred)
  val_accuracy.append(accuracy)
  # print(f'lambda:{i:7.3f}, accuracy:{accuracy:.3f}')

  eval_result = xgb_model.evals_result()
  logloss = eval_result['validation_0']['logloss'][best_round]
  val_logloss.append(logloss)

best_lambda = lambdas[np.argmax(val_accuracy)]
best_accuracy = max(val_accuracy)
print(f'best lambda:{best_lambda:7.3f}, best accuracy:{best_accuracy:.3f}')
# best lambda: 57.576, best accuracy:0.871

# plot best lambda
plt.plot(lambdas, val_accuracy) 
plt.scatter(best_lambda,best_accuracy, color = 'red')
plt.xlabel('Lambda (reg_lambda)')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs Lambda (XGBoost)')
plt.show()

# plot log loss curve
plt.plot(lambdas, val_logloss, label='Validation LogLoss')
plt.xlabel('Lambda')
plt.ylabel('Log Loss')
plt.title('LogLoss vs Lambda')
plt.show()

# plot learning curve
X_trainval = np.concatenate((X_train, X_val), axis=0)
y_trainval = np.concatenate((y_train, y_val), axis=0)
xgb_model = XGBClassifier(
      n_estimators = 500,
      max_depth=5,
      learning_rate = 0.1,
      reg_lambda = best_lambda,
      random_state=42
  )
train_sizes, train_scores, val_scores = learning_curve(
    estimator = xgb_model,
    X = X_trainval, y = y_trainval,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 100),
    scoring='accuracy',
    n_jobs=-1
)
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Validation score')
plt.title("XGBoost Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# test final XGBoost model
best_n_estimators = best_round + 1
final_xgb_model = XGBClassifier(
    n_estimators = best_n_estimators,
    max_depth = 5,
    learning_rate = 0.1,
    reg_lambda = best_lambda,
    random_state = 42
)
final_xgb_model.fit(X_trainval, y_trainval)
y_test_pred = final_xgb_model.predict(np.array(X_test))
accuracy = accuracy_score(y_test, y_test_pred)
print(f'accuracy:{accuracy:.3f}')
#final XGBoost test accuracy:0.838

def final_xgb_model(X_test):
    best_n_estimators = best_round + 1
    final_xgb_model = XGBClassifier(
        n_estimators = best_n_estimators,
        max_depth = 5,
        learning_rate = 0.1,
        reg_lambda = best_lambda,
        random_state = 42
    )
    final_xgb_model.fit(X_trainval, y_trainval)
    y_test_pred = final_xgb_model.predict(np.array(X_test))
    return y_test_pred