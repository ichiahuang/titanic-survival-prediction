import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from preprocess import preprocess_data


df = pd.read_csv(r"./data/train.csv") # load data
X = preprocess_data(df)
y = df['Survived']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25, random_state=42)
columns_used = X_train.columns.tolist()

# choose best C value
C_values = np.linspace(0.001, 100, 100)  
val_accuracy = []
for i in C_values:
  model = LogisticRegression(penalty = 'l2', C = i, solver = 'liblinear')
  model.fit(X_train, y_train)
  y_val_pred = model.predict(X_val)
  accuracy = accuracy_score(y_val, y_val_pred)
  val_accuracy.append(accuracy)
  # print(f'C value:{i:8.3f}, accuracy:{accuracy:.3f}')

best_C = C_values[np.argmax(val_accuracy)]
print(f'best C value:{best_C:.3f}, best accuracy:{max(val_accuracy):.3f}')
# best C value:1.011, best accuracy:0.820

# draw learning curve
X_trainval = np.concatenate((X_train, X_val), axis=0) 
y_trainval = np.concatenate((y_train, y_val), axis=0)
model = LogisticRegression(penalty = 'l2', C = best_C, solver = 'liblinear')
train_sizes, train_scores, val_scores = learning_curve(
    model, X_trainval, y_trainval,
    cv = 5,
    train_sizes = np.linspace(0.1, 1.0, 500),
    random_state = 42
)
train_mean = train_scores.mean(axis = 1)
val_mean = val_scores.mean(axis = 1)
plt.plot(train_sizes, train_mean, label = 'Traning accuracy')
plt.plot(train_sizes, val_mean, label = 'Validation accuracy')
plt.xlabel('Training size')
plt.ylabel('Accuracy')
plt.title('Learning curve')
plt.legend()
plt.show()


# test final logistic model
final_logistic_model = LogisticRegression(  
    penalty = 'l2',
    C = best_C,
    solver = 'liblinear'
)
final_logistic_model.fit(X_trainval, y_trainval)
y_test_pred = final_logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
print(f'Logistic test accuracy:{accuracy:.3f}')
# final Logistic test accuracy:0.816


def final_logistic_model(X_test):
    final_logistic_model = LogisticRegression(
        penalty = 'l2',
        C = best_C,
        solver = 'liblinear'
    )
    final_logistic_model.fit(X_trainval, y_trainval)
    y_test_pred = final_logistic_model.predict(X_test)
    return y_test_pred