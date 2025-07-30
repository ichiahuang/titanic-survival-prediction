import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from preprocess import preprocess_data

def final_logistic_model_predict(X_trainval, y_trainval, X_test):
    best_C = 1.011
    final_logistic_model = LogisticRegression(
        penalty = 'l2',
        C = best_C,
        solver = 'liblinear'
    )
    final_logistic_model.fit(X_trainval, y_trainval)
    y_test_pred = final_logistic_model.predict(X_test)
    return y_test_pred

if __name__ == '__main__':
# load data
    df = pd.read_csv(r"./data/train.csv") 
    X = preprocess_data(df)
    y = df['Survived']

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25, random_state=42)
    columns_used = X_train.columns.tolist()


    # build logistic regression model
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
    best_accuracy = max(val_accuracy)
    print(f'best C value:{best_C:.3f}, best accuracy:{best_accuracy:.3f}')
    # best C value:1.011, best accuracy:0.820

    plt.plot(C_values, val_accuracy)
    plt.scatter(best_C,best_accuracy, color = 'red')
    plt.xlabel('C values')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs C values (Logistic Regression)')
    plt.show()

    # plot learning curve
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
    coef = final_logistic_model.coef_[0] # find top 3 coef
    feature_names = X_test.columns
    df_coef = pd.DataFrame({
        'feature':feature_names,
        'coef':coef,
        'abs_coef':np.abs(coef)
    })
    top3_coef = df_coef.sort_values(by = 'abs_coef', ascending = False).head(3)
    print(top3_coef)
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    print(f'Logistic test accuracy:{accuracy:.3f}')
    print(f'Logistic test precision score:{precision:.3f}')
    print(f'Logistic test recall score:{recall:.3f}')
    print(f'Logistic test f1 score:{f1:.3f}')
    # final Logistic test accuracy:0.816, precision score:0.773, recall score:0.784, f1 score:0.779