import pandas as pd
import numpy as np
from preprocess import preprocess_data
from logistic_model import final_logistic_model_predict
from XGBoost_model import final_xgb_model_predict
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"./data/train.csv")  # 模型參數
X = preprocess_data(data)
y = data['Survived']
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25, random_state=42)
columns_used = X_train.columns.tolist()
X_trainval = np.concatenate((X_train, X_val), axis=0)
y_trainval = np.concatenate((y_train, y_val), axis=0)

df_test = pd.read_csv(r"./data/test.csv") # 被預測數據
X_result = preprocess_data(df_test)
X_result = X_result.reindex(columns=columns_used, fill_value=0)

# try XGBoost model
y_pred_XGBoost = final_xgb_model_predict(X_trainval, y_trainval,X_result)
result = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': y_pred_XGBoost  # 放一維陣列
})
result.to_csv('XGBoost_model_result.csv', index=False)
