
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# 加载数据集
#boston = load_boston()


X = df_train.iloc[:,:-1]
#X = X.drop(columns=['tested_positive'], inplace=False)
X = X.iloc[:,1:]
# X['tested_diff'] = X['tested_positive.1'] - X['tested_positive']
# X = X.drop(columns=['tested_positive', 'tested_positive.1'], inplace=False)
y = df_train.iloc[:,-1]

# 分割数据集为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 在训练集上进行预测
y_train_pred = model.predict(X_train)

# 在验证集上进行预测
y_val_pred = model.predict(X_val)

# 计算训练集的 MSE 损失
train_mse = mean_squared_error(y_train, y_train_pred)

# 计算验证集的 MSE 损失
val_mse = mean_squared_error(y_val, y_val_pred)

print(f'Training MSE: {train_mse:.4f}')
print(f'Validation MSE: {val_mse:.4f}')