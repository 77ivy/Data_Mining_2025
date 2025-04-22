## import sys
import pandas as pd
import numpy as np

# 读入数据
## data = pd.read_csv('./train.csv', encoding='big5')
# 讀入數據（使用絕對路徑）
data = pd.read_csv('/Users/nakaonki/Downloads/train.csv', encoding='big5')


# 数据预处理
data = data.iloc[:, 3:]
##data[data == 'A'] = 0.0
# 先把 `#, *, x, A` 這些無效數據轉換為 NaN，再用 0 填補
data.replace(["#", "*", "x", "A"], np.nan, inplace=True)
data.fillna(0, inplace=True)  # 用 0 替代 NaN

# 確保所有欄位都是數字格式
data = data.apply(pd.to_numeric, errors='coerce')
data.fillna(0, inplace=True)  # 再次用 0 替換 NaN
raw_data = data.to_numpy()

# 按月分割数据
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = sample

# 分割x和y
x = np.empty([12 * 471, 18 * 9 * 2], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x1 = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1,-1)  # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            x[month * 471 + day * 24 + hour, :18 * 9] = x1
            # 在这里加入了x的二次项
            x[month * 471 + day * 24 + hour, 18 * 9: 18 * 9 * 2] = np.power(x1, 2)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value

# 对x标准化
mean_x = np.mean(x, axis=0)  # 18 * 9 * 2
std_x = np.std(x, axis=0)  # 18 * 9 * 2
for i in range(len(x)):  # 12 * 471
    for j in range(len(x[0])):  # 18 * 9 * 2
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]


# === 分析訓練資料比例對 RMSE 的影響（Q2 圖表） ===
import matplotlib.pyplot as plt

def train_and_eval(x, y, ratio):
    size = int(len(x) * ratio)
    x_train = x[:size]
    y_train = y[:size]
    dim = x_train.shape[1] + 1
    w = np.ones([dim, 1])
    learning_rate = 2
    iter_time = 5000
    adagrad = np.zeros([dim, 1])
    eps = 1e-7
    lam = 0.05
    x2 = np.concatenate((np.ones([len(x_train), 1]), x_train), axis=1)
    for t in range(iter_time):
        gradient = 2 * np.dot(x2.T, (np.dot(x2, w) - y_train)) + 2 * lam * np.vstack([np.zeros((1, 1)), w[1:]])
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / (np.sqrt(adagrad) + eps)
    y_pred = np.dot(x2, w)
    rmse = np.sqrt(np.mean((y_pred - y_train) ** 2))
    return rmse

ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
rmses = []
for r in ratios:
    rmse = train_and_eval(x, y, r)
    ##print(f\"Training with {int(r*100)}% data: RMSE = {rmse:.5f}\")

    rmses.append(rmse)

plt.plot([int(r*100) for r in ratios], rmses, marker='o')
plt.xlabel('Training Data Percentage (%)')
plt.ylabel('RMSE')
plt.title('Effect of Training Data Size on RMSE')
plt.grid(True)
plt.savefig("/Users/nakaonki/Documents/DM/training_size_vs_rmse.png")
plt.show()



# 隨機打散 X 和 Y
randomize = np.arange(len(x))

# 初始化參數
dim = x.shape[1] + 1  # 加上 bias 的維度
w = np.ones([dim, 1])
learning_rate = 2
iter_time = 5000
adagrad = np.zeros([dim, 1])
eps = 1e-7
lam = 0.01 # L2 正則化係數 λ，可自由調整

for t in range(iter_time):
    # 打散資料順序
    np.random.shuffle(randomize)
    x_shuf = x[randomize]
    y_shuf = y[randomize]

    # 加上 bias 項（最前面補一個 1）
    x2 = np.concatenate((np.ones([len(x_shuf), 1]), x_shuf), axis=1).astype(float)

    # 正則化項（bias 不做懲罰）
    reg_term = 2 * lam * np.vstack([np.zeros((1, 1)), w[1:]])

    # 梯度計算：損失函數 + 正則化
    gradient = 2 * np.dot(x2.T, np.dot(x2, w) - y_shuf) + reg_term
    adagrad += gradient ** 2

    # 參數更新
    w = w - learning_rate * gradient / (np.sqrt(adagrad) + eps)

    # 每 100 次印出 loss
    loss = np.sqrt(np.sum(np.power(np.dot(x2, w) - y_shuf, 2)) / len(x_shuf))  # RMSE
    if t % 100 == 0:
        print(f"{t}: loss = {loss:.6f} | λ = {lam}")
np.save('weight.npy', w)

# 导入测试数据test.csv
#testdata = pd.read_csv('./test.csv', header = None, encoding = 'big5')
testdata = pd.read_csv('/Users/nakaonki/Downloads/test.csv', header=None, encoding='big5')
test_data = testdata.iloc[:, 2:]
#test_data[test_data == 'NR'] = 0
test_data = test_data.copy()  # 避免 slice 問題
# **將 `#, *, x, A` 轉換為 NaN**
test_data.replace(["#", "*", "x", "A"], np.nan, inplace=True)

# **將 NaN 值填補為 0**
test_data.fillna(0, inplace=True)

# **確保所有數據轉為數值**
test_data = test_data.apply(pd.to_numeric, errors='coerce')

# **再一次填補 NaN，確保沒有遺漏**
test_data.fillna(0, inplace=True)
test_data = test_data.to_numpy()
test_x1 = np.empty([244, 18*9], dtype = float)
test_x = np.empty([244, 18*9*2], dtype = float)
for i in range(244):
    test_x1 = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1).astype(float)
    # 同样在这里加入test x的二次项
    test_x[i, : 18 * 9] = test_x1
    test_x[i, 18 * 9:] = np.power(test_x1 , 2)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([244, 1]), test_x), axis = 1).astype(float)

# 对test的x进行预测，得到预测值ans_y
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
# 對 test 的 x（完整特徵）做預測 → Model 1
w1 = np.load('weight.npy')
ans_y1 = np.dot(test_x, w1)

#  建立 Model 2：只使用 PM2.5（第 9 row）與 PM10（第 8 row）訓練的模型 

# 重新取出簡單特徵 x2, y2
x2 = np.empty([12 * 471, 2 * 9], dtype=float)
y2 = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            pm25 = month_data[month][9, day * 24 + hour: day * 24 + hour + 9]
            pm10 = month_data[month][8, day * 24 + hour: day * 24 + hour + 9]
            x2[month * 471 + day * 24 + hour, :9] = pm25
            x2[month * 471 + day * 24 + hour, 9:] = pm10
            y2[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]

# 標準化 x2
mean_x2 = np.mean(x2, axis=0)
std_x2 = np.std(x2, axis=0)
for i in range(len(x2)):
    for j in range(len(x2[0])):
        if std_x2[j] != 0:
            x2[i][j] = (x2[i][j] - mean_x2[j]) / std_x2[j]

# Adagrad 訓練簡單模型
dim2 = x2.shape[1] + 1
w2 = np.ones([dim2, 1])
learning_rate = 2
iter_time = 5000
adagrad2 = np.zeros([dim2, 1])
eps = 1e-7
lam2 = 0.01  # 也加正則化

randomize2 = np.arange(len(x2))
for t in range(iter_time):
    np.random.shuffle(randomize2)
    x2_shuf = x2[randomize2]
    y2_shuf = y2[randomize2]
    x2_full = np.concatenate((np.ones([len(x2_shuf), 1]), x2_shuf), axis=1).astype(float)

    reg_term2 = 2 * lam2 * np.vstack([np.zeros((1, 1)), w2[1:]])
    gradient2 = 2 * np.dot(x2_full.T, np.dot(x2_full, w2) - y2_shuf) + reg_term2
    adagrad2 += gradient2 ** 2
    w2 = w2 - learning_rate * gradient2 / (np.sqrt(adagrad2) + eps)

# 🔹🔹 對 test.csv 做簡單特徵轉換 🔹🔹
test_x2 = np.empty([244, 2 * 9], dtype=float)
for i in range(244):
    pm25 = test_data[9 + 18 * i, :]
    pm10 = test_data[8 + 18 * i, :]
    test_x2[i, :9] = pm25
    test_x2[i, 9:] = pm10

# 標準化 test_x2
for i in range(len(test_x2)):
    for j in range(len(test_x2[0])):
        if std_x2[j] != 0:
            test_x2[i][j] = (test_x2[i][j] - mean_x2[j]) / std_x2[j]
test_x2 = np.concatenate((np.ones([244, 1]), test_x2), axis=1).astype(float)

# Model 2 預測 
ans_y2 = np.dot(test_x2, w2)

#  集成模型結果（加權平均）
ans_y = 0.7 * ans_y1 + 0.3 * ans_y2

# 後處理：負數設為 0，並四捨五入
for i in range(244):
    if(ans_y[i][0] < 0):
        ans_y[i][0] = 0
    else:
        ans_y[i][0] = np.round(ans_y[i][0])

# 輸出 submit.csv（使用 ensemble 結果）
import csv
output_path = "/Users/nakaonki/Documents/DM/submit.csv"
with open(output_path, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['index', 'answer']
    print(header)
    csv_writer.writerow(header)
    for i in range(244):
        row = ['index_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)

'''
# 加一个预处理<0的都变成0
for i in range(244):
    if(ans_y[i][0]<0):
        ans_y[i][0]=0
    else:
        ans_y[i][0]=np.round(ans_y[i][0])


import csv
# 指定輸出文件的絕對路徑
output_path = "/Users/nakaonki/Documents/DM/submit.csv"

with open(output_path, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['index', 'answer']
    print(header)
    csv_writer.writerow(header)
    for i in range(244):
        row = ['index_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
'''