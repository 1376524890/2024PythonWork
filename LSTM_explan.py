import pandas as pd
import numpy as np
from keras import Sequential
from keras.src.layers import Dense, LSTM, Dropout
from numpy import concatenate
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# 读取数据
# 从 CSV 文件中读取电力负荷数据和天气数据
load_1 = pd.read_csv(r'Area1_Load.csv', encoding='utf-8')
argument_1 = pd.read_csv(r'Area1_Argument.csv', encoding='utf-8')

# 异常值检测与修正
def outlier_detection(data):
    """
    检测并修正数据中的异常值。

    参数:
    data (pd.DataFrame): 需要检测和修正的 DataFrame

    返回:
    pd.DataFrame: 修正后的 DataFrame
    """
    num_rows = data.shape[0]  # 获取数据的行数
    num_cols = data.shape[1]  # 获取数据的列数
    # 检测并修正异常值
    for i in range(0, num_rows - 1):
        for j in range(2, num_cols - 1):
            if abs(data.iloc[i, j] - data.iloc[i, j - 1]) > 0.05 and abs(data.iloc[i, j] - data.iloc[i, j + 1]) > 0.05:
                data.iloc[i, j] = (data.iloc[i, j - 1] + data.iloc[i, j + 1]) / 2  # 用前后两个值的平均值替换异常值
    return data

# 对历史电力负荷数据进行异常值检测和修正
load_1 = outlier_detection(load_1)

# 把 96 个时刻的负荷数据和当天的天气数据合并，分别存放到 csv 中
load = load_1.iloc[:, 1:]  # 提取负荷数据
arg = argument_1.iloc[:, 1:]  # 提取天气数据
for i in range(96):
    load_0 = pd.DataFrame({
        'Load': load.iloc[:, i],  # 负荷数据
        'HighTemp': arg.iloc[:, 0],  # 最高温度
        'LowTemp': arg.iloc[:, 1],  # 最低温度
        'AveTemp': arg.iloc[:, 2],  # 平均温度
        'Humidity': arg.iloc[:, 3],  # 湿度
        'Rainfall': arg.iloc[:, 4],  # 降水量
        'DateType': arg.iloc[:, 5]  # 日期类型
    })
    # 为整理后的数据添加时间戳
    load_0.index = pd.date_range(start='2012-01-01', periods=1106, freq='D')
    load_0.to_csv('Data/Load_%d.csv' % i, index=True, header=True)

# 将数据缩放到 [-1, 1]之间
def scale(train, test):
    """
    将数据缩放到 [-1, 1] 之间。

    参数:
    train (pd.DataFrame): 训练数据
    test (pd.DataFrame): 测试数据

    返回:
    scaler: 缩放器对象
    train_scaled (np.ndarray): 缩放后的训练数据
    test_scaled (np.ndarray): 缩放后的测试数据
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))  # 构造最小-最大规范化缩放器
    train_scaled = scaler.fit_transform(train)  # 对训练数据进行转换
    test_scaled = scaler.transform(test)  # 对测试数据进行转换
    return scaler, train_scaled, test_scaled

# 将时间序列数据转换为监督型学习数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    将时间序列数据转换为监督型学习数据。

    参数:
    data (pd.DataFrame): 时间序列数据
    n_in (int): 输入序列长度
    n_out (int): 输出序列长度
    dropnan (bool): 是否删除包含 NaN 的行

    返回:
    pd.DataFrame: 转换后的监督型学习数据
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []
    # 输入序列（t-n, ... ,t-1）
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 输出序列，即预测序列（t,t+1, ... ,t+n）
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # 合并
    agg = concat(cols, axis=1)
    agg.columns = names
    # 删除具有缺失值 NaN 值的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# 初始化保存预测结果的列表
result = []

# 由于电力负荷每天呈现出相同的周期性，所以对于每一个时刻，都是基于前7天的历史负荷值气象数据和日期类型进行预测
# 循环遍历 96 次完成未来一天内 96 个时刻电力负荷的预测
for i in range(96):
    print("现在在预测第%d 个时刻" % i)
    # 读取前面保存的电力负荷和天气合并之后的 csv 文件
    load = pd.read_csv('Data/Load_%d.csv' % i, encoding='utf-8')
    load = load.iloc[:, 1:]
    # 划分训练集与测试集，可自定义训练集与测试集的比例
    split_rate = 0.8
    train_size = int(load.shape[0] * 0.8)
    train = load.iloc[0:train_size, :]
    test = load.iloc[train_size:, :]
    # 将数据统一映射到[-1,1]的区间内，以提高收敛速度
    scaler, train_scaled, test_scaled = scale(train, test)
    # 用前 7 天的七个特征预测下一天同一时刻的负荷
    # 将训练集转换为监督型学习数据
    train_supervised = series_to_supervised(train_scaled, 7, 1)
    train_X, train_y = train_supervised.iloc[:, 0:49], train_supervised.iloc[:, 49]
    train_X = train_X.values.reshape(train_X.shape[0], 7, 7)
    # 将测试集转换为监督型学习数据
    test_supervised = series_to_supervised(test_scaled, 7, 1)
    test_X, test_y = test_supervised.iloc[:, 0:49], test_supervised.iloc[:, 49]
    test_X = test_X.values.reshape(test_X.shape[0], 7, 7)
    # 搭建 LSTM 模型
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    # model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_X, train_y, epochs=50, batch_size=32, verbose=2, shuffle=False)
    # 使用刚刚训练的 LSTM 模型进行预测
    prd = model.predict(test_X)
    # 逆缩放时 shape 应与原来的大小一致，所以需将预测结果和部分测试集数据组合后再逆缩放
    # 将预测的电力负荷值进行逆缩放
    test_X = test_X.reshape((test_X.shape[0] * test_X.shape[1], test_X.shape[2]))
    inv_prd = concatenate((prd, test_X[0:len(prd), 1:]), axis=1)
    inv_prd = scaler.inverse_transform(inv_prd)
    inv_prd = inv_prd[:, 0]
    # 将测试集中的实际电力负荷值进行逆缩放
    test_y = test_y.values.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[0:len(test_y), 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # 将实际值与预测值合并，方便后续查看
    prd_act = np.concatenate((inv_y, inv_prd))
    result.append(prd_act)

# 将预测结果保存到 csv 中
# 共 96 行对应着 96 个时刻，每行前一半的列表示实际值，后一半的列表示对应的预测值
load_pred = pd.DataFrame(result)
load_pred.to_csv('result.csv', encoding='utf-8')

# 计算平均绝对百分比误差 MAPE
def calculate_mape(actual, pred):
    """
    计算平均绝对百分比误差 (MAPE)。

    参数:
    actual (np.ndarray): 实际值
    pred (np.ndarray): 预测值

    返回:
    float: MAPE 值
    """
    actual = np.array(actual)
    pred = np.array(pred)
    return np.mean(np.abs((actual - pred) / actual))

# 计算平均绝对误差 MAE
def calculate_mae(actual, pred):
    """
    计算平均绝对误差 (MAE)。

    参数:
    actual (np.ndarray): 实际值
    pred (np.ndarray): 预测值

    返回:
    float: MAE 值
    """
    actual = np.array(actual)
    pred = np.array(pred)
    return np.mean(np.abs(actual - pred))

# 计算均方根误差 RMSE
def calculate_rmse(actual, pred):
    """
    计算均方根误差 (RMSE)。

    参数:
    actual (np.ndarray): 实际值
    pred (np.ndarray): 预测值

    返回:
    float: RMSE 值
    """
    actual = np.array(actual)
    pred = np.array(pred)
    squared_errors = (actual - pred) ** 2
    return np.sqrt(np.mean(squared_errors))

# 计算误差平方和 SSE
def calculate_sse(actual, pred):
    """
    计算误差平方和 (SSE)。

    参数:
    actual (np.ndarray): 实际值
    pred (np.ndarray): 预测值

    返回:
    float: SSE 值
    """
    actual = np.array(actual)
    pred = np.array(pred)
    squared_errors = (actual - pred) ** 2
    return np.sum(squared_errors)

# 读取刚才保存预测值与对应实际值的 csv 文件
result = pd.read_csv('result.csv', encoding='utf-8')
# 获取列数,行数不用获取共 96 行
column_count = result.shape[1]
# 获取测试集的预测天数
days = int((column_count - 1) / 2)
# 初始化各个评价指标
mape = 0
mae = 0
rmse = 0
sse = 0

# 分别计算测试集中每天 96 个时刻的误差
for i in range(1, days + 1):
    actual = result.iloc[:, i]
    pred = result.iloc[:, i + days]
    mape += calculate_mape(actual, pred)
    mae += calculate_mae(actual, pred)
    rmse += calculate_rmse(actual, pred)
    sse += calculate_sse(actual, pred)
# 输出整个测试集的平均误差
print("MAPE 为" + str(mape / days) + ",MAE 为" + str(mae / days) + ",RMES 为" + str(rmse / days) + ",SSE 为" + str(sse / days))

# 随便选取测试集中的一天画实际值与预测值的对比图
# 选择不同的列对应着不同天的数据，测试集每天的预测误差可能有大有小，对比图呈现出的效果可能也会有所不同
actual = result.iloc[:, 1]
lstm = result.iloc[:, 1 + days]
plt.plot(actual, label='Actual')
plt.plot(lstm, label='LSTM', linestyle='--')

# 由于对每个时刻单独预测，相邻时刻之间的数据可能会比较分散，所以使用高斯滤波对预测数据进行平滑处理
lstm = gaussian_filter(lstm, sigma=1)
plt.xlabel('Time')
plt.xticks([0, 12, 24, 36, 48, 60, 72, 84], ['0', '3', '6', '9', '12', '15', '18', '21'])
plt.ylabel('Load')
plt.legend()
plt.show()
