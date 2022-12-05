import os
import json
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from core.data_processor import DataLoader
from core.model import Model
from sklearn.preprocessing import MinMaxScaler
close_in_91 = pd.read_excel('9月1日收盘价.xlsx', dtype=object)


# 绘图展示结果
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.savefig('results.png')


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.legend()
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
    # plt.show()
    plt.savefig('results_multiple.png')


def residuals_calculate(predicted_data,true_data):
    mse = np.sum((true_data - predicted_data) ** 2) / len(true_data)
    rmse = sqrt(mse)
    mae = np.sum(np.abs(true_data - predicted_data)) / len(true_data)
    mape = np.sum(np.abs((true_data - predicted_data)/true_data)) / len(true_data) * 100

    predicted_data = predicted_data.squeeze()
    true_data = true_data.squeeze()
    dtrue = np.diff(true_data)
    dpredicted = np.diff(predicted_data)
    true_trend = (dtrue >= 0)
    predicted_trend = (dpredicted >= 0)
    acc = sum(true_trend == predicted_trend) / len(predicted_trend)
    print(" mae:",round(mae,4),"mse:",round(mse,4)," rmse:",
          round(rmse,4),"acc:{a}% mape:{b}%".format(a=round(acc,4),b=round(mape,4)))


def main(filename):
    # 读取所需参数
    # 更换地址
    configs = json.load(open('D:/A-Pycharm-File/机器学习/股票预测/model_json/lstm.json', 'r', encoding='utf-8'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    # 读取数据
    dataframe =pd.read_excel(r"D:/A-Pycharm-File/机器学习/股票预测/data/{fname}.xlsx".
                             format(fname=filename)).get(configs['data']['columns'])
    Y_scaler = MinMaxScaler(feature_range=(0, 1))
    dataframe['收盘'] = Y_scaler.fit_transform(dataframe['收盘'].values.reshape(-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataframe = scaler.fit_transform(dataframe)

    data = DataLoader(
        dataframe,
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    # 创建RNN模型
    model = Model()
    mymodel = model.build_model(configs)
    # 加载训练数据
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # 训练模型
    model.train(x, y, epochs=configs['training']['epochs'],
                batch_size=configs['training']['batch_size'],
                save_dir=configs['model']['save_dir'],
                filename=filename)

    # 测试结果
    x_test, y_test = data.get_test_data(seq_len=configs['data']['sequence_length'],
                                        normalise=configs['data']['normalise'])
    pre = model.predict_point_by_point(x_test)
    pre = Y_scaler.inverse_transform(pre.reshape(-1, 1))
    y_test = Y_scaler.inverse_transform(y_test.reshape(-1, 1))
    plot_results(pre, y_test)
    return pre, y_test


if __name__ == '__main__':
    """输入i"""
    i = 32
    code = close_in_91['代码'][i]
    predict, true = main(code)
    residuals_calculate(predict, true)
    print('股票代码为：'+code)
