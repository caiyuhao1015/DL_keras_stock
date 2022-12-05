from keras.models import load_model
import datetime
import pandas as pd
import numpy as np
import akshare as ak
import warnings
import talib as ta
from sklearn import preprocessing
warnings.filterwarnings("ignore")

pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.set_option("expand_frame_repr", False)
pd.set_option("display.unicode.ambiguous_as_wide", True)
pd.set_option("display.unicode.east_asian_width", True)


class Create_Pool():
    def __init__(self, code_list):
        self.adj = 'hfq'
        self.period = 'daily'
        self.code_list = code_list
        self.cols =["收盘", "成交额", "换手率", "SMA", "upper",
                    "lower", "MA5", "MA10", "MA20", "MACD",
                    "DEA", "DIF", "RSI", "CMO", "K", "D",
                    "ADX", "OBV", "MFI", "TRIX"]

    def get_test_data(self, df, seq_len=19):
        len_df = len(df)
        data_windows = df[len_df-seq_len:len_df]
        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows)
        x = data_windows
        return x

    def normalise_windows(self, window_data):
        min_max_scaler = preprocessing.MinMaxScaler()
        normalised_window = min_max_scaler.fit_transform(window_data)
        return np.array(normalised_window)

    def my_predict(self, code, start, end):
        df = ak.stock_zh_a_hist(symbol=str(code), start_date=start, end_date=end,
                                adjust=self.adj, period=self.period)
        model = load_model("D:/A-Pycharm-File/机器学习/股票预测/saved_models/{code}.h5".format(code=str(code)))
        tmpdata = self.data_operation(df.iloc[:-1,:])
        Yscaler = preprocessing.MinMaxScaler()
        tmpdata["收盘"] = Yscaler.fit_transform(tmpdata["收盘"].values.reshape(-1, 1))
        tmpdata = tmpdata.get(self.cols).values
        data = self.get_test_data(tmpdata)
        normalised_data = data[np.newaxis,]
        pre = model.predict(normalised_data)
        pre = Yscaler.inverse_transform(pre)[0, 0]
        price = df["收盘"].iloc[-2]
        next_price = df['收盘'].iloc[-1]
        ratio = (pre-price)/price
        return round(ratio, 4), price, round(pre, 4), next_price

    def data_operation(self, data):
        close = data['收盘'].values
        # 简单移动平均指标SMA
        data["SMA"] = ta.SMA(close, 5)
        # 布林上中下轨线
        data["upper"], data["middle"], data["lower"] = ta.BBANDS(close, 5, matype=0)
        # 5日移动均线\10日移动均线\0日移动均线
        data["MA5"] = ta.MA(close, timeperiod=5, matype=0)
        data["MA10"] = ta.MA(close, timeperiod=10, matype=0)
        data["MA20"] = ta.MA(close, timeperiod=20, matype=0)
        # 异同移动平均线
        data["MACD"], data["DEA"], data["DIF"] = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        # 相对强弱指数
        data["RSI"] = ta.RSI(close, timeperiod=5)
        # 钱德动量摆动指标
        data["CMO"] = ta.CMO(close, timeperiod=5)
        # 随机指标K和D
        data["K"], data["D"] = ta.STOCH(data['最高'].values, data['最低'].values, close, fastk_period=5,
                                        slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        # 平均趋向指数
        data["ADX"] = ta.ADX(data['最高'].values, data['最低'].values, close, timeperiod=14)
        # 能量潮
        data["OBV"] = ta.OBV(close, data['成交量'])
        # 资金流向指标
        data["MFI"] = ta.MFI(data['最高'].values, data['最低'].values, close, data['成交量'], timeperiod=9)
        # 三重指数平滑移动平均
        data["TRIX"] = ta.TRIX(close, timeperiod=5)
        """
        1.动量指标：ADX、CMO、MACD、MFI
        2.波动率指标：RSI、KD、TRIX
        3.量价指标：OBV
        4.重叠研究指标；SMA、upper、lower、MA5、MA10、MA20
        """
        data = data.fillna(method='bfill')
        return data

    def today_pool(self):
        yeild_ratio = []
        buy_price = []
        predict = []
        next_close = []
        day = datetime.timedelta(days=1)
        month = datetime.timedelta(days=60)
        today_time = datetime.date.today() - day
        start_time = today_time - month
        start_time = str(start_time).replace('-', '')
        today_time = str(today_time).replace('-', '')
        for code in self.code_list:
            ratio, price, pre, next_ = self.my_predict(code, start_time, today_time)
            yeild_ratio.append(ratio)
            buy_price.append(price)
            predict.append(pre)
            next_close.append(next_)
        dataframe = {'code': self.code_list, 'ratio': yeild_ratio,
                     'buy_price': buy_price, 'predict':predict, 'next_close':next_close}
        dataframe = pd.DataFrame(dataframe)
        return dataframe[dataframe['ratio'] > 0]

    def history_pool(self, day):
        yeild_ratio = []
        buy_price = []
        predict = []
        next_close = []
        start_time,today_time = self.last_2month(day)
        for code in self.code_list:
            ratio, price, pre, next_ = self.my_predict(code, start_time, today_time)
            yeild_ratio.append(ratio)
            buy_price.append(price)
            predict.append(pre)
            next_close.append(next_)
        dataframe = {'code': self.code_list, 'ratio': yeild_ratio,
                     'buy_price': buy_price, 'predict':predict, 'next_close':next_close}
        dataframe = pd.DataFrame(dataframe)
        return dataframe[dataframe['ratio'] > 0]

    def last_2month(self, day):
        month = datetime.timedelta(days=60)
        start_time = day - month
        start_time = str(start_time).replace('-', '')
        today_time = str(day).replace('-', '')
        return start_time,today_time





