import pandas as pd
import numpy as np
import akshare as ak
import warnings
import talib as ta

pd.options.display.max_rows = None
pd.options.display.max_columns = None
pd.set_option("expand_frame_repr", False)
pd.set_option("display.unicode.ambiguous_as_wide", True)
pd.set_option("display.unicode.east_asian_width", True)
warnings.filterwarnings("ignore")

start_date = 20150601
end_date = 20220901
adj = "hfq"  # None未复权\qfq前复权\hfq后复权
period = "daily"  # daily\weekly\monthly

# sh300 = ak.index_stock_cons_csindex(symbol="000300")
# sh300.to_excel('C:/Users/86136/Desktop/code.xlsx')
sh300 = pd.read_excel("code.xlsx", dtype=object)


def data_operation(data):
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


code_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 88, 89, 90, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299,300]

code_pool = sh300['代码'][code_list]
print(code_pool)
print(len(code_pool))



