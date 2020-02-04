# Model-01
Case to lean in Stocktomorow













import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import DeeperTradeLibrary as dpt
import catboost 
import lightgbm 
df_master = pd.read_csv('GOLD_H4.csv')

def add_data(data):
    df_data = data
    df_volume = df_data['volume'] 

    df = df_data.drop(['volume'], axis=1).copy()

    df['time'] = pd.to_datetime(df.time) 

    df.isna().sum().sum() # เรียกดูค่า NA ว่ามีหรือไม่

    import ta 
    df['rsi14'] = ta.momentum.rsi(df.close, n=14)

    df['volume'] = df_volume

    df['ma5'] = df.close.rolling(5).mean().diff()
    #df['ma13'] = df.close.rolling(13).mean().diff()
    #df['ma34'] = df.close.rolling(34).mean().diff()
    #df['ma89'] = df.close.rolling(89).mean().diff()
    #df['ma233'] = df.close.rolling(233).mean().diff()
    
    df['rsi14-t1'] = df['rsi14'].shift(1)
   
    df['bbh'] = ta.volatility.bollinger_hband(df['close']) - df['close'] # ระยะระหว่างราคาปิดกับ BBH
    df['bbh-t1'] = df['bbh'].shift(1) 

    df['open-close'] = df.open - df.close
    df['open-close-t1'] = df['open-close'].shift(1)

    df['close-open'] = df.close - df.open
    df['close-open-t1'] = df['close-open'].shift(1)

    df.head(50)

    df['vsa'] = df['volume'].diff()*df['close-open-t1']
    df['vsa'] = df['vsa'].shift(1)

    df['(high+low)/2'] = (df.high + df.low)/2
    df['(high+low)/2'] = df['(high+low)/2'].shift(1)

    df['high-low'] = df.high - df.low
    df['high-low-t1'] = df['high-low'].shift(1)
    return df
    
 df = add_data(df_master)  # เรียกใช้ Function
 df['label'] = (df.close.rolling(5, center=True).mean().diff() > 0).astype(int)
 df.dropna(inplace=True)
 df_temp = df.copy()
 df_temp['signal'] = df_temp['label']
 df_temp.dropna(inplace=True)
 from catboost import CatBoostClassifier
 import catboost
 #Load Model
new_model = CatBoostClassifier()
new_model = new_model.load_model('630203_model_01.bob')

































 
 
 
