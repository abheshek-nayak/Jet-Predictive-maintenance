import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def BASIC_CLEAN(df):
        
        df = df.dropna(axis=1,how='all')
        index_names = ['unit','time_cycle']
        settings = ["setting_"+str(i) for i in range(1,4)]
        sensor_names = ['S{}'.format(i) for i in range(1,22)] 
        column_names = index_names+settings+sensor_names
        old_names = df.columns
        df2=df.rename(columns=dict(zip(old_names, column_names)))
        dropped_features = ['setting_1','setting_2','setting_3','S1','S5','S6','S10','S13','S14','S16','S18','S19']
        df2.drop(dropped_features,axis=1,inplace=True)
        return df2

    
    
