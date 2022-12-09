import pandas as pd
import numpy as np


def SETUP(id_df, seq_length, seq_cols, mask_value):

    df_mask = pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    df_mask[:] = mask_value
    
    id_df = df_mask.append(id_df,ignore_index=True)
    
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]
    
    start = num_elements-seq_length
    stop = num_elements
    
    lstm_array.append(data_array[start:stop, :])
    
    return np.array(lstm_array)

def FINAL_TOUCHES(predictions):
  results = predictions.flatten()
  df = pd.DataFrame(results)
  df.columns = ['RUL']
  df.index = df.index + 1
  return df


    