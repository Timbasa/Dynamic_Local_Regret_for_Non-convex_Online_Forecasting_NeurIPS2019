from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# create the scaler to normalise the datasets
def Scaler(path, cols, start):
    data_frame = pd.read_csv(path)
    data_frame = data_frame[start:]
    data_train = data_frame.get(cols).values
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    return scaler
