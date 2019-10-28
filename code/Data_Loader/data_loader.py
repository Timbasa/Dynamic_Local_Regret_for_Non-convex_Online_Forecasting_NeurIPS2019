import pandas as pd
from Data_Loader.reshape_data import reshape_data
from Data_Loader.to_supervised import to_supervised
import torch


# load the file
class DataLoader:
    def __init__(self, path=None, cols=None, start=None, time_flag=None, scaler=None):
        data_frame = pd.read_csv(path)
        data_frame = data_frame[start:]
        self.train = reshape_data(scaler.transform(data_frame.get(cols).values), time_flag)

    def get_train(self, configs, device, mode_flag):
        x, y = to_supervised(configs, self.train, mode_flag)
        x, y = torch.tensor(x, dtype=torch.float32).to(device), \
               torch.tensor(y, dtype=torch.float32).to(device)
        return x, y
