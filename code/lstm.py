import os
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from time import time
from data_loader import DataLoader


# LSTM model
class LSTM(nn.Module):
    def __init__(self, configs, device, loss_function, scaler):
        super(LSTM, self).__init__()
        self.configs = configs
        self.input_size = configs['model']['input_size']
        self.hidden_size = configs['model']['hidden_size']
        self.number_layer = configs['model']['number_layer']
        self.output_size = configs['model']['output_size']
        self.output_layer = configs['model']['output_layer']
        self.batch_size = configs['model']['batch_size']
        self.verbose = configs['model']['verbose']
        self.device = device
        self.loss_function = loss_function
        self.scaler = scaler
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.number_layer,
                            batch_first=True,
                            dropout=0.2)
        self.out = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.output_size) for _ in range(self.output_layer)])

    def forward(self, x):
        h0 = Variable(torch.zeros(self.number_layer, x.size(0), self.hidden_size)).to(self.device)
        c0 = Variable(torch.zeros(self.number_layer, x.size(0), self.hidden_size)).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = torch.cat([layer(out[:, -1, :]) for layer in self.out], dim=1)
        out = out.view(out.size(0), self.output_size, self.output_layer)
        return out

    def train_online(self, x, y, optimizer):
        self.train()
        losses = []
        times = []
        len_batch = math.ceil(x.size(0) / self.batch_size)
        for batch_idx in range(len_batch):
            start = time()
            if self.batch_size * (batch_idx + 1) > x.size(0):
                output = self.forward(x[batch_idx * self.batch_size:])
                target = y[batch_idx * self.batch_size:]
            else:
                output = self.forward(x[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size])
                target = y[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
            loss = self.loss_function(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(batch_idx + 1)
            stop = time()
            duration = stop - start
            if batch_idx % 20 == 0:
                test_loss = self.test()
                losses.append(test_loss.item())
                times.append(duration)
        return losses, times

    def train_online_hsgd(self, x, y, w, optimizer):
        self.train()
        losses = []
        times = []
        loss_list = []
        len_batch = math.ceil(x.size(0) / self.batch_size)
        for batch_idx in range(len_batch):
            start = time()
            if self.batch_size * (batch_idx + 1) > x.size(0):
                output = self.forward(x[batch_idx * self.batch_size:])
                target = y[batch_idx * self.batch_size:]
            else:
                output = self.forward(x[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size])
                target = y[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
            loss = self.loss_function(output, target)
            if len(loss_list) == w:
                loss_list.pop(0)
            loss_list.append(loss)
            sum_loss = sum(loss_list)
            sum_loss = sum_loss / len(loss_list)
            optimizer.zero_grad()
            sum_loss.backward(retain_graph=True)
            optimizer.step(batch_idx + 1)
            stop = time()
            duration = stop - start
            if batch_idx % 20 == 0:
                test_loss = self.test()
                losses.append(test_loss.item())
                times.append(duration)
        return losses, times

    def train_offline(self, x, y, optimizer):
        self.train()
        losses = []
        times = []
        len_batch = math.ceil(x.size(0) / self.batch_size)
        for batch_idx in range(len_batch):
            start = time()
            if self.batch_size * (batch_idx + 1) > x.size(0):
                output = self.forward(x)
                target = y
            else:
                output = self.forward(x[:(batch_idx + 1) * self.batch_size])
                target = y[:(batch_idx + 1) * self.batch_size]
            loss = self.loss_function(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(batch_idx + 1)
            stop = time()
            duration = stop - start
            if batch_idx % 20 == 0:
                test_loss = self.test()
                losses.append(test_loss.item())
                times.append(duration)
        return losses, times

    def test(self):
        # using present model predict the load values from 2010-10 to 2011-12
        # the loss is the average quantile loss of these 15 months
        self.eval()
        online_losses = []
        for i in range(1, 16):
            task = DataLoader(path=os.path.join(self.configs['data']['path']['task' + str(i)]),
                              start=0,
                              cols=self.configs['data']['columns'],
                              time_flag=self.configs['data']['time_flag']['task' + str(i)],
                              scaler=self.scaler)
            x, y = task.get_train(configs=self.configs,
                                  device=self.device,
                                  mode_flag=self.configs['data']['mode']['test'])
            output = self.forward(x)
            # inverse transform output and y to calculate true loss
            output_inv = self.scaler.inverse_transform(output.view(-1, 1).cpu().detach().numpy())
            y_inv = self.scaler.inverse_transform(y.cpu().detach().numpy())
            loss = self.loss_function(torch.tensor(output_inv).view(-1, 1, self.output_layer),
                                      torch.tensor(y_inv).view(-1, 1))
            online_losses.append(loss)
        self.train()
        return sum(online_losses) / 15

    def predict(self, x):
        p = self.forward(x).view(-1, 1, self.output_layer)
        return p
