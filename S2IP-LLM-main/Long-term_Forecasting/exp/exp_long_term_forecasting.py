from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual,adjust_model
from utils.metrics import metric
import torch
import torch.nn as nn
from models import  S2IPLLM
from torch.nn.utils import clip_grad_norm_
from utils.losses import mape_loss, mase_loss, smape_loss
from sklearn.metrics import accuracy_score, f1_score

from transformers import AdamW





from torch.utils.data import Dataset, DataLoader
from torch import optim
import os
import time
import warnings
import numpy as np

from tqdm import tqdm

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'S2IPLLM': S2IPLLM,
            
        }

        self.device = torch.device('cuda:0')
        self.model = self._build_model()
        
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')
        # self.test_data, self.test_loader = self._get_data(flag='test')

        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()

      

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).to(self.device)
        

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # if args.task_name == 'classification':
        #     criterion = nn.CrossEntropyLoss()
        # else:
        #     criterion = nn.MSELoss()
         
        if self.args.task_name == 'long_term_forecast':
            criterion = nn.MSELoss()
    
        else:
            criterion = nn.CrossEntropyLoss()

        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, name) in tqdm(enumerate(vali_loader)):
                if self.args.task_name == 'classification':
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.long().to(self.device)
                    outputs,res = self.model(batch_x,name)
                else:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    outputs,res = self.model(batch_x,name)
                
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        
       
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    

   

    def train(self, setting):

        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            simlarity_losses = []
            mses = []
            maes = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, name) in tqdm(enumerate(self.train_loader),total=len(self.train_loader)):
                iter_count += 1
                self.optimizer.zero_grad()
                if self.args.task_name == 'classification':
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.long().to(self.device)
                    outputs, res = self.model(batch_x, name)
                    # batch_y = batch_y.squeeze()
                else:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    outputs, res = self.model(batch_x, name)
                    # batch_y = batch_y.squeeze()
                # batch_x = batch_x.float().to(self.device)
                # batch_y = batch_y.float().to(self.device)
                
                # outputs, res = self.model(batch_x, name)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)
                # loss = criterion(outputs, batch_y)
                loss = self.criterion(outputs, batch_y)
                    
                train_loss.append(loss.item())
                simlarity_losses.append(res['simlarity_loss'].item())
                loss += self.args.sim_coef*res['simlarity_loss']
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            sim_loss = np.average(simlarity_losses)
            vali_loss = self.vali(self.vali_data, self.vali_loader, self.criterion)
            if self.args.task_name == 'classification':
                test_acc, test_f1 = self.test(self.test_data, self.test_loader, self.criterion, train_loss, vali_loss)
            else:
                test_mse, test_mae = self.test(self.test_data, self.test_loader, self.criterion, train_loss, vali_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Sim Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss,sim_loss))
            
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
            adjust_model(self.model, epoch + 1,self.args)
           

            

    def test(self, test_data, test_loader, criterion, train_loss, vali_loss):

        
      
        preds = []
        trues = []
        folder_path = './test_results/' + self.args.features + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        sim_matrix = []
        input_embedding = []
        prompted_embedding = []
        last_embedding = []


        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,name) in tqdm(enumerate(test_loader)):
                if self.args.task_name == 'classification':
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.long().to(self.device)
                    outputs,res =  self.model(batch_x,name)
                    outputs = torch.argmax(outputs, dim=1)
                else:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    outputs,res =  self.model(batch_x,name)
                
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                # # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                pred = outputs.flatten().cpu()
                true = batch_y.flatten().cpu()
                preds.append(pred)
                trues.append(true)
        if self.args.task_name == 'classification':
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print('preds',preds)
            print('trues',trues)
            accuracy = accuracy_score(trues,preds)
            f1 = f1_score(trues,preds, average='weighted')
            folder_path = './results/' + self.args.task_name + '/' + self.args.features + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            print('train_loss:{}, vali_loss{}, accuracy:{}, f1:{}'.format(train_loss, vali_loss, accuracy, f1))
            f = open(folder_path + f"result_long_term_forecast_{self.args.window_size}.txt", 'a')
            f.write('train_loss:{}, vali_loss:{}, accuracy:{}, f1:{}'.format(train_loss, vali_loss, accuracy, f1))
            f.write('\n')
            f.close()
        else:
            preds = np.array(preds)
            trues = np.array(trues)

            # result save
            folder_path = './results/' + self.args.task_name + '/' + self.args.features + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
    
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('train_loss:{}, vali_loss{}, mse:{}, mae:{}'.format(train_loss, vali_loss, mse, mae))
            f = open(folder_path + "result_long_term_forecast.txt", 'a')
            # f.write(setting + "  \n")
            f.write('train_loss:{}, vali_loss{}, mse:{}, mae:{}'.format(train_loss, vali_loss, mse, mae))
            f.write('\n')
            f.close()
    
            # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            # np.save(folder_path + 'pred.npy', preds)
            # np.save(folder_path + 'true.npy', trues)
        self.model.train()
        if self.args.task_name == 'classification':
            return accuracy, f1
        else:
            return mse, mae
    
