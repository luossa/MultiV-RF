from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, name) in enumerate(tqdm(vali_loader, desc='vali')):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.squeeze(1).long().to(self.device)
                outputs = self.model(batch_x)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.save_dir, self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            pbar = tqdm(train_loader)
            for i, (batch_x, batch_y, name) in enumerate(pbar):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.squeeze(1).long().to(self.device)
                
                outputs = self.model(batch_x)
                
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                pbar.set_description("epoch: {0} | loss: {1:.7f}".format(epoch + 1, loss.item()))
                if (i + 1) % 100 == 0:
                    tqdm.write("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    tqdm.write('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_acc, test_f1 = self.test()
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test acc: {4:.7f} Test f1: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_acc, test_f1))
            folder_path = self.args.save_dir
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            f = open(folder_path + "result_classification.txt", 'a')
            f.write('Epoch:{}, Train Loss:{}, Vali Loss:{}, acc:{}, f1:{}'.format(epoch + 1, train_loss, vali_loss, test_acc, test_f1))
            f.write('\n')
            f.close()
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        if os.path.isfile(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            print("Test without train!",best_model_path)

        return self.model

    def test(self):
        test_data, test_loader = self._get_data(flag='test')
        # if test:
        #     print('loading model')
        #     self.model.load_state_dict(torch.load(os.path.join(f'{self.args.save_dir}/checkpoints/' + setting, 'checkpoint.pth')))

        # valid_loss_path = os.path.join(f'{self.args.save_dir}/checkpoints/' + setting, 'valid_loss.json')
        # if os.path.isfile(valid_loss_path):
        #     with open(valid_loss_path) as f:
        #         valid_loss = json.load(f)
        #         best_valid_loss = valid_loss['best_valid_loss']
        #         best_valid_epoch = valid_loss['best_valid_epoch']
        # else:
        #     best_valid_loss = -1
        #     best_valid_epoch = -1
        
        preds = []
        trues = []
        folder_path = f'{self.args.save_dir}/test_results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, name) in enumerate(tqdm(test_loader, desc='test')):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                
                outputs = self.model(batch_x)
                outputs = torch.argmax(outputs, dim=1)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                pred = outputs.flatten()
                true = batch_y.flatten()

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        accuracy = accuracy_score(trues,preds)
        f1 = f1_score(trues,preds, average='weighted')

        return accuracy, f1
