from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from sklearn.metrics import accuracy_score, f1_score
from utils.cmLoss import cmLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
warnings.filterwarnings('ignore')

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.flag = 0

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args, self.device).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        param_dict = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' in n], "lr": 1e-4},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' not in n], "lr": self.args.learning_rate}
        ]
        model_optim = optim.Adam([param_dict[1]], lr=self.args.learning_rate)
        loss_optim = optim.Adam([param_dict[0]], lr=self.args.learning_rate)

        return model_optim, loss_optim

    def _select_criterion(self):
        criterion = cmLoss(self.args.feature_loss, 
                           self.args.output_loss, 
                           self.args.task_loss, 
                           self.args.task_name, 
                           self.args.feature_w, 
                           self.args.output_w, 
                           self.args.task_w)
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim, loss_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)

        for epoch in range(self.args.train_epochs):
            # 热力图用

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # self.flag = 1
            for i, (batch_x, batch_y, name) in tqdm(enumerate(train_loader),desc="Processing_train{}".format(iter_count), total=len(train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                loss_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                
                outputs_dict = self.model(batch_x, name)
                # self.flag = 0
                loss = criterion(outputs_dict, batch_y)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # self.flag = 1
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                loss_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_accuracy, test_f1 = self.test(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test accuracy: {4:.7f} Test f1: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_accuracy, test_f1))
            # results save
            folder_path = './test_results/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            f = open(folder_path + "result_classification_{}_{}.txt".format(self.args.features, self.args.window_size), 'a')
            f.write('train_loss:{}, vali_loss:{}, test_accuracy:{}, test_f1:{}'.format(train_loss, vali_loss, test_accuracy, test_f1))
            f.write('\n')
            f.write('\n')
            f.close()
            
            if self.args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        print(best_model_path)
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []

        self.model.in_layer.eval()
        self.model.out_layer.eval()
        self.model.time_proj.eval()
        self.model.text_proj.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, name) in tqdm(enumerate(vali_loader),desc="Processing_vali{0}"):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                outputs = self.model(batch_x,name)

                outputs_ensemble = outputs['outputs_time']
                
                batch_y = batch_y.to(self.device)

                pred = outputs_ensemble.detach().cpu()
                true = batch_y.detach().cpu()

                loss = F.cross_entropy(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)

        self.model.in_layer.train()
        self.model.out_layer.train()
        self.model.time_proj.train()
        self.model.text_proj.train()

        return total_loss

    def test(self, test_data, test_loader, criterion):

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,name) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                outputs = self.model(batch_x,name)

                outputs_ensemble = outputs['outputs_time']
                outputs_ensemble = torch.argmax(outputs_ensemble, dim=1)
                pred = outputs_ensemble.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                pred = pred.flatten()
                true = true.flatten()
                preds.append(pred)
                trues.append(true)

        # preds = np.array(preds).reshape(-1)
        # trues = np.array(trues).reshape(-1)
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        accuracy = accuracy_score(trues,preds)
        f1 = f1_score(trues,preds, average='weighted')
        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.model.train()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return accuracy, f1
