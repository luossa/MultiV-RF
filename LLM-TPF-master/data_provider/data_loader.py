import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class Dataset_stock(Dataset):
    def __init__(self, data_path, window_size, window_stride,flag='train',
                 feature='S', target='OT', task_name='long_term_forecast', scale=True, timeenc=0,
                 seasonal_patterns=None):
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.stride = window_stride
        self.set_type = type_map[flag]
        self.window_size = window_size
        self.feature = feature
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.data_path = data_path
        self.task_name = task_name
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.data = [] 
        self.tot_len_per_file = []  
        self.scaler = StandardScaler()
        self.scaler1 = StandardScaler()
        all_train_data_feature = []
        all_train_data_target = []
        for data_path in tqdm(self.data_path):
            df_raw = pd.read_csv(data_path)
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.window_size, len(df_raw) - num_test - self.window_size]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            
            # border1s = [0, int(len(df_raw) * 0.8) - 1, int(len(df_raw) * 0.9) - 1]
            # border2s = [int(len(df_raw) * 0.8) - 1, int(len(df_raw) * 0.9) - 1, len(df_raw) - 1]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            
            df_data = df_raw[[self.feature, self.target]]
#             print(df_raw[border1:border2]['Date'].iloc[0])

# # 打印最后一行
#             print(df_raw[border1:border2]['Date'].iloc[-1])
            # if self.task_name == 'long_term_forecast':
            if self.scale:
                train_data_feature = df_data[border1s[0]:border2s[0]][self.feature]
                train_data_target = df_data[border1s[0]:border2s[0]][self.target]
                all_train_data_feature.append(train_data_feature.values)
                if self.task_name == 'long_term_forecast':
                    all_train_data_target.append(train_data_target.values)

                #     self.scaler.fit(train_data_feature.values.reshape(-1, 1))
                #     self.scaler1.fit(train_data_target.values.reshape(-1, 1))
                #     df_data.loc[:, self.feature] = self.scaler.transform(df_data[self.feature].values.reshape(-1, 1)).ravel()
                #     df_data.loc[:, self.target] = self.scaler1.transform(df_data[self.target].values.reshape(-1, 1)).ravel()
                #     data = df_data
                # else:
                #     data = df_data
        if self.scale:
            # 合并所有训练数据
            all_train_data_feature = np.concatenate(all_train_data_feature, axis=0).reshape(-1, 1)
            self.scaler.fit(all_train_data_feature)  # 在全局训练数据上拟合特征的scaler

            #测试回归不进行归一化的结果
            # if self.task_name == 'long_term_forecast':
            #     all_train_data_target = np.concatenate(all_train_data_target, axis=0).reshape(-1, 1)
            #     self.scaler1.fit(all_train_data_target)  # 在全局训练数据上拟合目标的scaler



#----------------------------------------------------------------
        for data_path in tqdm(self.data_path):
            df_raw = pd.read_csv(data_path)
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.window_size, len(df_raw) - num_test - self.window_size]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            
            df_data = df_raw[[self.feature, self.target]]
            
            if self.task_name == 'long_term_forecast':
                if self.scale:
                    # 应用全局归一化
                    df_data.loc[:, self.feature] = self.scaler.transform(df_data[self.feature].values.reshape(-1, 1)).ravel()

                    #target不进行归一化
                    # df_data.loc[:, self.target] = self.scaler1.transform(df_data[self.target].values.reshape(-1, 1)).ravel()
                    data = df_data
                else:
                    data = df_data
                data = data.values
                tot_len = (len(data[border1:border2]) - self.window_size) // self.stride + 1
                if tot_len <= 0:
                    raise ValueError(f"File {data_path} too short for window_size={self.window_size}, stride={self.stride}")
                self.data.append(data[border1:border2])  
                self.tot_len_per_file.append(tot_len)
            else:
                if self.scale:
                    # 应用全局归一化（仅对特征）
                    df_data.loc[:, self.feature] = self.scaler.transform(df_data[self.feature].values.reshape(-1, 1)).ravel()
                    data = df_data
                else:
                    data = df_data
                data = data.values
                tot_len = (len(data[border1:border2]) - self.window_size) // self.stride + 1
                if tot_len <= 0:
                    raise ValueError(f"File {data_path} too short for window_size={self.window_size}, stride={self.stride}")
                self.data.append(data[border1:border2])  
                self.tot_len_per_file.append(tot_len)
        
        self.tot_len = tot_len
        self.total_samples = sum(self.tot_len_per_file)


#----------------------------------------------------------------
        #         data = data.values
        #         tot_len = (len(data[border1:border2]) - self.window_size) // self.stride + 1
        #         if tot_len <= 0:
        #             raise ValueError(f"File {data_path} too short for window_size={self.window_size}, stride={self.stride}")
        #         self.data.append(data[border1:border2])  
        #         self.tot_len_per_file.append(tot_len)
        #     else:
        #         if self.scale:
        #             train_data = df_data[border1s[0]:border2s[0]][self.feature]
        #             self.scaler.fit(train_data.values.reshape(-1, 1))
        #             df_data.loc[:, self.feature] = self.scaler.transform(df_data[self.feature].values.reshape(-1, 1)).ravel()
        #             data = df_data
        #         else:
        #             data = df_data
            
        #         data = data.values
                
        #         tot_len = (len(data[border1:border2]) - self.window_size) // self.stride + 1
        #         if tot_len <= 0:
        #             raise ValueError(f"File {data_path} too short for window_size={self.window_size}, stride={self.stride}")
        #         self.data.append(data[border1:border2])  
        #         self.tot_len_per_file.append(tot_len)
        # self.tot_len = tot_len
        # self.total_samples = sum(self.tot_len_per_file)
    def __getitem__(self, index):
        
        # 用除法定位文件和局部索引
        file_idx = index // self.tot_len  # 文件索引
        local_index = index % self.tot_len  # 文件内局部索引
    
        # 按步幅计算窗口起始位置
        s_begin = local_index * self.stride
        s_end = s_begin + self.window_size
        
        # 确保不越界（可选检查）
        if s_end > len(self.data[file_idx]):
            raise ValueError(f"Window end {s_end} exceeds data length {len(self.data[file_idx])}")

        # 提取窗口数据
        window_data = self.data[file_idx][s_begin:s_end]  # (window_size, 2)
        seq_x = window_data[:, 0:1]  # (window_size, 1)，特征序列
        seq_y = window_data[-1, 1:2]  # (1, 1)，窗口最后一个时间步的分类标签
        file_name = os.path.splitext(os.path.basename(self.data_path[file_idx]))[0]
        # seq_x = torch.tensor(seq_x, dtype=torch.float32)
        # seq_y = torch.tensor(seq_y, dtype=torch.float32)
        # print('seq_y',seq_y)
        return seq_x, seq_y, file_name

    def __len__(self):
        return self.total_samples

