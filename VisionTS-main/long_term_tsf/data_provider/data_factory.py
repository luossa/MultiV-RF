from data_provider.data_loader import  Dataset_stock
from torch.utils.data import DataLoader
import glob
data_dict = {
    'STOCK': Dataset_stock,
}
data_path = glob.glob('/gemini/code/SP500/SP500_global_200/*.csv')
data_path = data_path
def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'STOCK':
        data_set = Data(
            data_path=data_path,
            window_size = args.window_size,
            window_stride=args.window_stride,
            flag = flag,
            feature=args.features,
            target=args.target,
            task_name = args.task_name
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

    return data_set, data_loader
