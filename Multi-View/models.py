from Text.models import TimeLLM
from Vison.long_term_tsf.models import VisionTS
import torch
import torch.nn as nn

class Fusion_model(nn.Module):

    def __init__(self, configs):
        super(Fusion_model, self).__init__()
        self.text_model = TimeLLM.Model(configs)
        self.vison_model = VisionTS.Model(configs)
        self.task_name = configs.task_name
        self.label_l = configs.label_l
        self.num_class = configs.num_class
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.out_layer = nn.Linear(60, self.label_l)
        elif self.task_name == 'classification':
            self.out_layer = nn.Linear(60, self.num_class)
        else:
            raise NotImplementedError

    def forward(self, x_enc, name):
        text_feature = self.text_model(x_enc, name)
        vison_feature = self.vison_model(x_enc)
        fusion_feature = torch.cat((text_feature, vison_feature), dim=1)
        result = self.out_layer(fusion_feature)
        return result