import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from typing import Dict
import numpy as np

class CNN(nn.Module):
    def __init__(self, input_channels = 25, num_class=15):
        super(CNN, self).__init__()
        self.aggregator = nn.Parameter(torch.randn((input_channels, 1,1), dtype=torch.float))
        self.input_channels = input_channels

        # init bn
        self.bn_init = nn.BatchNorm2d(1)

        # layer 1
        self.conv_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.mp_1 = nn.MaxPool2d((2, 4))

        # layer 2
        self.conv_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.mp_2 = nn.MaxPool2d((2, 4))

        # layer 3
        self.conv_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.mp_3 = nn.MaxPool2d((2, 4))

        # layer 4
        self.conv_4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(128)
        self.mp_4 = nn.MaxPool2d((3, 5))

        # layer 5
        self.conv_5 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(64)
        self.mp_5 = nn.MaxPool2d((3, 3))

        # classifier
        self.dense = nn.Linear(640, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        aggregator_weights = F.softmax(self.aggregator)
        # aggregator_weights = aggregator_weights.view(self.input_channels, 1)
        # print("0 x shape : ")
        x = (x * aggregator_weights).sum(dim=0)

        # print("aggregator_output shape ", x.shape)

        x = x.unsqueeze(0).unsqueeze(0)

        # print("1 x shape ", x.shape)
        # init bn
        x = self.bn_init(x)
        # print("2 x shape ", x.shape)

        # layer 1
        x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))
        # print("3 x shape ", x.shape)

        # layer 2
        x = self.mp_2(nn.ELU()(self.bn_2(self.conv_2(x))))
        # print("4 x shape ", x.shape)

        # layer 3
        x = self.mp_3(nn.ELU()(self.bn_3(self.conv_3(x))))
        # print("5 x shape ", x.shape)

        # layer 4
        x = self.mp_4(nn.ELU()(self.bn_4(self.conv_4(x))))
        # print("6 x shape ", x.shape)

        # layer 5
        x = self.mp_5(nn.ELU()(self.bn_5(self.conv_5(x))))
        # print("7 x shape ", x.shape)

        # classifier
        x = x.view(x.size(0), -1)
        # print("8 x shape ", x.shape)
        x = self.dropout(x)
        # print("9 x shape ", x.shape)
        logit = nn.Sigmoid()(self.dense(x))
        # print("logit shape ", logit.shape)

        return logit


class MLP(nn.Module):
    def __init__(self, input_channels=25, num_class=15):
        super(MLP, self).__init__()
        self.aggregator = nn.Parameter(torch.randn((input_channels, 1,1), dtype=torch.float))
        self.input_channels = input_channels

        self.hidden_layer_1 = nn.Linear(768, 512)
        self.output = nn.Linear(512, num_class)
        self.dropout = nn.Dropout(p=0.2)
        self.loss = self.get_loss() # can return a dict of losses

    def forward(self, x):
        """
        x: (B, L, T, H)
        T=#chunks, can be 1 or several chunks
        """

        weights = F.softmax(self.aggregator, dim=1)
        x = (x * weights).sum(dim=1)

        x = x.mean(-2)

        x = self.hidden_layer_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        return self.output(x)

    def get_loss(self):
        return nn.BCEWithLogitsLoss()

class MLPBackbone(nn.Module):
    def __init__(self, input_features=768, hidden_dim=512):
        super(MLPBackbone, self).__init__()

        self.hidden_layer_1 = nn.Linear(input_features, hidden_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.loss = self.get_loss() # can return a dict of losses

    def forward(self, x):
        """
        x: (B, L, T, H)
        T=#chunks, can be 1 or several chunks
        """

        x = self.hidden_layer_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x

    def get_loss(self):
        return nn.BCEWithLogitsLoss()

class MLPShared(nn.Module):
    def __init__(self, input_channels=25, num_class=15):
        super(MLPShared, self).__init__()
        self.aggregator = nn.Parameter(torch.randn((input_channels, 1,1), dtype=torch.float))
        self.input_channels = input_channels

        self.hidden_layer_1 = nn.Linear(512, 256)
        self.output = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.2)
        self.loss = self.get_loss() # can return a dict of losses

    def forward(self, x):
        """
        x: (B, L, T, H)
        T=#chunks, can be 1 or several chunks
        """

        weights = F.softmax(self.aggregator, dim=1)
        x = (x * weights).sum(dim=1)

        x = x.mean(-2)

        x = self.hidden_layer_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        return self.output(x)

    def get_loss(self):
        return nn.BCEWithLogitsLoss()

class MLPAggTaskHead(nn.Module):
    def __init__(self, input_channels: int, input_size: int, output_size: int, use_aggregator: bool, use_time_average: bool, use_sigmoid: bool, use_transpose: bool, num_layers: int, hidden_dim: int, width: int):
        super(MLPAggTaskHead, self).__init__()
        if use_aggregator:
            self.aggregator = nn.Parameter(torch.randn((input_channels), dtype=torch.float))
        self.use_aggregator = use_aggregator
        self.use_time_average = use_time_average
        self.use_transpose = use_transpose
        self.use_sigmoid = use_sigmoid
        self.input_channels = input_channels
        self.output_size = output_size
        self.width = width

        if self.width > 1:
            self.layers = nn.ModuleList()
            for i in range(self.width):
                mlp_layers = [nn.GELU()]
                mlp_layers += self._create_mlp_layers(input_size, output_size, num_layers, hidden_dim)
                if self.use_sigmoid: mlp_layers += [nn.Sigmoid()]
                self.layers.append(nn.Sequential(*mlp_layers))
        else:
            mlp_layers = [nn.GELU()]
            mlp_layers += self._create_mlp_layers(input_size, output_size, num_layers, hidden_dim)
            if self.use_sigmoid: mlp_layers += [nn.Sigmoid()]
            self.layers = nn.Sequential(*mlp_layers)

    def _create_mlp_layers(self, input_size, output_size, num_layers, hidden_dim):
        if num_layers >=2:
            layers = [nn.Linear(input_size, hidden_dim)]
            layers.append(nn.GELU())
            if num_layers > 2:
                for _ in range(1, num_layers - 2):
                    layers += [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU()
                    ]
            layers.append(nn.Linear(hidden_dim, output_size))
        else:
            layers = [nn.Linear(input_size, output_size)]
        return layers


    def forward(self, x):
        if self.use_transpose:
            x = x.transpose(1, 0)
        if self.use_time_average:
            x = x.mean(-2)
        if self.use_aggregator:
            aggregator_weights = F.softmax(self.aggregator)
            aggregator_weights = aggregator_weights.view(self.input_channels, 1)
            aggregator_output = (x * aggregator_weights).sum(dim=0)
            aggregator_output = aggregator_output.unsqueeze(dim=0)
            # print("Agg output ", aggregator_output.shape)
        else:
            aggregator_output = x

        if self.width > 1:
            if (self.input_channels < 1):
                return torch.cat([layer(aggregator_output.unsqueeze(dim=0)) for layer in self.layers], dim=-2)
            else:
                return torch.cat([layer(aggregator_output.unsqueeze(dim=0)).squeeze(dim=0) for layer in self.layers], dim=-2)
        else:
            if (self.input_channels < 1):
                return self.layers(aggregator_output.unsqueeze(dim=0))
            else:
                return self.layers(aggregator_output.unsqueeze(dim=0)).squeeze()


class MultiTaskModel(nn.Module):
    def __init__(self, tasks: Dict):
        super(MultiTaskModel, self).__init__()
        self.tasks = tasks
        for task_name, task_head in self.tasks["task_heads"].items():
            setattr(self, task_name, MLP(13, task_head["output_size"]))
            if task_name in self.tasks["task_projectors"].keys():
                task_projector = tasks["task_projectors"][task_name]
                setattr(self, task_name + "_projector", MLPAggTaskHead(task_projector["input_channels"], task_projector["input_size"], task_projector["output_size"], task_projector["use_aggregator"], task_projector["use_time_average"], task_projector["use_sigmoid"], task_projector["use_transpose"], task_projector["num_layers"], task_projector["hidden_size"], task_projector["width"]))

    def forward(self, x):
        task_head_outputs = {}
        task_projector_outputs = []

        backbone_output = x

        for task_name in self.tasks["task_heads"]:
            if task_name != "lmm_projector":
                task_head_outputs[task_name] = getattr(self, task_name)(backbone_output)
                if task_name in self.tasks["task_projectors"].keys():
                    task_projector_outputs.append(getattr(self, task_name + "_projector")(task_head_outputs[task_name]))
            else:
                task_projector_outputs.append(getattr(self, task_name)(backbone_output))

        if len(task_projector_outputs) > 0:
            task_projector_outputs_unsqueezed = [task_projector_output.unsqueeze(0) for task_projector_output in task_projector_outputs]
            task_head_outputs["projectors"] = torch.cat(task_projector_outputs_unsqueezed, dim=-2)

        return task_head_outputs

class MultiTaskSharedModel(nn.Module):
    def __init__(self, tasks: Dict):
        super(MultiTaskSharedModel, self).__init__()
        self.tasks = tasks
        self.use_backbone = False
        if "backbone" in self.tasks.keys():
            self.use_backbone = True
        if self.use_backbone: self.backbone = MLPBackbone(768, 512)
        for task_name, task_head in self.tasks["task_heads"].items():
            if task_name != "lmm_projector":
                setattr(self, task_name, MLPShared(13, task_head["output_size"]))
            else:
                setattr(self, task_name, MLPAggTaskHead(task_head["input_channels"], task_head["input_size"], task_head["output_size"], task_head["use_aggregator"], task_head["use_time_average"], task_head["use_sigmoid"], task_head["use_transpose"], task_head["num_layers"], task_head["hidden_size"], task_head["width"]))
            if task_name in self.tasks["task_projectors"].keys():
                task_projector = tasks["task_projectors"][task_name]
                setattr(self, task_name + "_projector", MLPAggTaskHead(task_projector["input_channels"], task_projector["input_size"], task_projector["output_size"], task_projector["use_aggregator"], task_projector["use_time_average"], task_projector["use_sigmoid"], task_projector["use_transpose"], task_projector["num_layers"], task_projector["hidden_size"], task_projector["width"]))

    def forward(self, x):
        task_head_outputs = {}
        task_projector_outputs = []

        if self.use_backbone:
            backbone_output = self.backbone(x)
        else:
            backbone_output = x

        #print("Output shape ", backbone_output.shape)
        for task_name in self.tasks["task_heads"]:
            #print("task namee ", task_name)
            if task_name != "lmm_projector":
                task_head_outputs[task_name] = getattr(self, task_name)(backbone_output)
                if task_name in self.tasks["task_projectors"].keys():
                    task_projector_outputs.append(getattr(self, task_name + "_projector")(task_head_outputs[task_name]))
            else:
                llm_input = x
                if self.tasks["task_heads"][task_name]["use_backbone_output"]:
                    llm_input = backbone_output
                task_projector_outputs.append(getattr(self, task_name)(llm_input))

        if len(task_projector_outputs) > 0:
            task_projector_outputs_unsqueezed = [task_projector_output.unsqueeze(0) for task_projector_output in task_projector_outputs]
            task_head_outputs["projectors"] = torch.cat(task_projector_outputs_unsqueezed, dim=-2)

        return task_head_outputs



