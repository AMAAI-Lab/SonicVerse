import torch.nn as nn
import torch
from typing import Dict
import numpy as np

import torch.nn.functional as F

def build_patch_mlp_projector(
    input_hidden_size: int, lm_hidden_size: int, num_layers: int
) -> nn.Module:
    modules = [nn.Linear(input_hidden_size, lm_hidden_size)]
    for _ in range(1, num_layers):
        modules.append(nn.GELU())
        modules.append(nn.Linear(lm_hidden_size, lm_hidden_size))
    return nn.Sequential(*modules)


class _MLPVectorProjector(nn.Module):
    def __init__(
        self, input_hidden_size: int, lm_hidden_size: int, num_layers: int, width: int
    ):
        super(_MLPVectorProjector, self).__init__()
        self.mlps = nn.ModuleList()
        for _ in range(width):
            mlp = [nn.Linear(input_hidden_size, lm_hidden_size)]
            for _ in range(1, num_layers):
                mlp.append(nn.GELU())
                mlp.append(nn.Linear(lm_hidden_size, lm_hidden_size))
            self.mlps.append(nn.Sequential(*mlp))

    def forward(self, x):
        output = torch.cat([mlp(x) for mlp in self.mlps], dim=-2)
        return output

def build_mlp_vector_projector(
    input_hidden_size: int, lm_hidden_size: int, num_layers: int, num_tokens: int
):
    return _MLPVectorProjector(
        input_hidden_size, lm_hidden_size, num_layers, num_tokens
    )

class MLPBackbone(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_layers: int, hidden_dim: int):
        super(MLPBackbone, self).__init__()
        self.output_size = output_size
        mlp_layers = self._create_mlp_layers(input_size, output_size, num_layers, hidden_dim)
        self.layers = nn.Sequential(*mlp_layers)       

    def _create_conv_layers(self, input_channels, num_conv_layers, hidden_dim):
        layers = []
        for _ in range(num_conv_layers):
            layers += [
                nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ]
            input_channels = hidden_dim
        return layers

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
        return self.layers(x)

class MLPTaskHead(nn.Module):
    def __init__(self, backbone: nn.Module, input_size: int, output_size: int, num_layers: int, hidden_dim: int, width: int = 1):
        super(MLPTaskHead, self).__init__()
        self.backbone = backbone
        self.width = width
        if width > 1:
            self.layers = nn.ModuleList()
            for i in range(width):
                mlp_layers = [nn.GELU()]
                mlp_layers += self._create_mlp_layers(input_size, output_size, num_layers, hidden_dim)
                self.layers.append(nn.Sequential(*mlp_layers))
        else:
            mlp_layers = [nn.GELU()]
            mlp_layers += self._create_mlp_layers(input_size, output_size, num_layers, hidden_dim)
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

    def _create_conv_layers(self, input_channels, num_conv_layers, hidden_dim):
        layers = []
        for _ in range(num_conv_layers):
            layers += [
                nn.Conv2d(in_channels = input_channels, out_channels = hidden_dim, kernel_size=(3,3), stride=1, padding=1),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ]
            input_channels = hidden_dim
        return layers

    def forward(self, x):
        output = self.backbone.forward(x)
        if self.width > 1:
            return torch.cat([layer(output) for layer in self.layers], dim=-2)
        else:
            return self.layers(output)

class MLPTaskModule(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_layers: int, hidden_dim: int, width: int = 1):
        super(MLPTaskModule, self).__init__()
        self.width = width
        if width > 1:
            self.layers = nn.ModuleList()
            for i in range(width):
                mlp_layers = [nn.GELU()]
                mlp_layers += self._create_mlp_layers(input_size, output_size, num_layers, hidden_dim)
                self.layers.append(nn.Sequential(*mlp_layers))
        else:
            mlp_layers = [nn.GELU()]
            mlp_layers += self._create_mlp_layers(input_size, output_size, num_layers, hidden_dim)
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

    def _create_conv_layers(self, input_channels, num_conv_layers, hidden_dim):
        layers = []
        for _ in range(num_conv_layers):
            layers += [
                nn.Conv2d(in_channels = input_channels, out_channels = hidden_dim, kernel_size=(3,3), stride=1, padding=1),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ]
            input_channels = hidden_dim
        return layers

    def forward(self, x):
        if self.width > 1:
            return torch.cat([layer(x) for layer in self.layers], dim=-2)
        else:
            return self.layers(x)


class MultiTaskModel(nn.Module):
    def __init__(self, input_hidden_size: int, input_channels: int, time_average: bool, time_dimension: int, use_aggregator: bool, tasks: Dict):
        super(MultiTaskModel, self).__init__()
        self.tasks = tasks
        self.time_average = time_average
        self.time_dimension = time_dimension
        self.use_aggregator = use_aggregator
        if self.use_aggregator:
            if (time_average):
                self.aggregator = nn.Parameter(torch.randn((input_channels, 1), dtype = torch.float))
            else:
                self.aggregator = nn.Parameter(torch.randn((input_channels, 1, 1), dtype = torch.float))

        self.backbone = MLPBackbone(input_hidden_size, self.tasks["backbone"]["output_size"], self.tasks["backbone"]["num_layers"], self.tasks["backbone"]["hidden_size"])
        for task_name, task_head in self.tasks["task_heads"].items():
            setattr(self, task_name, MLPTaskModule(self.tasks["backbone"]["output_size"], task_head["output_size"], task_head["num_layers"], task_head["hidden_size"], task_head["width"]))
            if task_name in self.tasks["task_projectors"].keys():
                task_projector = tasks["task_projectors"][task_name]
                setattr(self, task_name + "_projector", MLPTaskModule(task_head["output_size"], task_projector["output_size"], task_projector["num_layers"], task_projector["hidden_size"], task_projector["width"]))

    def forward(self, x):
        task_head_outputs = {}
        task_projector_outputs = []

        if self.time_average:
            x = x.mean(self.time_dimension)
        if self.use_aggregator:
            aggregator_weights = F.softmax(self.aggregator, dim=0)
            aggregator_output = (x * aggregator_weights).sum(dim=0)
            aggregator_output = aggregator_output.unsqueeze(0)
        else:
            aggregator_output = x

        backbone_output = self.backbone(aggregator_output)

        for task_name in self.tasks["task_heads"]:
            if task_name != "lmm_projector":
                task_head_output = getattr(self, task_name)(backbone_output)
                min_val = torch.min(task_head_output)
                max_val = torch.max(task_head_output)

                normalized_task_head_output = (task_head_output - min_val) / (max_val - min_val)
                task_head_outputs[task_name] = normalized_task_head_output
                if task_name in self.tasks["task_projectors"].keys():
                    task_projector_outputs.append(getattr(self, task_name + "_projector")(task_head_outputs[task_name]))
            else:
                task_projector_outputs.append(getattr(self, task_name)(backbone_output))

        task_projector_outputs_unsqueezed = [task_projector_output.unsqueeze(0) for task_projector_output in task_projector_outputs]
        if len(task_projector_outputs_unsqueezed) > 0:
            task_head_outputs["projectors"] = torch.cat(task_projector_outputs_unsqueezed, dim=-2)

        return task_head_outputs


def build_mt_vector_projector(
        input_hidden_size: int, lm_hidden_size: int, tasks: Dict
):
    projector = nn.ModuleDict()
    projector["backbone"] = MLPBackbone(input_hidden_size, tasks["backbone"]["output_size"], tasks["backbone"]["num_layers"], tasks["backbone"]["hidden_size"])
    for task_name, task_head in tasks["task_heads"].items():
        projector[task_name] = MLPTaskHead(projector["backbone"], task_head["hidden_size"], task_head["output_size"], task_head["num_layers"], task_head["hidden_size"], task_head["width"])

    return projector

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        energy = torch.tanh(self.linear_in(x))
        attention_scores = torch.softmax(self.linear_out(energy), dim=1)
        context_vector = torch.sum(attention_scores * x, dim=1)
        return context_vector

class _CNNAttentionTokenizer(nn.Module):
    def __init__(self, input_channels, output_size, width, hidden_dim, num_conv_layers):
        super(_CNNAttentionTokenizer, self).__init__()
        self.width = width
        self.cnns = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for _ in range(width):
            cnn = self._create_conv_layers(input_channels, num_conv_layers)
            self.cnns.append(cnn)
            attention = [Attention(hidden_dim, 125)]
            linear_input_size = hidden_dim
            attention.append(nn.Linear(linear_input_size, output_size))
            self.attentions.append(nn.Sequential(*attention))


    def _create_conv_layers(self, input_channels, num_conv_layers):
        layers = []
        in_channels = input_channels
        for _ in range(num_conv_layers):
            layers += [
                nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ]
            in_channels = 64
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        for token in range(self.width):
            # Input shape: (batch_size, input_channels, sequence_length)
            token_output = self.cnns[token](x)  # Apply convolutional layers
            token_output = token_output.permute(0, 2, 1)  # Reshape for attention mechanism (batch_size, sequence_length, input_dim
            token_output = self.attentions[token](token_output)  # Apply attention mechanism
            outputs.append(token_output)
        output = torch.cat(outputs, dim=-2)
        output = torch.stack([output])
        return output

def build_attentive_cnn_projector(
    input_channels: int, lm_hidden_size: int, num_tokens: int, hidden_dim: int, num_layers: int
    ):
    return _CNNAttentionTokenizer(input_channels, lm_hidden_size, num_tokens, hidden_dim, num_layers)

class _CNNMLPProjector(nn.Module):
    def __init__(self, input_channels, input_size, output_size = 4096, width = 5, hidden_dim = 64, num_conv_layers = 1, num_mlp_layers = 2):
        super(_CNNMLPProjector, self).__init__()
        self.width = width
        self.cnnmlps = nn.ModuleList()
        for _ in range(self.width):
            cnnmlp = self._create_conv_layers(input_channels, num_conv_layers, hidden_dim)
            cnnmlp.append(nn.Flatten())
            cnn_output_size = hidden_dim*((input_size + 2*1 - 3*num_conv_layers) // (2**num_conv_layers) + 1)
            cnnmlp.append(nn.Linear(cnn_output_size, output_size))
            cnnmlp.append(nn.GELU())
            cnnmlp += self._create_mlp_layers(output_size, output_size, num_mlp_layers, output_size)
            self.cnnmlps.append(nn.Sequential(*cnnmlp))

    def _create_conv_layers(self, input_channels, num_conv_layers, hidden_dim):
        layers = []
        for _ in range(num_conv_layers):
            layers += [
                nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ]
            input_channels = hidden_dim
        return layers

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
        return torch.stack([torch.cat([cnnmlp(x) for cnnmlp in self.cnnmlps], dim=-2)])

def build_cnn_mlp_projector(
    input_channels: int, input_size: int, lm_hidden_size: int, num_tokens: int, hidden_dim: int, num_conv_layers: int, num_mlp_layers: int
    ):
    return _CNNMLPProjector(input_channels, input_size, lm_hidden_size, num_tokens, hidden_dim, num_conv_layers, num_mlp_layers)

class _MultiLayeredCNNMLPProjector(nn.Module):
    def __init__(self, input_channels, input_size, num_feature_layers, output_size = 4096, width = 5, hidden_dim = 64, num_conv_layers = 1, num_mlp_layers = 2):
        super(_MultiLayeredCNNMLPProjector, self).__init__()
        self.width = width
        self.num_feature_layers = num_feature_layers
        self.cnnmlps = nn.ModuleList()
        for _ in range(self.width*self.num_feature_layers):
            cnnmlp = self._create_conv_layers(input_channels, num_conv_layers, hidden_dim)
            cnnmlp += [nn.GELU()]
            cnnmlp += self._create_mlp_layers(input_size, output_size, num_mlp_layers, output_size)
            self.cnnmlps.append(nn.Sequential(*cnnmlp))

    def _create_conv_layers(self, input_channels, num_conv_layers, hidden_size):
        layers = []

        if input_channels >= hidden_size:
            hidden_dim = int(input_channels/2)
        else:
            hidden_dim = hidden_size

        layers += [nn.Conv1d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1), nn.GELU()]
        if num_conv_layers > 2:
            for _ in range(num_conv_layers - 2):
                if hidden_dim/2 >= hidden_size:
                    output_dim = int(hidden_dim/2)
                else:
                    output_dim = hidden_size
                layers += [
                    nn.Conv1d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                ]
                hidden_dim = output_dim
        layers += [nn.Conv1d(in_channels=hidden_dim, out_channels=1, kernel_size=3, stride=1, padding=1)]
        return layers

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
        print("X SHAPE ", x.shape)
        inp_feature_layers = []
        for feature_id in range(self.num_feature_layers):
            in_feat_layer = x[feature_id].unsqueeze(0).permute(0,2,1)
            inp_feature_layers.append(in_feat_layer)

        outputs = []
        for layer_count in range(self.width*self.num_feature_layers):
            feature_id = int(layer_count/self.width)
            outputs+=[self.cnnmlps[layer_count](inp_feature_layers[feature_id])]
        
        return torch.cat(outputs, dim=-2)


def build_multi_layer_cnn_mlp_projector(
    input_channels: int, input_size: int, num_feature_layers: int, lm_hidden_size: int, num_tokens: int, hidden_dim: int, num_conv_layers: int, num_mlp_layers: int
    ):
    assert(num_tokens % num_feature_layers == 0)
    width = int(num_tokens/num_feature_layers)
    return _MultiLayeredCNNMLPProjector(input_channels, input_size, num_feature_layers, lm_hidden_size, width, hidden_dim, num_conv_layers, num_mlp_layers)

