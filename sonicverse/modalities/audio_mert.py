from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, AutoModel

from sonicverse.model_utils import MultiTaskType
from sonicverse.data_tools import load_audio
from sonicverse.modalities.base_modality import Modality
from sonicverse.modalities.projectors import (
    build_mlp_vector_projector, build_mt_vector_projector, build_multi_layer_cnn_mlp_projector, MultiTaskModel
)
from sonicverse.modalities.multi_task_projector_shared import MultiTaskSharedModel

import json

OUTPUT_EMB_CHANNELS = 768 #1024
OUTPUT_EMB_SIZE = 760
OUTPUT_FEATURE_LAYERS = 13 #25

cache_dir="/home/ubuntu/.cache/"

class MERTAudioModule(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.processor = None

        self.load_model()

    def load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True, cache_dir=cache_dir)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name_or_path,trust_remote_code=True, cache_dir=cache_dir)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, audios) -> torch.Tensor:
        embs = []
        for audio_features in audios:
            outputs = self.model(**audio_features.to(torch.float32), output_hidden_states=True)
            features = torch.stack(outputs.hidden_states).squeeze()
            embs.append(features)
        embs = torch.stack(embs)
        embs = embs.squeeze()
        padding_needed = OUTPUT_EMB_SIZE - embs.shape[1]
        embs = torch.nn.functional.pad(embs, (0, 0, 0, padding_needed, 0, 0))
        return embs

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device


class MERTAudioModality(Modality):
    def __init__(
        self,
        model_name_or_path: str = "m-a-p/MERT-v1-95M",
        num_tokens_output: int = 10,
        hidden_dim: int = 32,
        num_conv_layers: int = 5,
        num_mlp_layers: int = 5,
        use_multi_task: MultiTaskType = MultiTaskType.NO_MULTI_TASK,
        tasks_config: str = None
    ):
        self.model_name_or_path = model_name_or_path
        self.module = MERTAudioModule(model_name_or_path=self.model_name_or_path)
        self.num_tokens_output = num_tokens_output
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.num_mlp_layers = num_mlp_layers
        self.dtype = torch.float32
        self.use_multi_task = use_multi_task
        self.tasks = None
        if self.use_multi_task != MultiTaskType.NO_MULTI_TASK:            
            with open(tasks_config, 'r') as f:
                self.tasks = json.load(f)

        print("Tasks :", self.tasks)

    # all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
    # print(all_layer_hidden_states.shape) # [25 layer, Time steps, 1024 feature_dim]
    # time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
    # print(time_reduced_hidden_states.shape) # [25, 1024]

    def build_projector(self, lm_hidden_size: int) -> nn.Module:
        if self.use_multi_task == MultiTaskType.PROJECTED_MULTI_TASK:
            projector = MultiTaskSharedModel(self.tasks)
            print("projector ", projector)
            return projector
        elif self.use_multi_task == MultiTaskType.SIMPLE_MULTI_TASK:
            return build_mt_vector_projector(
            # return build_mlp_vector_projector(
                input_hidden_size=OUTPUT_EMB_SIZE,
                lm_hidden_size=lm_hidden_size,
            #     num_layers=self.num_projector_layers,
            #     num_tokens=self.num_tokens_output,
            # )
                tasks = self.tasks
            )
            # )["llm_projector"]
        else:
            return build_multi_layer_cnn_mlp_projector(
                input_channels = OUTPUT_EMB_CHANNELS,
                input_size = OUTPUT_EMB_SIZE,
                num_feature_layers= OUTPUT_FEATURE_LAYERS,
                lm_hidden_size = lm_hidden_size,
                num_tokens = self.num_tokens_output,
                hidden_dim = self.hidden_dim,
                num_conv_layers = self.num_conv_layers,
                num_mlp_layers = self.num_mlp_layers
            )

    @property
    def name(self) -> str:
        return "audio_mert"

    @property
    def token(self) -> str:
        return "<sound>"

    @property
    def data_key(self) -> str:
        return "sounds"

    @property
    def token_width(self) -> int:
        return self.num_tokens_output

    def to(self, dtype: torch.dtype, device: torch.device) -> "MERTAudioModality":
        self.dtype = dtype
        self.module.to(device=device)
        return self

    def preprocess_rows(self, rows: List[Dict]) -> List[Optional[Dict]]:
        row_values = []
        for row in rows:
            audios = []
            for audio_dict in row[self.data_key]:
                audio_dict = load_audio(
                    audio_dict,
                    target_sampling_rate=self.module.processor.sampling_rate,
                )
                audio_processed = self.module.processor(
                    audio_dict["array"],
                    return_tensors="pt",
                    sampling_rate=audio_dict["sampling_rate"],
                )
                audios.append(audio_processed)
            row_values.append(audios)
        return row_values

    @torch.no_grad()
    def forward(self, encoded_values: List[torch.Tensor]) -> List[torch.Tensor]:
        audio_features = []
        for audio_batch in encoded_values:
            audio_features.append(self.module.forward(audio_batch).to(dtype=self.dtype))
        return audio_features
