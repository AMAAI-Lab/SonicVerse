from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import ClapModel, ClapProcessor

from sonicverse.model_utils import MultiTaskType
from sonicverse.data_tools import load_audio
from sonicverse.modalities.base_modality import Modality
from sonicverse.modalities.projectors import (
    build_mlp_vector_projector, build_mt_vector_projector, MultiTaskModel
)

import json

OUTPUT_EMB_SIZE = 512


class CLAPAudioModule(nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.processor = None

        self.load_model()

    def load_model(self):
        self.model = ClapModel.from_pretrained(self.model_name_or_path)
        self.processor = ClapProcessor.from_pretrained(self.model_name_or_path)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, audios) -> torch.Tensor:
        embs = []
        for audio_features in audios:
            features = self.model.get_audio_features(
                input_features=audio_features["input_features"].to(torch.float32),
                is_longer=audio_features["is_longer"],
            )
            embs.append(features)
        embs = torch.stack(embs)
        return embs.view(-1, 1, OUTPUT_EMB_SIZE)

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device


class CLAPAudioModality(Modality):
    def __init__(
        self,
        model_name_or_path: str = "laion/clap-htsat-fused",
        num_projector_layers: int = 2,
        num_tokens_output: int = 10,
        use_multi_task: int = MultiTaskType.NO_MULTI_TASK,
        tasks_config: str = None
    ):
        self.model_name_or_path = model_name_or_path
        self.module = CLAPAudioModule(model_name_or_path=self.model_name_or_path)
        self.num_projector_layers = num_projector_layers
        self.num_tokens_output = num_tokens_output
        self.dtype = torch.float32
        self.use_multi_task = use_multi_task
        self.tasks = None
        if self.use_multi_task != MultiTaskType.NO_MULTI_TASK:            
            with open(tasks_config, 'r') as f:
                self.tasks = json.load(f)

        print("Tasks :", self.tasks)

    def build_projector(self, lm_hidden_size: int) -> nn.Module:
        if self.use_multi_task == MultiTaskType.PROJECTED_MULTI_TASK:
            return MultiTaskModel(OUTPUT_EMB_SIZE, self.tasks)
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
            return build_mlp_vector_projector(
                input_hidden_size=OUTPUT_EMB_SIZE,
                lm_hidden_size=lm_hidden_size,
                num_layers=self.num_projector_layers,
                num_tokens=self.num_tokens_output,
            )

    @property
    def name(self) -> str:
        return "audio_clap"

    @property
    def token(self) -> str:
        return "<sound>"

    @property
    def data_key(self) -> str:
        return "sounds"

    @property
    def token_width(self) -> int:
        return self.num_tokens_output

    def to(self, dtype: torch.dtype, device: torch.device) -> "CLAPAudioModality":
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
                    target_sampling_rate=self.module.processor.feature_extractor.sampling_rate,
                )
                audio_processed = self.module.processor(
                    audios=audio_dict["array"],
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
