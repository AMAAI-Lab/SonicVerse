from typing import Dict, List, Optional

import torch
import torch.nn as nn
import dac
from audiotools import AudioSignal

from sonicverse.model_utils import MultiTaskType
from sonicverse.data_tools import load_audio_signal
from sonicverse.modalities.base_modality import Modality
from sonicverse.modalities.projectors import (
    build_mlp_vector_projector, build_attentive_cnn_projector, build_cnn_mlp_projector, MultiTaskModel
)

import json

OUTPUT_FRAMES_SIZE = 512
# OUTPUT_EMB_SIZE = 2048
OUTPUT_EMB_CHANNELS = 96

class DescriptAudioModule(nn.Module):
    def __init__(self, model_name_or_path: str, codebooks = 4):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.processor = None
        self.codebooks = codebooks

        self.load_model()

    def load_model(self):
        # self.model = ClapModel.from_pretrained(self.model_name_or_path)
        self.model = dac.DAC.load(self.model_name_or_path)

    def forward(self, audios) -> torch.Tensor:
        embs = []
        for audio_features in audios:
            # print("Audio features sample rate ", audio_features[0].sample_rate)
            x = self.model.preprocess(audio_features[0].audio_data, audio_features[0].sample_rate)
            z, codes, latents, _, _ = self.model.encode(x)

            # print("latents og shape ", latents.shape)
            # If the tensor is larger than desired_shape, crop it
            if latents.shape[2] > OUTPUT_FRAMES_SIZE:
                latents = latents[:, :, :OUTPUT_FRAMES_SIZE]
            # If the tensor is smaller than desired_shape, pad it
            elif latents.shape[2] < OUTPUT_FRAMES_SIZE:
                pad_width = (0, OUTPUT_FRAMES_SIZE - latents.shape[2])
                latents = torch.nn.functional.pad(latents, pad_width)
                # print("Codes new shape ", codes_new.shape)

            # print("latents int shape ", latents.shape)

            latents = latents[0][:self.codebooks]
            
            # print("latents final shape ", latents.shape)

            embs.append(latents)

        embs = torch.stack(embs)

        # output_embs = embs.view(-1, 1, OUTPUT_FRAMES_SIZE*self.codebooks)
        # print("embs post view shape ", output_embs.shape)

        return embs

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device


class DescriptAudioModality(Modality):
    def __init__(
        self,
        model_name_or_path: str = dac.utils.download(model_type="16khz"),
        num_projector_conv_layers: int = 2,
        num_projector_mlp_layers: int = 2,
        num_tokens_output: int = 10,
        codebooks: int = 96,
        use_multi_task: MultiTaskType = MultiTaskType.NO_MULTI_TASK,
        tasks_config: str = None
    ):
        self.model_name_or_path = model_name_or_path
        self.module = DescriptAudioModule(model_name_or_path=self.model_name_or_path, codebooks=codebooks)
        self.num_projector_conv_layers = num_projector_conv_layers
        self.num_projector_mlp_layers = num_projector_mlp_layers
        self.num_tokens_output = num_tokens_output
        self.dtype = torch.float32
        self.codebooks = codebooks
        self.use_multi_task = use_multi_task
        self.tasks = None
        if self.use_multi_task != MultiTaskType.NO_MULTI_TASK:            
            with open(tasks_config, 'r') as f:
                self.tasks = json.load(f)

        print("Tasks :", self.tasks)


    def build_projector(self, lm_hidden_size: int) -> nn.Module:
        if self.use_multi_task == MultiTaskType.PROJECTED_MULTI_TASK:
            projector = MultiTaskModel(OUTPUT_EMB_CHANNELS, 1, True, -1, False, self.tasks)
            print("projector ", projector)
            return projector
        elif self.use_multi_task == MultiTaskType.SIMPLE_MULTI_TASK:
            return build_mt_vector_projector(
            # return build_mlp_vector_projector(
                input_hidden_size=OUTPUT_EMB_CHANNELS,
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
        return "audio_descript"

    @property
    def token(self) -> str:
        return "<sound>"

    @property
    def data_key(self) -> str:
        return "sounds"

    @property
    def token_width(self) -> int:
        return self.num_tokens_output

    def to(self, dtype: torch.dtype, device: torch.device) -> "DescriptAudioModality":
        self.dtype = dtype
        self.module.to(device=device)
        return self

    def preprocess_rows(self, rows: List[Dict]) -> List[Optional[Dict]]:
        row_values = []
        for row in rows:
            audios = []
            for audio_dict in row[self.data_key]:
                audio_dict = load_audio_signal(
                    audio_dict
                )
                audios.append(audio_dict["array"])
            row_values.append(audios)
        return row_values

    @torch.no_grad()
    def forward(self, encoded_values: List[torch.Tensor]) -> List[torch.Tensor]:
        audio_features = []
        for audio_batch in encoded_values:
            audio_features.append(self.module.forward(audio_batch).to(dtype=self.dtype))
        return audio_features
