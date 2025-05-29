from sonicverse.model_utils import MultiTaskType
from sonicverse.modalities.vision_clip import (
    CLIPVisionModality,
    OUTPUT_LAYER as CLIP_POOL_LAYER,
)
from sonicverse.modalities.imagebind import ImageBindModality
from sonicverse.modalities.document_gte import DocumentGTEModality
from sonicverse.modalities.audio_whisper import WhisperAudioModality
from sonicverse.modalities.audio_clap import CLAPAudioModality
from sonicverse.modalities.video_xclip import XCLIPVideoModality
from sonicverse.modalities.audio_descript import DescriptAudioModality
from sonicverse.modalities.audio_mert import MERTAudioModality

MODALITY_BUILDERS = {
    "vision_clip": lambda: [CLIPVisionModality()],
    "vision_clip_pool": lambda: [
        CLIPVisionModality(feature_layer=CLIP_POOL_LAYER, num_tokens_output=10)
    ],
    "audio_whisper": lambda: [
        WhisperAudioModality(
            num_tokens_output=10, model_name_or_path="openai/whisper-small"
        )
    ],
    "audio_mert": lambda use_multi_task=MultiTaskType.NO_MULTI_TASK, tasks_config=None :[MERTAudioModality(use_multi_task=use_multi_task, tasks_config=tasks_config, num_tokens_output=60, hidden_dim=32, num_conv_layers = 3, num_mlp_layers = 2)],
    "audio_clap": lambda use_multi_task=MultiTaskType.NO_MULTI_TASK, tasks_config=None :[CLAPAudioModality(use_multi_task=use_multi_task, tasks_config=tasks_config, num_tokens_output=20)],
    "audio_descript": lambda use_multi_task=MultiTaskType.NO_MULTI_TASK, tasks_config=None : [DescriptAudioModality(use_multi_task=use_multi_task, tasks_config=tasks_config, num_projector_conv_layers=1, num_projector_mlp_layers=1, num_tokens_output=60, codebooks=96)],
    "video_xclip": lambda: [XCLIPVideoModality(num_tokens_output=10)],
    "imagebind": lambda: [ImageBindModality()],
    "document_gte": lambda: [DocumentGTEModality()],
    "document_gte_x16": lambda: [DocumentGTEModality(num_tokens_output=32)],
}
