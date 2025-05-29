from dataclasses import dataclass, field
import logging

import gradio as gr
import torch
import transformers
import torchaudio

from sonicverse.model_utils import MultiTaskType
from sonicverse.training import ModelArguments
from sonicverse.inference import load_trained_lora_model
from sonicverse.data_tools import encode_chat


@dataclass
class ServeArguments(ModelArguments):
    load_bits: int = field(default=16)
    max_new_tokens: int = field(default=128)
    temperature: float = field(default=0.01)


# Load arguments and model
logging.getLogger().setLevel(logging.INFO)

parser = transformers.HfArgumentParser((ServeArguments,))
serve_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

model, tokenizer = load_trained_lora_model(
    model_name_or_path=serve_args.model_name_or_path,
    model_lora_path=serve_args.model_lora_path,
    load_bits=serve_args.load_bits,
    use_multi_task=MultiTaskType(serve_args.use_multi_task),
    tasks_config=serve_args.tasks_config
)


def generate_caption(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)

    req_json = {
        "audio": {
            "tensor": waveform,
            "sampling_rate": sample_rate,
        }
    }

    encoded_dict = encode_chat(req_json, tokenizer, model.modalities)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=encoded_dict["input_ids"].unsqueeze(0).to(model.device),
            max_new_tokens=serve_args.max_new_tokens,
            use_cache=True,
            do_sample=True,
            temperature=serve_args.temperature,
            modality_inputs={
                m.name: [encoded_dict[m.name]] for m in model.modalities
            },
        )

    outputs = tokenizer.decode(
        output_ids[0, encoded_dict["input_ids"].shape[0]:],
        skip_special_tokens=True
    ).strip()

    return outputs


demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Audio(type="filepath", label="Upload a WAV file"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Audio Caption Generator",
    description="Upload a .wav audio file to generate a caption using a LoRA fine-tuned model."
)

if __name__ == "__main__":
    demo.launch()
