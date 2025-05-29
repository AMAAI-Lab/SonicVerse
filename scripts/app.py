from dataclasses import dataclass, field
import logging

from huggingface_hub import login
import os

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not hf_token:
    raise ValueError("Missing HUGGINGFACE_HUB_TOKEN. Set it as a secret in your Space.")

login(token=hf_token)

import gradio as gr
import torch
import transformers
import torchaudio

from openai import OpenAI
client = OpenAI()
MODEL       = "gpt-4"
SLEEP_BETWEEN_CALLS = 1.0

from sonicverse.model_utils import MultiTaskType
from sonicverse.training import ModelArguments
from sonicverse.inference import load_trained_lora_model
from sonicverse.data_tools import encode_chat

CHUNK_LENGTH = 10

@dataclass
class ServeArguments(ModelArguments):
    load_bits: int = field(default=16)
    max_new_tokens: int = field(default=128)
    temperature: float = field(default=0.01)


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

def caption_audio(audio_file):
    chunk_audio_files = split_audio(audio_file, CHUNK_LENGTH)
    chunk_captions = []
    for audio_chunk in chunk_audio_files:
        chunk_captions.append(generate_caption(audio_chunk))

    if len(chunk_captions) > 1:
        audio_name = os.path.splitext(os.path.basename(audio_file))[0]
        long_caption = summarize_song(audio_name, chunk_captions)

        delete_files(chunk_audio_files)

        return long_caption

    else:
        if len(chunk_captions) == 1:
            return chunk_captions[0]
        else:
            return ""

def summarize_song(song_name, chunks):
    prompt = f"""
You are a music critic. Given the following chronological 10‑second chunk descriptions of a single piece, write one flowing, detailed description of the entire song—its structure, instrumentation, and standout moments. Mention transition points in terms of time stamps. If the description of certain chunks does not seem to fit with those for the chunks before and after, treat those as bad descriptions with lower accuracy and do not incorporate the information. Retain concrete musical attributes such as key, chords, tempo.

Chunks for “{song_name} ”:
"""
    for i, c in enumerate(chunks, 1):
        prompt += f"\n {(i - 1)*0} to {i*10} seconds. {c.strip()}"
    prompt += "\n\nFull song description:"

    resp = client.chat.completions.create(model=MODEL,
    messages=[
        {"role": "system", "content": "You are an expert music writer."},
        {"role": "user",   "content": prompt}
    ],
    temperature=0.0,
    max_tokens=1000)
    return resp.choices[0].message.content.strip()

def delete_files(file_paths):
    for path in file_paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
                print(f"Deleted: {path}")
            else:
                print(f"Skipped (not a file or doesn't exist): {path}")
        except Exception as e:
            print(f"Error deleting {path}: {e}")

def split_audio(input_path, chunk_length_seconds):

    waveform, sample_rate = torchaudio.load(input_path)
    num_channels, total_samples = waveform.shape
    chunk_samples = int(chunk_length_seconds * sample_rate)

    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    base, ext = os.path.splitext(input_path)
    output_paths = []

    if (num_chunks <= 1):
        return [input_path]

    for i in range(num_chunks):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, total_samples)
        chunk_waveform = waveform[:, start:end]
        
        output_file = f"{base}_{i+1:03d}{ext}"
        torchaudio.save(output_file, chunk_waveform, sample_rate)
        output_paths.append(output_file)

    return output_paths

def generate_caption(audio_file):
    req_json = {
        "messages": [
            {"role": "user", "content": "Describe the music. <sound>"}
        ],
        "sounds": [audio_file]
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
    fn=caption_audio,
    inputs=gr.Audio(type="filepath", label="Upload an audio file"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="SonicVerse",
    description="Upload an audio file to generate a caption using SonicVerse"
)

if __name__ == "__main__":
    demo.launch()
