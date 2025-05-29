from dataclasses import dataclass, field
import logging

from flask import Flask, request, jsonify
import transformers
import torch

from datasets import load_from_disk

from sonicverse.model_utils import MultiTaskType
from sonicverse.training import (
    ModelArguments,
)
from sonicverse.inference import load_trained_lora_model
from sonicverse.data_tools import encode_chat

import evaluate

import random

PRETRAIN_PHRASES = [
    "What is happening in the given music <sound>?",
    "Describe the sound. <sound>",
    "Describe the music. <sound>",
    "<sound> Provide a description of the music.",
    "<sound> Provide a description of the sound.",
    "Can you interpret <sound>?",
    "Please explain what's happening in <sound>",
    "What does <sound> represent?",
    "Could you describe <sound> for me?",
    "What's the content of <sound>?",
    "Can you depict <sound>?",
    "What is <sound>?",
    "In the music clip, <sound>, what is happening?",
    "Provide a description of the music. <sound>",
    "Provide a description of the sound. <sound>",
    "Provide a caption for the sound. <sound>",
    "Provide a caption for the music. <sound>",
]


@dataclass
class ServeArguments(ModelArguments):
    port: int = field(default=8080)
    host: str = field(default="0.0.0.0")
    load_bits: int = field(default=16)
    max_new_tokens: int = field(default=128)
    temperature: float = field(default=0.01)


def generate(input_json):
    encoded_dict = encode_chat(input_json, tokenizer, model.modalities)

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
        output_ids[0, encoded_dict["input_ids"].shape[0] :],
        skip_special_tokens=True,
    ).strip()

    return {"output": outputs}


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser((ServeArguments,))

    serve_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    dataset_path = "/data/musicbench_multitoken_official_split/val"

    ds = load_from_disk(dataset_path)

    model, tokenizer = load_trained_lora_model(
        model_name_or_path=serve_args.model_name_or_path,
        model_lora_path=serve_args.model_lora_path,
        load_bits=serve_args.load_bits,
        use_multi_task=MultiTaskType(serve_args.use_multi_task),
        tasks_config=serve_args.tasks_config
    )

    predictions = []
    references = []
    content_phrase = random.choice(PRETRAIN_PHRASES)
    for data_point_id in range(100):
        data_point = ds[data_point_id]
        # print("datapoint", data_point)
        input_json={"messages": [{"role": "user", "content": content_phrase}], "sounds": data_point["sounds"]}
        output_json = generate(input_json)

        print("Prediction ",output_json["output"])
        print("Reference ", data_point["messages"][1]["content"])
        print()
        print()
        predictions.append(output_json["output"])
        references.append(data_point["messages"][1]["content"])

    sacrebleu = evaluate.load("sacrebleu")
    sacrebleu_results=sacrebleu.compute(predictions=predictions, references=references)

    print(sacrebleu_results["score"])
