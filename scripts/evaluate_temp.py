from dataclasses import dataclass, field
import logging
from flask import Flask, request, jsonify
import transformers
import torch
from datasets import load_from_disk
from sonicverse.model_utils import MultiTaskType
from sonicverse.training import ModelArguments
from sonicverse.inference import load_trained_lora_model
from sonicverse.data_tools import encode_chat
import evaluate
import random
import bert_score
import os

os.environ['HF_EVALUATE_OFFLINE'] = '1'

PRETRAIN_PHRASES = ["Describe the audio in detail <sound>"]

PRETRAIN_PHRASES_old = [
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
        output_ids[0, encoded_dict["input_ids"].shape[0]:],
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
    for data_point_id in range(10):
        data_point = ds[data_point_id]
        input_json = {"messages": [{"role": "user", "content": content_phrase}], "sounds": data_point["sounds"]}
        output_json = generate(input_json)
        print("Prediction ", output_json["output"])
        print("Reference ", data_point["messages"][1]["content"])
        print()
        print()
        predictions.append(output_json["output"])
        references.append(data_point["messages"][1]["content"])

    # Load evaluation metrics
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")
    
    # Compute BLEU scores
    bleu_results = bleu.compute(predictions=predictions, references=references, max_order=4)
    print(bleu_results)
    #bleu_score = sum(bleu_results[f"bleu{i}"] for i in range(1, 5)) / 4

    # Compute METEOR score
    meteor_results = meteor.compute(predictions=predictions, references=references)
    meteor_score = meteor_results["meteor"]

    # Compute ROUGE-L score
    rouge_results = rouge.compute(predictions=predictions, references=references, rouge_types=["rougeL"])
#    rouge_l_score = rouge_results["rougeL"].mid.fmeasure
    print(rouge_results)

    # Compute BERT-Score
    P, R, F1 = bert_score.score(predictions, references, lang="en", rescale_with_baseline=True)
    bert_score_f1 = F1.mean().item()

    # Print results
    #print(f"BLEU Score: {bleu_score}")
    print(f"METEOR Score: {meteor_score}")
#    print(f"ROUGE-L Score: {rouge_l_score}")
    print(f"BERT-Score F1: {bert_score_f1}")
