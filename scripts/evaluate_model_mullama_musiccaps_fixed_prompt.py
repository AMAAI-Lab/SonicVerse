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
from tqdm import tqdm

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score as meteor_scorer
from nltk.tokenize import wordpunct_tokenize
import json
from bert_score import score
from tqdm.auto import tqdm

import yaml

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


PRETRAIN_PHRASES_OLD = [
    "Describe the audio in detail"
]

PRETRAIN_PHRASES = [
    # "What is happening in the given music <sound>?",
    # "Describe the sound. <sound>",
    # "Describe the music. <sound>",
    # "<sound> Provide a description of the music.",
    # "<sound> Provide a description of the sound.",
    # "Can you interpret <sound>?",
    # "Please explain what's happening in <sound>",
    # "What does <sound> represent?",
    # "Could you describe <sound> for me?",
    # "What's the content of <sound>?",
    # "Can you depict <sound>?",
    # "What is <sound>?",
    # "In the music clip, <sound>, what is happening?",
    # "Provide a description of the music. <sound>",
    # "Provide a description of the sound. <sound>",
    # "Provide a caption for the sound. <sound>",
    "Provide a caption for the music. <sound>",
]

random.seed(1234)

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

def evaluate(candidates, mult_reference):
    rouge_score, bleu_score, bleu4_score, meteor_score = 0, 0, 0, 0
    for ref, cand in tqdm(zip(mult_reference, candidates), total=len(mult_reference)):
        rouge_score += scorer.score(ref, cand)['rougeL'].recall
        cand_split = wordpunct_tokenize(cand)
        ref_split = wordpunct_tokenize(ref)
        bleu4_score += sentence_bleu([ref], cand, weights=(0.0, 0.0, 0.0, 1.0))
        bleu_score += sentence_bleu([ref], cand)
        meteor_score += meteor_scorer([ref_split], cand_split)
    rouge_score, bleu_score, bleu4_score, meteor_score = rouge_score / (len(candidates)), bleu_score / (len(candidates)), bleu4_score / (len(candidates)), meteor_score / (len(candidates))
    P, R, F1 = score(candidates, mult_reference, lang="en", verbose=True)
    bert_score = R.mean().item()
    #print(f"Model: {model_name}")
    print(f"BLEU Score: {bleu_score}")
    print(f"BLEU-4 Score: {bleu4_score}")
    print(f"METEOR Score: {meteor_score}")
    print(f"ROUGE Score: {rouge_score}")
    print(f"BERT Score: {bert_score}")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = transformers.HfArgumentParser((ServeArguments,))
    serve_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    dataset_path = "/data/musiccaps/musiccaps_val"
    ds = load_from_disk(dataset_path)
    shuffled_ds = ds.shuffle(seed=1234)
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
#    for data_point_id in range(len(ds)):
    print("len(ds)", len(ds))
    for data_point in tqdm(ds):
        print(data_point["audio"])
        # data_point = ds[data_point_id]
        input_json = {"messages": [{"role": "user", "content": content_phrase}], "sounds": [data_point["audio"]]}
        output_json = generate(input_json)
        print("Prediction ", output_json["output"])
        print("Reference ", data_point["caption"])
        print()
        print()
        predictions.append(output_json["output"])
        references.append(data_point["caption"])

    pairs = {"predictions": predictions, "references": references}

    evaluate(predictions, references)

    with open('/experiments/captioning/mert_tasks_separate_backbone_train_001_ft/checkpoint_1985_test/musiccaps_val_fixed_prompt.yaml', 'w') as file:
        yaml.dump(pairs, file, default_flow_style=False)
