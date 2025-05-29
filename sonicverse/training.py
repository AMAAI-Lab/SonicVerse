from typing import Optional, List
from dataclasses import field, dataclass
import logging
import subprocess
import pathlib
import torch
import shutil
import glob
import os
import json

import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import Trainer

from sonicverse.training_data import (
    DataArguments,
    LMMDataset,
    DataCollatorForSupervisedLMMDataset,
)
from sonicverse.model_utils import (
    make_model_lora,
    get_peft_state,
    get_peft_state_non_lora,
    fix_tokenizer,
    MultiTaskType
)
from sonicverse.modalities.base_modality import Modality


README_TEMPLATE = """
---
license: apache-2.0
base_model: {base_model}
dataset: {dataset}
tags:
  - finetuned
  - multimodal
inference: false
---

These are weights for a version of `{base_model}` finetuned for multimodal applications. 

### Modalities

{modalities}

### Usage

GitHub: https://github.com/sshh12/multi_token (includes training scripts and basic inference server)

### Dataset

{dataset} ({num_examples} examples)

```
{dataset_example}
```

### Training Device(s)

```
{training_devices_dump}
```


### Model

```
{repr_model}
```

"""


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    pretrain_projectors: bool = field(default=False)
    pretrained_projectors_path: Optional[str] = field(default=None)
    pretrained_projectors_config: Optional[str] = field(default=None)
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.1")
    model_cls: str = field(default="MistralLMMForCausalLM")
    modality_builder: str = field(default="vision_clip")
    use_multi_task: int = field(default=MultiTaskType.PROJECTED_MULTI_TASK)
    tasks_config: str = field(default="configs/tasks.json")
    model_lora_path: Optional[str] = field(default="amaai-lab/SonicVerse")


class LMMTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self._save_extras(output_dir)

        super(LMMTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        self._save_extras(output_dir)
        super(LMMTrainer, self)._save(output_dir, state_dict)
        for unused_dir in glob.iglob(os.path.join(output_dir, "global_step*")):
            shutil.rmtree(unused_dir)

    def _save_extras(self, output_dir: Optional[str] = None):
        self.model.config.save_pretrained(output_dir)

        task_names = []
        for m in self.model.modalities:
            task_names += m.tasks["task_heads"].keys()

        non_lora_state_dict = get_peft_state_non_lora(self.model.named_parameters(), task_names)
        torch.save(
            non_lora_state_dict,
            os.path.join(output_dir, "non_lora_trainables.bin"),
        )


def _get_training_devices_dump() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=gpu_name,gpu_bus_id,vbios_version", "--format=csv"]
    )
    return out.decode("utf-8").strip()


def train_for_modalities(
    model_cls,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    train_data_args: DataArguments,
    evaluation_data_args: DataArguments,
    modalities: List[Modality],
):
    for m in modalities:
        m.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    fix_tokenizer(tokenizer)

    train_dataset = LMMDataset(train_data_args, tokenizer, modalities)
    evaluation_dataset = LMMDataset(evaluation_data_args, tokenizer, modalities)
    collator = DataCollatorForSupervisedLMMDataset(tokenizer, modalities)

    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )
    model.modalities = modalities
    model.config.use_cache = False
    model.config.model_cls = model_cls.__name__
    model.config.modality_builder = model_args.modality_builder

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if model_args.model_lora_path:
        raise ValueError(
            "LoRA path not supported for training -- set the output path to an existing model to resume training"
        )

    if training_args.lora_enable:
        logging.info("Adding LoRA adapters...")
        model = make_model_lora(model, training_args)

    if training_args.pretrained_projectors_path:
        projector_weights_og = torch.load(
            training_args.pretrained_projectors_path, map_location="cpu"
        )
        if model_args.use_multi_task==MultiTaskType.SIMPLE_MULTI_TASK:
            projector_weights = {}
            for k, v in projector_weights_og.items():
                for m in modalities:
                    for task_name in m.tasks["task_heads"].keys():
                        if task_name in k:
                            projector_weights[k] = v
        else:
            projector_weights = {
                k: v for k, v in projector_weights_og.items() if "_lmm_projector" in k
            }

    elif training_args.pretrained_projectors_config:
        with open(training_args.pretrained_projectors_config, "r") as f:
            pretrained_weights_config = json.load(f)

        projector_weights = {}

        for pretrained_path_info in pretrained_weights_config["pretrained_paths"]:
            pretrained_path = pretrained_path_info["path"]
            components = pretrained_path_info["components"]
            use_prefix = pretrained_path_info["use_prefix"]
            prefix = pretrained_path_info["prefix"]
            
            pretrained_weights = torch.load(pretrained_path, map_location="cpu")
            
            for k, v in pretrained_weights.items():
                if any(component in k for component in components):
                    weight_key = k
                    if use_prefix:
                        weight_key = prefix + "." + k
                    projector_weights[weight_key] = v

    else:
        projector_weights = {}

    model.get_model().initialize_modules(modalities, projector_weights)

    task_names = []
    tasks = {}
    for m in model.modalities:
        if m.use_multi_task != MultiTaskType.NO_MULTI_TASK:
            tasks = m.tasks
            task_names += m.tasks["task_heads"].keys()

    if training_args.pretrain_projectors:
        model.requires_grad_(False)
        for m in modalities:
            if m.use_multi_task == MultiTaskType.SIMPLE_MULTI_TASK:
                for task_name in m.tasks["task_heads"].keys():
                    task_model = getattr(model.get_model(), m.name + "_" + task_name)
                    for p in task_model.parameters():
                        p.requires_grad = True
            elif m.use_multi_task == MultiTaskType.PROJECTED_MULTI_TASK:
                proj = getattr(model.get_model(), m.name + "_lmm_projector")
                
                if "backbone" in m.tasks.keys():
                    backbone = getattr(proj,  "backbone")
                    for backbone_param in backbone.parameters():
                        backbone_param.requires_grad = tasks["backbone"]["requires_grad"]

                for task in task_names:
                    task_head = getattr(proj, task)
                    for task_head_param in task_head.parameters():
                        task_head_param.requires_grad = tasks["task_heads"][task]["requires_grad"]
                    if task in tasks["task_projectors"]:
                        task_projector = getattr(proj, task + "_projector")
                        for task_projector_param in task_projector.parameters():
                            task_projector_param.requires_grad = tasks["task_projectors"][task]["requires_grad"]

            else:
                proj = getattr(model.get_model(), m.name + "_lmm_projector")
                for p in proj.parameters():
                    p.requires_grad = True

    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(
        os.path.join(training_args.output_dir, "model_named_parameters.txt"), "w"
    ) as f:
        for name, param in model.named_parameters():
            f.write(f"{name} {param.shape} {param.requires_grad}\n")

    with open(os.path.join(training_args.output_dir, "README.md"), "w") as f:
        modalities_text = [
            f"* {m.__class__.__name__} (use `{m.token}` in text and provide `{m.data_key}`, encoded as {m.token_width} tokens)"
            for m in modalities
        ]
        readme_text = README_TEMPLATE.format(
            base_model=model_args.model_name_or_path,
            dataset=train_data_args.dataset_path,
            dataset_example=repr(train_dataset.get_example()),
            num_examples=len(train_dataset),
            modalities="\n".join(modalities_text),
            training_devices_dump=_get_training_devices_dump(),
            repr_model=f"{model_cls.__name__}.model =\n\n{repr(model)}",
        )
        f.write(readme_text)

    trainer = LMMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=evaluation_dataset,
    )

    if list(pathlib.Path(training_args.output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    model.config.save_pretrained(training_args.output_dir)
    state_dict = get_peft_state(model.named_parameters(), training_args.lora_bias)
    model.save_pretrained(training_args.output_dir, state_dict=state_dict)

    non_lora_state_dict = get_peft_state_non_lora(model.named_parameters(), task_names)
    torch.save(
        non_lora_state_dict,
        os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
    )
