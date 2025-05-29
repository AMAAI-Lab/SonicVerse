from typing import List, Dict, Sequence
from dataclasses import dataclass, field
import logging
import os

from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset, Dataset as HFDataset
import transformers
import torch

from sonicverse.modalities.base_modality import Modality
from sonicverse.constants import IGNORE_INDEX
from sonicverse.data_tools import encode_chat, encode_chat_multitask
from sonicverse.model_utils import MultiTaskType


@dataclass
class DataArguments:
    dataset_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )

@dataclass
class TrainDataArguments:
    train_dataset_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )

@dataclass
class EvaluationDataArguments:
    evaluation_dataset_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )


def _resolve_dataset(path: str) -> HFDataset:
    if os.path.exists(path):
        return load_from_disk(path)
    else:
        return load_dataset(path, split="train", data_files="*.arrow")


class LMMDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        tokenizer: transformers.PreTrainedTokenizer,
        modalities: List[Modality],
    ):
        super(LMMDataset, self).__init__()
        self.dataset = _resolve_dataset(data_args.dataset_path)
        self.tokenizer = tokenizer
        self.modalities = modalities

    def __len__(self):
        return len(self.dataset)

    def get_example(self) -> Dict:
        return self.dataset[0]

    def __getitem__(self, i) -> Dict:
        try:
            item = self.dataset[i]
            use_multi_task = MultiTaskType.NO_MULTI_TASK
            for m in self.modalities:
                if m.use_multi_task != MultiTaskType.NO_MULTI_TASK:
                    use_multi_task = m.use_multi_task
                    break
            if use_multi_task != MultiTaskType.NO_MULTI_TASK:
                return encode_chat_multitask(item, self.tokenizer, self.modalities)
            else:
                return encode_chat(item, self.tokenizer, self.modalities)
        except Exception as e:
            new_i = i + 1
            if new_i >= len(self):
                new_i = 0
            logging.error(f"Error encoding chat: {e} index={i} trying index={new_i}")
            return self.__getitem__(new_i)


@dataclass
class DataCollatorForSupervisedLMMDataset:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        modalities: List[Modality],
    ):
        self.tokenizer = tokenizer
        self.modalities = modalities

        self.use_multi_task = MultiTaskType.NO_MULTI_TASK
        for modality in self.modalities:
            if modality.use_multi_task != MultiTaskType.NO_MULTI_TASK:
                self.use_multi_task = modality.use_multi_task
                break

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, List]:
        input_ids = []
        lmm_labels = []
        task_labels = []
        for instance in instances:
            input_ids.append(instance["input_ids"])
            if self.use_multi_task == MultiTaskType.NO_MULTI_TASK:
                lmm_labels.append(instance["labels"])
            else:
                lmm_labels.append(instance["labels"][0])
                inst_task_labels = []
                for label_id in range(1, len(instance["labels"])):
                    inst_task_labels.append(instance["labels"][label_id])
                task_labels.append(inst_task_labels)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # print("Lmm labels 1 type :", type(lmm_labels))
        lmm_labels = torch.nn.utils.rnn.pad_sequence(
            lmm_labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        # print("Lmm labels 2 type :", type(lmm_labels))

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        lmm_labels = lmm_labels[:, : self.tokenizer.model_max_length]
        output_labels = [lmm_labels, task_labels]
        batch = dict(
            input_ids=input_ids,
            labels=output_labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        for m in self.modalities:
            batch[m.name] = [instance[m.name] for instance in instances]

        return batch
