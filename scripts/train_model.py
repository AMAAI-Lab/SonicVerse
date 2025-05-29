import transformers
import logging

from sonicverse.training import (
    TrainingArguments,
    ModelArguments,
    train_for_modalities,
)
from sonicverse.training_data import (
    DataArguments,
    TrainDataArguments,
    EvaluationDataArguments,
)

from sonicverse.model_utils import MultiTaskType
from sonicverse.language_models import LANGUAGE_MODEL_NAME_TO_CLASS
from sonicverse.modalities import MODALITY_BUILDERS

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser(
        (TrainingArguments, ModelArguments, TrainDataArguments, EvaluationDataArguments)
    )

    training_args, model_args, train_data_args, evaluation_data_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    _train_data_args = DataArguments()
    _evaluation_data_args = DataArguments()

    _train_data_args.dataset_path = train_data_args.train_dataset_path
    _evaluation_data_args.dataset_path = evaluation_data_args.evaluation_dataset_path

    if MultiTaskType(model_args.use_multi_task) != MultiTaskType.NO_MULTI_TASK:
        modalities = MODALITY_BUILDERS[model_args.modality_builder](use_multi_task = MultiTaskType(model_args.use_multi_task), tasks_config = model_args.tasks_config)
    else:
        modalities = MODALITY_BUILDERS[model_args.modality_builder]()

    model_cls = LANGUAGE_MODEL_NAME_TO_CLASS[model_args.model_cls]

    train_for_modalities(model_cls, training_args, model_args, _train_data_args, _evaluation_data_args, modalities)
