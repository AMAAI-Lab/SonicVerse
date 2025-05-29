
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    MistralConfig,
    MistralModel,
    MistralForCausalLM,
)

from transformers.modeling_outputs import CausalLMOutputWithPast

from sonicverse.language_models.base_model import (
    LMMMetaModel,
    LMMMetaForCausalLM,
)


class MistralLMMConfig(MistralConfig):
    model_type = "mistral-lmm"


class MistralLMMModel(LMMMetaModel, MistralModel):
    config_class = MistralLMMConfig

    def __init__(self, config: MistralLMMConfig):
        super(MistralLMMModel, self).__init__(config)


class MistralLMMForCausalLM(MistralForCausalLM, LMMMetaForCausalLM):
    config_class = MistralLMMConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        self.model = MistralLMMModel(config)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.modalities = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self) -> "MistralLMMForCausalLM":
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        #print("Past keys ",past_key_values)
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels != None:
            labels_inp = labels[0]
        else:
            labels_inp = labels
        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            lmm_labels,
            task_values
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels_inp, **kwargs
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # print("Labels 1 size ", len(labels[1]))
        # print("labels 1 element size ", len(labels[1][0]))
        # print("labels 1 element 1 task size ", labels[1][0][0].shape)
        # print("labels 1 element 2 task size ", labels[1][0][1].shape)
        # print("labels 1 element 3 task size ", labels[1][0][2].shape)
        # print("task vals size ", len(task_values))
        # for task in task_values.keys():
        #     print(" task ", task, len(task_values[task]))
        #     print(" task element", task, task_values[task][0].shape)


        if labels != None:
            task_pairs = {}
            task_list = list(task_values.keys())
            for task_id in range(len(task_list)):
                _task_labels = []
                _task_outputs = []

                _task = task_list[task_id]
                for inst in range(len(task_values[_task])):
                    # print("task output shape ", _task, task_values[_task][inst].shape)
                    _task_outputs.append(task_values[_task][inst].unsqueeze(0))
                    _task_labels.append(torch.stack([labels[1][inst][task_id]]))

                task_pairs[_task] = [_task_labels, _task_outputs]
                # print("TASK ", _task)
                # print(" LABELS LEN ", len(task_pairs[_task][0]))
                # print(" LABELS ELEM shape ", task_pairs[_task][0][0].shape)
                # print(" VALUES LEN ", len(task_pairs[_task][1]))
                # print(" VALUES ELEM shape ", task_pairs[_task][1][0].shape)

        loss = None
        if lmm_labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = lmm_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # print("loss ", loss)


        if labels != None:
            task_loss = {}
            for task in task_list:
                preds = torch.cat(task_pairs[task][1], dim=0)
                labs = torch.cat(task_pairs[task][0], dim=0)
                preds_flat = preds.view(-1, preds.size(-1))  # Reshape to (batch_size * sequence_length, num_classes)
                labs_flat = labs.view(-1)  # Reshape to (batch_size * sequence_length)

                #print("task ", task)
                #print("preds shape ", preds.shape)
                #print("labs shape ", labs.shape)
                if task == "lmm_projector":
                    task_loss[task] = CrossEntropyLoss()(preds,labs)
                else:
                    task_loss[task] = nn.BCEWithLogitsLoss()(preds, labs)
        # print("task losses ", task_loss)

        total_loss = None
        if labels != None:
            total_task_loss = None
            for task in task_list:
                if self.modalities[0].tasks["task_heads"][task]["weight"] != 0.0:
                    if total_task_loss != None:
                        total_task_loss += self.modalities[0].tasks["task_heads"][task]["weight"]*task_loss[task]
                    else:
                        total_task_loss = self.modalities[0].tasks["task_heads"][task]["weight"]*task_loss[task]

            if total_task_loss != None:
                total_loss = self.modalities[0].tasks["task_heads"]["lmm_projector"]["weight"]*loss + total_task_loss
            else:
                total_loss = loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (total_loss,) + output if total_loss is not None else output

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        modality_inputs=None,
        **kwargs
    ):
        #print("hoooo", past_key_values)

        #past_key_values = None
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None:
            raise ValueError("inputs_embeds not supported")

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": None,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            **(modality_inputs or {}),
        }

        return model_inputs


AutoConfig.register("mistral-lmm", MistralLMMConfig)
AutoModelForCausalLM.register(MistralLMMConfig, MistralLMMForCausalLM)
