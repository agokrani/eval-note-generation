import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from datasets import Dataset
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, DPOTrainer
from utils import create_and_prepare_model, create_datasets, preprocess
import wandb

wandb.login(key='3455d958c107d8074dd52fc9f08829fa6984d7a1')

# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    cache_volume: Optional[str] = field(
        default="/pretrained", 
        metadata={"help": "modal volume for looking up cache dir"}
    )
    cache_dir: Optional[str] = field(
        default="base_model", 
        metadata={"help": "directory for caching the model from huggingface"},
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_qunatization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_qunatization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )

@dataclass 
class FinetuningArguments: 
    training_type: Optional[str] = field(
        default="sft",
        metadata={"help": "sft or dpo to define the type of finetuning"},
    )
    beta_dpo: Optional[float] = field(default=0.1, metadata={"help": "beta for dpo training. This value will be ignored during sft training"})
    max_length: Optional[int] = field(default=512, metadata={"help": "The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "The maximum length of the prompt. This argument is required if you want to use the default data collator"})
    max_target_length: Optional[int] = field(default=256, metadata={"help": "The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder"})

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "Dataset field to use as input text."}
    )
    max_seq_length: Optional[int] = field(default=512)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, appends `eos_token_id` at the end of each sample being packed."
        },
    )
    preprocess_data: Optional[bool] = field(
        default=False, 
        metadata={
            "help": "If True, apply the preprocessing function to the dataset"
        }
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, tokenizers adds special tokens to each sample being packed."
        },
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    train_on_completions: Optional[bool] = field(
        default = False,
        metadata={
            "help": "train the model on completions only. Packing must be false"
        },
    )
    response_template: Optional[str] = field(
        default=None, 
        metadata={
            "help": "The response template when training on completions only"
        },
    )

def main(model_args, data_args, training_args, finetuning_args, train_dataset, eval_dataset):
    model, model_ref, peft_config, tokenizer = create_and_prepare_model(model_args, finetuning_args)
    train_dataset = Dataset.from_list(train_dataset)
    if eval_dataset is not None: 
        eval_dataset = Dataset.from_list(eval_dataset)
    # gradient ckpt
    model.config.use_cache = training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
       training_args.gradient_checkpointing_kwargs = {
           "use_reentrant": model_args.use_reentrant
       }
    
    data_collator = None
    if finetuning_args.training_type == "dpo" and data_args.train_on_completions: 
        print(f"WARNING: Train on completions is set to true. Since dpo training does that by default, we will ignore this argument")
    if finetuning_args.training_type == "sft":
        if data_args.train_on_completions and not data_args.packing:
            if data_args.response_template is not None:  
                print(f"WARNING: Training on completions only is set to true and it might lead to unecessary outcomes if you don't wish for that")
                response_template_with_context = "\n" + data_args.response_template  # We added context here: "\n". This is enough for this tokenizer
                response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
                print(f"template_ids: {response_template_ids}")
                data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
            else: 
                raise ERROR("Response template can't be none when training on completions")
        elif data_args.train_on_completions and data_args.packing:
            raise ERROR("Packing must be set to false when train_on_completions is set to true")
    
    # trainer
    trainer = None
    if finetuning_args.training_type == "sft": 
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            packing=data_args.packing,
            dataset_kwargs={
                "append_concat_token": data_args.append_concat_token,
                "add_special_tokens": data_args.add_special_tokens,
            },
            data_collator=data_collator,
            dataset_text_field=data_args.dataset_text_field,
            max_seq_length=data_args.max_seq_length,
        )
    
    elif finetuning_args.training_type == "dpo": 
        trainer = DPOTrainer(
            model=model, 
            #ref_model=model_ref,
            tokenizer=tokenizer,
            args=training_args,
            beta=finetuning_args.beta_dpo,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config, 
            #max_prompt_length=finetuning_args.max_prompt_length,
            #max_length=finetuning_args.max_length,
            #max_target_length=finetuning_args.max_target_length
        )

    trainer.accelerator.print(f"{trainer.model}")
    if model_args.use_peft_lora:
    # handle PEFT+FSDP case
        trainer.model.print_trainable_parameters()
        if getattr(trainer.accelerator.state, "fsdp_plugin", None):
            from peft.utils.other import fsdp_auto_wrap_policy
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
            )

            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
            print(f"cpu_offload: {fsdp_plugin.cpu_offload}")
            kwargs = {
                "sharding_strategy": fsdp_plugin.sharding_strategy,
                "cpu_offload": fsdp_plugin.cpu_offload,
                "auto_wrap_policy": auto_wrap_policy,
                "mixed_precision": fsdp_plugin.mixed_precision_policy,
                "sync_module_states": fsdp_plugin.sync_module_states,
                "use_orig_params": False,  # this should be `False`
                "limit_all_gathers": True,
                "param_init_fn": fsdp_plugin.param_init_fn,
                "device_id": trainer.accelerator.device,
            }
            trainer.model = trainer.model_wrapped = FSDP(trainer.model, **kwargs)
            trainer.args.remove_unused_columns = False

    # train
    trainer.train()

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    
    trainer.save_model(training_args.output_dir)
    trainer.model.save_pretrained(training_args.output_dir + "/final_checkpoint")
    print("✅ done")

if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, FinetuningArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, finetuning_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, finetuning_args = parser.parse_args_into_dataclasses()
    
    datasets = create_datasets(data_args, preprocess_func=preprocess if data_args.preprocess_data else None)
    main(
        model_args, 
        data_args, 
        training_args, 
        finetuning_args, 
        train_dataset=datasets["train"].to_list(), 
        eval_dataset=datasets["test"].to_list() if "test" in datasets.keys() else None
    )