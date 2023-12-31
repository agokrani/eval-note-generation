from enum import Enum
import os
import re
import torch
from string import Template
from accelerate import Accelerator
from datasets.builder import DatasetGenerationError
from pathlib import Path
from tqdm import tqdm
from peft import LoraConfig
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from common import (
    stub,
    VOLUME_CONFIG,
    image,
    GPU_CONFIG
)

PROMPT = Template(
"""
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
summarize the conversation to generate a clinical note with four sections: HISTORY OF PRESENT ILLNESS, PHYSICAL EXAM, RESULTS, ASSESSMENT AND PLAN.
The conversation is:
$conversation

###Response to evaluate:
$summary

###Reference Answer (Score 5):
$reference

###Score Rubric:
[Is the model able to accurately and effectively summarize a medical conversation into a clinical note with four sections: HISTORY OF PRESENT ILLNESS, PHYSICAL EXAM, RESULTS, ASSESSMENT AND PLAN?]

Score 1: The summary utterly fails to reflect the conversation. It is incoherent, irrelevant, excessively verbose, or filled with hallucinations. There is a blatant disregard for standard clinical terminology, and critical information is omitted.

Score 2: The summary sporadically reflects elements of the conversation, but it frequently includes irrelevant or incoherent content. There is a noticeable lack of standard clinical terminology, verbosity is apparent, hallucinations are present, and critical information is often omitted.

Score 3: The summary generally captures the conversation accurately but occasionally includes irrelevant or incoherent content. It mostly uses standard clinical terms but can be verbose at times. Minor hallucinations may occur, and there might be instances of critical information being overlooked.

Score 4: The summary often accurately reflects the conversation, maintaining coherence and relevance throughout. There are minor cases of verbosity or use of non-standard clinical terms. The summary may have slight omissions or infrequent minor hallucinations.

Score 5: The summary flawlessly encapsulates the conversation, demonstrating complete coherence, relevance, and succinctness. It consistently employs standard clinical terminology, contains no hallucinations, and does not omit any critical information.

###Feedback:
"""
)

def preprocess(dataset):
    prompts = []
    for i in range(len(dataset)):
        example = dataset[i]
        print("Processing file: ", example["file"])
    
        conversation = example["src"].replace('[doctor]', 'doctor:').replace('[patient]', 'patient:')
        conversation = re.sub(r'\s+([,.?])', r'\1', conversation)
        summary = example["pred"]
        reference = example["tgt"]
        
        prompt = PROMPT.substitute(conversation=conversation, summary=summary, reference=reference)
        prompts.append(prompt)
    
    dataset = dataset.add_column("prompt", prompts)
    dataset = dataset.map(lambda example: {'text': example['prompt'] + example['chosen']})

    return dataset

def create_datasets(data_args, **kwargs):
    raw_datasets = DatasetDict()
    for split in data_args.splits.split(","):
        try:
            # Try first if dataset on a Hub repo
            dataset = load_dataset(data_args.dataset_name, split=split)
        except:
            # If not, check local dataset
            raw_datasets = load_dataset(Path(data_args.dataset_name).suffix[1:], data_files=data_args.dataset_name)
            break
        if "train" in split:
            raw_datasets["train"] = dataset
        elif "test" in split:
            raw_datasets["test"] = dataset
        else:
            raise ValueError(
                f"Split type {split} not recognized as one of test or train."
            )
    
    if data_args.preprocess_data and kwargs["preprocess_func"] is not None: 
        func = kwargs["preprocess_func"]
        raw_datasets["train"] = func(raw_datasets["train"])
        if "test" in raw_datasets.keys(): 
            raw_datasets["test"] = func(raw_datasets["test"])

    
    print(
        f"Size of the train set: {len(raw_datasets['train'])}. Size of the validation set: {len(raw_datasets['test']) if 'test' in raw_datasets.keys() else 0}"
    )
    
    print(f"A sample of train dataset: {raw_datasets['train'][0]}")

    return raw_datasets

@stub.function(gpu=GPU_CONFIG, image=image, timeout=3600, volumes=VOLUME_CONFIG)
def create_and_prepare_model(model_args, finetuning_args):
    device_map = None
    bnb_config = None
    load_in_8bit = model_args.use_8bit_qunatization

    if model_args.use_4bit_qunatization:
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
                    
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit_qunatization,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and model_args.use_4bit_qunatization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print(
                    "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
                )
                print("=" * 80)

    if model_args.use_4bit_qunatization or model_args.use_8bit_qunatization:
        device_map = (
            "auto"
            #{"": Accelerator().local_process_index}
            # int(os.environ.get("LOCAL_RANK", -1))
            # if torch.distributed.is_available() and torch.distributed.is_initialized()
            # else "auto"
        )  # {"": 0}
    
    cache_dir = str(Path(model_args.cache_volume) / model_args.cache_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path if not os.path.exists(cache_dir) else cache_dir,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model_ref = None
    if finetuning_args.training_type == "dpo": 
        model_ref = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path if not os.path.exists(cache_dir) else cache_dir,
            quantization_config=bnb_config,
            device_map=device_map
        )
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name_or_path if not os.path.exists(cache_dir) else cache_dir,
    #     load_in_8bit=load_in_8bit,
    #     quantization_config=bnb_config,
    #     device_map=device_map,
    #     trust_remote_code=True,
    #     attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
    # )
    if not os.path.exists(cache_dir): 
        model.save_pretrained(cache_dir)
    
        VOLUME_CONFIG[Path(model_args.cache_volume)].commit()

    peft_config = None
    chat_template = None
    if model_args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            r=model_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=model_args.lora_target_modules.split(","),
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True, cache_dir=model_args.cache_volume+ model_args.cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    
    
    return model, model_ref, peft_config, tokenizer


# special_tokens = None
    # chat_template = None
    # if args.chat_template_format == "chatml":
    #     special_tokens = ChatmlSpecialTokens
    #     chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    # elif args.chat_template_format == "zephyr":
    #     special_tokens = ZephyrSpecialTokens
    #     chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    # if special_tokens is not None:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         args.model_name_or_path,
    #         pad_token=special_tokens.pad_token.value,
    #         bos_token=special_tokens.bos_token.value,
    #         eos_token=special_tokens.eos_token.value,
    #         additional_special_tokens=special_tokens.list(),
    #         trust_remote_code=True,
    #     )
    #     tokenizer.chat_template = chat_template
    #     # make embedding resizing configurable?
    #     model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # else: