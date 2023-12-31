# ---
# args: ["--prompt", "How do planes work?"]
# ---
# # Run Falcon-40B with bitsandbytes
#
# In this example, we download the full-precision weights of the Falcon-40B LLM but load it in 4-bit using
# Tim Dettmer's [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) library. This enables it to fit
# into a single GPU (A100 40GB).
#
# Due to the current limitations of the library, the inference speed is a little over 2 tokens/second and due
# to the sheer size of the model, the cold start time on Modal is around 2 minutes.
#
# For faster cold start at the expense of inference speed, check out
# [Running Falcon-40B with AutoGPTQ](/docs/examples/falcon_gptq).
#
# ## Setup
#
# First we import the components we need from `modal`.
import os
import json
import copy
import re
from string import Template
from pathlib import Path
from datasets import load_dataset, Dataset
from modal import Image, Stub, gpu, method, web_endpoint, Secret, Volume
from transformers import TrainingArguments
from peft import LoraConfig
from trl import DPOTrainer

VOL_MOUNT_PATH = Path("/vol")

# Spec for an image where prometheus-7b is cached locally
def download_prometheus_7b():
    from huggingface_hub import snapshot_download

    model_name = "kaist-ai/Prometheus-7b-v1.0"
    snapshot_download(model_name)

image = (
    Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.7",
        "cudnn=8.1.0",
        "cuda-nvcc",
        "scipy",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(
        "bitsandbytes==0.39.0",
        "bitsandbytes-cuda117==0.26.0.post2",
        "transformers==4.30",
        "accelerate @ git+https://github.com/huggingface/accelerate.git",
        "hf-transfer~=0.1",
        "torch==2.0.0",
        "torchvision==0.15.1",
        "sentencepiece==0.1.97",
        "huggingface_hub==0.14.1",
        "einops==0.6.1",
        "datasets",
        "trl",
        "peft"
    )
    # Use huggingface's hi-perf hf-transfer library to download this large model.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_prometheus_7b)
)

stub = Stub(image=image, name="gen_prometheus_with_conversation")
output_vol = Volume.persisted("finetune-volume")

# ## The model class
#
# Next, we write the model code. We want Modal to load the model into memory just once every time a container starts up,
# so we use [class syntax](/docs/guide/lifecycle-functions) and the __enter__` method.
#
# Within the [@stub.cls](/docs/reference/modal.Stub#cls) decorator, we use the [gpu parameter](/docs/guide/gpu)
# to specify that we want to run our function on an [A100 GPU](/pricing). We also allow each call 10 mintues to complete,
# and request the runner to stay live for 5 minutes after its last request.
#
# We load the model in 4-bit using the `bitsandbytes` library.
#
# The rest is just using the [pipeline()](https://huggingface.co/docs/transformers/en/main_classes/pipelines)
# abstraction from the `transformers` library. Refer to the documentation for more parameters and tuning.
@stub.cls(
    gpu=gpu.A100(memory=80, count=1),
    secret=Secret.from_name("huggingface-secret"), 
    volumes={VOL_MOUNT_PATH: output_vol},
    timeout=3600
)  # Use A10s)
class Prometheus7B_8bit:
    def __enter__(self):
        import torch
        from transformers import (
            LlamaForCausalLM,
            AutoTokenizer,
            TrainingArguments
        )

        self.model_name = "kaist-ai/Prometheus-7b-v1.0"

        model = LlamaForCausalLM.from_pretrained(self.model_name, local_files_only=True, device_map="auto", load_in_4bit=True)
        model_ref = LlamaForCausalLM.from_pretrained(self.model_name, local_files_only=True, device_map="auto", load_in_4bit=True)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=os.environ["HUGGINGFACE_TOKEN"], device_map="auto")
        tokenizer.pad_token = tokenizer.eos_token
        self.model = model
        self.model_ref = model_ref
        self.tokenizer = tokenizer

    @method()
    def parse_feedback(self, input_string):
        """
        Parses the string to extract the content that comes after '###Feedback:'.

        Args:
        input_string (str): The input string containing the '###Feedback:' marker.

        Returns:
        str: The substring that comes after '###Feedback:'.
        """
        # Define the marker
        marker = "###Feedback:"

        # Find the index of the marker
        index = input_string.find(marker)

        # Check if the marker is found
        if index != -1:
            # Extract and return the substring that comes after the marker
            return input_string[index + len(marker):].strip()
        else:
            # Return an empty string if the marker is not found
            return ""
    
    
    # @method()
    # def train_sft(self, dataset):
    #     train_dataset = Dataset.from_list(dataset)
        
    #     self.model.train()
    #     peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM", lora_dropout=0.05, inference_mode=False)


    #     training_args = TrainingArguments(
    #         output_dir= str(VOL_MOUNT_PATH / "model_SFT_0"),
    #         num_train_epochs= 1,
    #         per_device_train_batch_size= 1,
    #         optim = "paged_adamw_8bit",
    #         logging_steps= 10,
    #         fp16=True,
    #         evaluation_strategy="no",
    #         logging_first_step=True,
    #         learning_rate= 2e-4,
    #         weight_decay= 0.001,
    #         max_grad_norm= 0.3,
    #         warmup_ratio= 0.03,
    #         lr_scheduler_type= "linear",
    #     )

    #     trainer = SFTTrainer(
    #         model=self.model,
    #         train_dataset=train_dataset,
    #         tokenizer=self.tokenizer,
    #         max_seq_length=4096,
    #         peft_config=peft_config,
    #         dataset_text_field="text",
    #         args=training_args,
    #         packing= False,
    #     ) 
        
    #     trainer.train()
    #     trainer.save_model(str(VOL_MOUNT_PATH / "model_SFT_0"))

    #     trainer.model.save_pretrained(str(VOL_MOUNT_PATH / "model_SFT_0" / "final_checkpoint"))
    #     output_vol.commit()
    #     print("✅ done")

    @method()
    def train_dpo(self, dataset):
        
        train_dataset = Dataset.from_list(dataset)
       
        self.model.train()
        
        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            learning_rate=3e-04,
            max_steps=335,
            remove_unused_columns=False,
            evaluation_strategy="no",
            logging_first_step=True,
            logging_steps=10,  # match results in blog post
            output_dir=str(VOL_MOUNT_PATH / "model_3"),
            run_name="dpo_prometheus",
        )
        peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM", lora_dropout=0.05, inference_mode=False)
        
        dpo_trainer = DPOTrainer(self.model, self.model_ref, args=training_args, beta=0.1, train_dataset=train_dataset, tokenizer=self.tokenizer, peft_config=peft_config)
        dpo_trainer.train()
        dpo_trainer.save_model(str(VOL_MOUNT_PATH / "model_3"))

        dpo_trainer.model.save_pretrained(str(VOL_MOUNT_PATH / "model_3" / "final_checkpoint"))
        output_vol.commit()
        print("✅ done")
    
    @method()
    def generate(self, prompt: str):
       
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized.input_ids
        input_ids = input_ids.to(self.model.device)

        outputs = self.model.generate(input_ids, temperature=1.0, top_p=0.4, max_new_tokens=512, repetition_penalty=1.03)
        outputs = self.parse_feedback.local(self.tokenizer.decode(outputs[0]))
        print(f"remote: {outputs}")
        return outputs


# ## Run the model
# We define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run -q falcon_bitsandbytes.py`. The `-q` flag
# enables streaming to work in the terminal output.

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

TEMP_PATH = "/mnt/c/Users/amangokrani/OneDrive - Microsoft/Personal/evaluations/outputs/temp/prompt-1"


@stub.local_entrypoint()
def main():
    dataset = load_dataset("json", data_files="/mnt/c/Users/amangokrani/OneDrive - Microsoft/Personal/evaluations/outputs/train.dpo.gpt-4.claude-2.1.max-rand.json")
    dataset = dataset["train"]
    
    model = Prometheus7B_8bit()
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
    train_dataset = dataset.select_columns(['prompt', 'chosen', 'rejected'])
    
    #train_dataset = train_dataset.map(lambda example: {'text': example['prompt'] + example['chosen']})
    model.train_dpo.remote(dataset=train_dataset.to_list())
