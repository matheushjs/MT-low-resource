from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, setup_chat_format
import bitsandbytes as bnb

base_model = "Qwen/Qwen2.5-7B-Instruct"
new_model = "llama-3.2-3b-it-Ecommerce-ChatBot"
dataset_name = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# Set torch dtype and attention implementation
# NOTE: This might cause result differences between Euler and KIT clusters
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    torch_dtype = torch.float16
    attn_implementation = "eager"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation,
    use_cache=False
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

#Importing the dataset
dataset = load_dataset(dataset_name, split="train")

dataset = dataset.shuffle(seed=65).select(range(1000)) # Only use 1000 samples for quick demo

instruction = """You are a top-rated customer service agent named John. 
    Be polite to customers and answer all their questions.
    """
def format_chat_template(row):
    row_json = [{"role": "system", "content": instruction },
               {"role": "user", "content": row["instruction"]},
               {"role": "assistant", "content": row["response"]}]
    
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc= 4,
)

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules
)
tokenizer.chat_template = None
model, tokenizer = setup_chat_format(model, tokenizer)
model = get_peft_model(model, peft_config)

#Hyperparamter
training_arguments = SFTConfig(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    max_seq_length=512,
    dataset_text_field="text", # This argument was on Trainer
    packing= False, # This argument was on Trainer
    #max_length=512, # Idk what's going on. This is the latest version of TRL but it doesn't seem like it.
    #report_to="wandb"
)

dataset = dataset.train_test_split(0.1)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    #max_seq_length=512,
    #dataset_text_field="text",
    processing_class=tokenizer,
    args=training_arguments
)

train_result = trainer.train()
metrics = train_result.metrics
max_train_samples = len(dataset["train"])
metrics["train_samples"] = len(dataset["train"])
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

