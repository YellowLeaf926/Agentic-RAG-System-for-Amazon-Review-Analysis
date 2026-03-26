import os
import re
import random
import json
import logging
import torch
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, default_data_collator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from parameters1 import OPENAI_API_KEY

# Model checkpoint
CHECKPOINT = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Use bfloat16 throughout for stable mixed-precision training
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(CHECKPOINT,
    dtype=torch.bfloat16,
    device_map="auto"
)

# Logger
def setup_logger():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/lora_train_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.info("Logger initialized.")

# Call LLM (for evaluation)
def call_llm(prompt, current_model):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(current_model.device)

    with torch.no_grad():
        outputs = current_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    return response

# Generate training data using GPT-4o-mini
def generate_training_data(csv_path: str, openai_api_key: str, num_samples: int = 200):
    logging.info("Generating training data using GPT-4o-mini...")

    df = pd.read_csv(csv_path)
    df['at'] = pd.to_datetime(df['at'])
    df = df[df['at'] > pd.Timestamp('2025-01-01')]
    df = df.dropna(subset=["content"]).drop_duplicates(subset="reviewId")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logging.info(f"Dataset loaded: {len(df)} reviews after filtering")

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    prompt = ChatPromptTemplate.from_template("""
You are an expert analyst specializing in mobile app user feedback.
Summarize the following Amazon Shopping app reviews:

1. Overall Sentiment: (positive / negative / mixed) with a one-line explanation
2. Key Praises: bullet points of what users liked most
3. Key Complaints: bullet points of what users disliked most

Only use information from the provided reviews. Be specific.

Reviews:
{reviews}

Summary:
""")
    chain = prompt | llm | StrOutputParser()

    examples = []
    batch_size = 10
    max_batches = num_samples  # each batch -> 1 sample

    for i in range(0, min(max_batches * batch_size, len(df)), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        combined = "\n\n---\n\n".join([
            f"Rating: {row['score']}/5\nReview: {row['content']}"
            for _, row in batch_df.iterrows()
        ])
        try:
            summary = chain.invoke({"reviews": combined})
            examples.append({"input": combined, "target": summary})
            if len(examples) % 20 == 0:
                logging.info(f"Generated {len(examples)}/{num_samples} samples")
        except Exception as e:
            logging.warning(f"Failed to generate sample at batch {i}: {e}")
            continue

        if len(examples) >= num_samples:
            break

    os.makedirs("data", exist_ok=True)
    with open("data/amazon_summaries.json", "w") as f:
        json.dump({"examples": examples}, f)

    logging.info(f"Saved {len(examples)} training examples to data/amazon_summaries.json")
    #print(f"Saved {len(examples)} training examples to data/amazon_summaries.json")
    return examples

# Tokenize training data
def prepare_and_tokenize(example):
    # Dataset reviews are short (avg 129 chars), so 512 tokens is sufficient
    MAX_LENGTH = 512

    messages = [{"role": "user", "content": f"""You are an expert analyst specializing in mobile app user feedback.
    Summarize the following Amazon Shopping app reviews:

    1. Overall Sentiment: (positive / negative / mixed) with a one-line explanation
    2. Key Praises: bullet points of what users liked most
    3. Key Complaints: bullet points of what users disliked most

    Only use information from the provided reviews. Be specific.

    Reviews:
    {example['input']}

    Summary:"""}]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    full_text = prompt_text + example['target'] + tokenizer.eos_token

    # Tokenize separately to get exact prompt length
    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False
    )["input_ids"]

    full_tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        add_special_tokens=False
    )

    input_ids = full_tokenized["input_ids"]
    attention_mask = full_tokenized["attention_mask"]

    prompt_len = min(len(prompt_ids), len(input_ids))

    labels = [-100] * prompt_len + input_ids[prompt_len:]
    labels = labels[:MAX_LENGTH]
    labels += [-100] * (MAX_LENGTH - len(labels))

    # Mask based on attention_mask to handle padding correctly
    labels = [label if attention_mask[i] == 1 else -100 for i, label in enumerate(labels)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# LoRA training
def lora_train(base_model, train_data, lora_config, training_args):
    tokenized_train = train_data.map(
        prepare_and_tokenize,
        remove_columns=train_data.column_names
    )

    # Verify labels are not all -100 (sanity check)
    sample = tokenized_train[0]
    non_masked = sum(1 for l in sample["labels"] if l != -100)
    logging.info(f"Sanity check - non-masked label tokens in first sample: {non_masked}")
    if non_masked == 0:
        logging.warning("WARNING: All labels are -100 in first sample! Check tokenization.")

    peft_model = get_peft_model(base_model, lora_config)

    peft_model.print_trainable_parameters()

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=default_data_collator,
    )

    trainer.train()
    return peft_model

# Evaluation
def evaluate(test_data, model_name, current_model):
    logging.info(f"\n{'='*50}")
    logging.info(f"Evaluating: {model_name}")
    logging.info(f"{'='*50}")

    current_model.eval()
    results = []

    for i, example in enumerate(test_data):
        prompt = f"""You are an expert analyst specializing in mobile app user feedback.
        Summarize the following Amazon Shopping app reviews:

        1. Overall Sentiment: (positive / negative / mixed) with a one-line explanation
        2. Key Praises: bullet points of what users liked most
        3. Key Complaints: bullet points of what users disliked most

        Only use information from the provided reviews. Be specific.

        Reviews:
        {example['input']}

        Summary:"""

        response = call_llm(prompt, current_model)

        results.append({
            "input": example["input"],
            "expected": example["target"],
            "predicted": response,
        })

        logging.info(f"\n[Sample {i+1}/{len(test_data)}]")
        logging.info(f"Expected:\n{example['target']}")
        logging.info(f"Predicted:\n{response}")

    return results

# Main
if __name__ == "__main__":
    setup_logger()

    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    DATA_PATH = "data/amazon_reviews.csv"

    # 1. Generate training data
    if not os.path.exists("data/amazon_summaries.json"):
        generate_training_data(DATA_PATH, OPENAI_API_KEY, num_samples=200)
    else:
        logging.info("Training data already exists, skipping generation")

    # 2. Load data
    with open("data/amazon_summaries.json", "r") as f:
        data = json.load(f)

    logging.info(f"Loaded {len(data['examples'])} total examples")

    dataset = Dataset.from_list(data["examples"])
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    test_data = split["test"]

    logging.info(f"Train size: {len(train_data)}")
    logging.info(f"Test size: {len(test_data)}")

    # 3. Base model evaluation
    logging.info("=== BASE MODEL EVALUATION ===")
    base_results = evaluate(test_data, "Base Model", model)
    torch.cuda.empty_cache()

    # 4. LoRA fine-tuning
    # Switch padding_side to right for training
    tokenizer.padding_side = "right"
    logging.info("=== STARTING LORA FINE-TUNING ===")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    training_args = TrainingArguments(
        output_dir="./model_checkpoints",
        num_train_epochs=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=20,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,                      
        report_to="none",
    )

    peft_model = lora_train(model, train_data, lora_config, training_args)

    os.makedirs("model_checkpoints/final", exist_ok=True)
    peft_model.save_pretrained("model_checkpoints/final")
    tokenizer.save_pretrained("model_checkpoints/final")
    logging.info("Fine-tuned model saved to model_checkpoints/final")

    # 5. Fine-tuned model evaluation
    logging.info("=== FINE-TUNED MODEL EVALUATION ===")
    tokenizer.padding_side = "left"
    ft_results = evaluate(test_data, "LoRA Fine-tuned Model", peft_model)

    logging.info("=== DONE ===")