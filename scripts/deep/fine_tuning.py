import kagglehub
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
import os
from unsloth import is_bfloat16_supported
from huggingface_hub import login
import wandb
from dotenv import load_dotenv

load_dotenv()

# Constants
MAX_SEQ_LENGTH = 3000  # Adjust based on your dataset and GPU memory
DTYPE = None  # Let the library decide the best dtype
LOAD_IN_4BIT = True  # Use 4-bit quantization for memory efficiency
LORA_R = 8  # LoRA rank
LORA_ALPHA = 8  # LoRA alpha
LORA_DROPOUT = 0  # LoRA dropout
USE_GRADIENT_CHECKPOINTING = "unsloth"  # Use gradient checkpointing for memory efficiency
RANDOM_STATE = 3407  # Random seed for reproducibility
TRAIN_BATCH_SIZE = 2  # Adjust based on GPU memory
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients to simulate larger batch sizes
LEARNING_RATE = 1e-5  # Learning rate for fine-tuning
NUM_TRAIN_EPOCHS = 1  # Number of training epochs
OUTPUT_DIR = "/content/drive/MyDrive/FinetuningLlama/arguments"  # Directory to save training arguments
MODEL_SAVE_PATH = "/content/drive/MyDrive/FinetuningLlama/llama-finetuned_model"  # Directory to save the fine-tuned model
HUGGINGFACE_REPO = "aaditey932/llama-resume-optimizer"  # Hugging Face repository to push the model
HUGGINGFACE_TOKEN = os.get_env("HF_TOKEN")  # Hugging Face API token
WB_TOKEN = os.get_env("WB_TOKEN")  # Weights & Biases API token
NEW_MODEL_NAME = "llama-resume-optimizer"  # Name of the fine-tuned model

def download_dataset():
    """Download dataset from Kaggle Hub."""
    path = kagglehub.dataset_download("thedrcat/llm-prompt-recovery-data")
    print(f"✅ Dataset downloaded to: {path}")
    return path

def format_to_sharegpt(example, tokenizer):
    """Format the dataset into the training prompt structure."""
    resume_text = example["resume"]
    job_description = example["job_description"]
    expected_output = example["expected_output"]

    training_prompt = f"""
You are an AI-powered resume optimization assistant. Your task is to analyze the provided resume against the job description and generate ATS-friendly, structured, and impact-driven improvements. These improvements should be directly replaceable in the original document.

### Key Requirements:
- Output will be used programmatically to modify the resume while maintaining its structure.
- Ensure maximum ATS compatibility by:
  - Embedding relevant keywords from the job description naturally.
  - Using active, impact-driven language to highlight achievements.
  - Making job titles, dates, and formatting consistent.
- Response must be a structured JSON object.

### Output Format:
Generate a structured JSON object with numbered edits (`"edit1"`, `"edit2"`, etc.) where:
- `"to_be_edited"` → The exact resume text that needs improvement.
- `"edited"` → The optimized version with clear, ATS-friendly, and impact-driven phrasing.
  - Include missing keywords from the job description.
  - Add impact-driven metrics (e.g., "Increased efficiency by 30%").
  - Use concise phrasing improvements (e.g., replace "responsible for" with "managed").
  - Fix formatting issues (e.g., consistent bullet points or date formats).

### Instructions:
1. **Word-level**: Replace weak, generic words with stronger ATS-friendly alternatives.
2. **Phrase-level**: Refine short phrases for clarity, conciseness, and keyword inclusion.
3. **Sentence-level**: Enhance structure to highlight quantifiable impact and results.
4. **Skills**: Ensure any skills mentioned in the job description are included in the edits.

---

### Resume:
{resume_text}

### Job Description:
{job_description}

---

### Expected JSON Output Example:
{expected_output}
"""
    return {"text": training_prompt}

def load_model_and_tokenizer():
    """Load the model and tokenizer with LoRA configuration."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        random_state=RANDOM_STATE,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.2")
    return model, tokenizer

def load_and_prepare_dataset(data_file, tokenizer):
    """Load and prepare the dataset for training."""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"❌ Dataset file {data_file} not found.")

    dataset = load_dataset("csv", data_files=data_file, split="train")
    dataset = dataset.map(lambda x: format_to_sharegpt(x, tokenizer), batched=True)
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return dataset["train"], dataset["test"]

def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Train the model using SFTTrainer."""
    wandb.login(key=WB_TOKEN)
    if not wandb.api.api_key:
        raise ValueError("❌ Failed to log in to W&B. Check your API key.")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        evaluation_strategy="epoch",
        dataset_text_field="text",  # Ensure this matches the key in the dataset
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            evaluation_strategy="epoch",
            warmup_steps=5,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=RANDOM_STATE,
            output_dir=OUTPUT_DIR,
            push_to_hub=True,
            hub_model_id=f"aaditey932/{NEW_MODEL_NAME}",
            report_to="wandb"
        ),
    )

    trainer.train()
    return trainer

def save_and_push_model(model, tokenizer, trainer):
    """Save and push the entire model (base + LoRA adapter) to Hugging Face Hub."""
    print("🔄 Logging into Hugging Face...")
    login(token=HUGGINGFACE_TOKEN)

    print("💾 Merging LoRA adapter with the base model...")
    model = model.merge_and_unload()

    print("💾 Saving the entire model...")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"❌ Failed to create directory: {MODEL_SAVE_PATH}")

    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    saved_files = os.listdir(MODEL_SAVE_PATH)
    print("Saved files:", saved_files)

    model_files = ["pytorch_model.bin", "model.safetensors"]
    if not any(file in saved_files for file in model_files):
        raise FileNotFoundError(f"❌ Model files were not saved correctly. Expected one of: {model_files}")

    print("🚀 Pushing the entire model to Hugging Face Hub...")
    model.push_to_hub(HUGGINGFACE_REPO, token=HUGGINGFACE_TOKEN)
    tokenizer.push_to_hub(HUGGINGFACE_REPO, token=HUGGINGFACE_TOKEN)

    if trainer.args.push_to_hub:
        trainer.push_to_hub()

    print("✅ Entire model pushed to Hugging Face Hub!")

def main():
    """Main function to execute the script."""
    try:
        dataset_path = download_dataset()
        model, tokenizer = load_model_and_tokenizer()
        train_dataset, eval_dataset = load_and_prepare_dataset(dataset_path, tokenizer)

        # Debug: Print a sample training prompt
        print("Sample training prompt:")
        print(train_dataset[0]["text"])

        trainer = train_model(model, tokenizer, train_dataset, eval_dataset)
        save_and_push_model(model, tokenizer, trainer)
        wandb.finish()
        print("🎯 Fine-tuning completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()