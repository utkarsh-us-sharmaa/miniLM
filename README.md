## WikiGPT-Mini
A lightweight language model trained on Wikipedia data that generates coherent natural language responses. This project demonstrates how to build and train an efficient text generation model without unnecessary complexity.

## Project Overview
WikiGPT-Mini is a compact language model fine-tuned on Wikipedia content that can generate coherent text continuations. The project prioritizes functionality, efficiency, and simplicity over elaborate interfaces.

## Features
Lightweight model architecture (82M parameters)

Trained on curated Wikipedia content

Fast inference suitable for resource-constrained environments

Simple API for text generation

Hugging Face Hub integration for easy deployment

## Model Architecture
This project uses distilgpt2, a distilled version of GPT-2 with 82 million parameters. This architecture was chosen for several reasons:

Efficiency: Significantly smaller than full GPT-2 (124M) while retaining strong generation capabilities

Speed: Faster training and inference times compared to larger models

Resource requirements: Can be trained on Google Colab without requiring expensive GPU resources

Performance: Despite its smaller size, still capable of producing coherent and contextually relevant text

## Dataset
The model is trained on wikitext-2-raw-v1, a clean subset of Wikipedia articles:

Curated collection of verified Wikipedia articles

Contains natural English prose with proper grammar and structure

Small enough for quick training iterations but comprehensive enough to learn language patterns

Diverse topics providing broad knowledge context

## Tokenization
The tokenization strategy uses GPT-2's native Byte Pair Encoding (BPE) tokenizer:

Efficiency: BPE offers a good balance between character and word-level tokenization

Vocabulary coverage: Handles out-of-vocabulary words effectively

Consistency: Using the same tokenization approach as the base model ensures compatibility

Language support: Well-suited for English text processing

## Training Strategy
The model was fine-tuned using Hugging Face's Trainer API with causal language modeling:

Training objective: The model learns to predict the next token based on previous tokens

Input/output setup: Inputs and labels are identical for causal language modeling

Training duration: Limited to a few epochs to prevent overfitting while ensuring learning

Model persistence: Saved using model.save_pretrained() and uploaded to Hugging Face Hub for easy access

## Usage
python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "your-username/wikigpt-mini"  # Replace with your model path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(
    inputs["input_ids"],
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
Sample Outputs
Prompt: "The future of artificial intelligence is"
Output: [Sample output from your model would go here]

## Installation
bash
Clone the repository
git clone https://github.com/your-username/wikigpt-mini.git
cd wikigpt-mini

## Install dependencies
pip install -r requirements.txt

## Training Your Own Version
python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

// Load base model and tokenizer
model_name = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

// Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

// Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

// Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

// Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

// Train model
trainer.train()

// Save model
model.save_pretrained("./wikigpt-mini")
tokenizer.save_pretrained("./wikigpt-mini")

![miniLM Prompt-Output-1](https://github.com/user-attachments/assets/c3ef1ece-c3e9-4c40-8c2b-5fb94f793696)



Requirements
Python 3.7+

PyTorch 1.9+

Transformers 4.12+

Datasets 1.11+

Future Improvements
Experiment with different model sizes to find optimal performance/size ratio

Expand training dataset to include more diverse Wikipedia content

Implement domain-specific fine-tuning for specialized use cases

Add evaluation metrics to quantitatively assess generation quality

License
MIT

Acknowledgments
Hugging Face for their transformers library

OpenAI for the original GPT-2 architecture

The Wikitext dataset creators
