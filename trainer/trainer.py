from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformer, SentenceTransformerTrainer
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss, CoSENTLoss
import torch

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.cuda.empty_cache()

def print_gpu_memory():
    print(f"Memoria usada: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Memoria reservada: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

print_gpu_memory()

model_name = "intfloat/multilingual-e5-large-instruct"

base="measuring-embeddings-v5.3"
output_dir =  f".models/finetuned--{base}"
log_dir = f"{output_dir}/logs"

# Training arguments
num_epochs = 20
batch_size = 8
lr = 5e-6

train_args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=8,
    learning_rate=lr,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=150,
    logging_steps=50,
    save_strategy="steps",
    save_steps=150,
    max_grad_norm=0.5,
    weight_decay=0.01,
    seed=42,
    lr_scheduler_type="cosine",
    save_total_limit = 1,
    report_to="wandb"
)

# Dataset
dataset = load_dataset("Lauther/measuring-embeddings-v5")
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Model
from transformers import AutoModel, AutoTokenizer
_model = AutoModel.from_pretrained(model_name)
_model.config.hidden_dropout_prob = 0.2
_model.config.attention_probs_dropout_prob = 0.2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SentenceTransformer(model_name, device="cuda", trust_remote_code=True)
model._first_module().auto_model = _model

# Loss function
loss = CoSENTLoss(model)

print_gpu_memory()

# Train
trainer = SentenceTransformerTrainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    # evaluator=dev_evaluator,
)

checkpoint_dir=f"/home/azureuser/projects/embeddings-train/.models/finetuned--measuring-embeddings-v5.2/checkpoint-"
checkpoint_dir="/home/azureuser/projects/embeddings-train/.models/finetuned--measuring-embeddings-v5.3/checkpoint-300"

trainer.train(resume_from_checkpoint=checkpoint_dir)


# 