from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformer, SentenceTransformerTrainer
from huggingface_hub import login
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

model_name = "Alibaba-NLP/gte-large-en-v1.5"
base="gte-large-en-v1.5"

output_dir =  f".models/finetuned--{base}"

# Training arguments
num_epochs = 3
batch_size = 4
lr = 2e-5

train_args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=4,
    learning_rate=lr,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=150,
    logging_steps=10,
    save_strategy="steps",
    save_steps=150,
    seed=42,
    save_total_limit = 1,
    logging_dir=f"{output_dir}/logs",
    # dataloader_num_workers=0
)

# Dataset
dataset = load_dataset("Lauther/embeddings-train-semantic")
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Model
model = SentenceTransformer(model_name, device="cuda", trust_remote_code=True)

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

# checkpoint_dir=f"{output_dir}/checkpoint-200"
checkpoint_dir=None

trainer.train(resume_from_checkpoint=checkpoint_dir)

# Push to hub
login(token="xxx")

hub_name=f"emb-{base}-{num_epochs}e"
model.push_to_hub(f"Lauther/{hub_name}")
