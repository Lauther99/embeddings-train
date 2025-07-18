from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformer, SentenceTransformerTrainer
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss, CoSENTLoss, MultipleNegativesRankingLoss, ContrastiveLoss
import torch

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.cuda.empty_cache()

def print_gpu_memory():
    print(f"Memoria usada: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Memoria reservada: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

print_gpu_memory()

# Dataset

# Dataset para multiple negative loss
dataset = load_dataset("Lauther/d4-embeddings-MultipleNegative")
dataset = dataset.rename_columns({
    "sentence1": "anchor",
    "sentence2": "positive"
})
dataset = dataset.remove_columns("__index_level_0__")


train_dataset = dataset["train"]
test_dataset = dataset["test"]
print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))

# Model
from transformers import AutoModel, AutoTokenizer
# _model = AutoModel.from_pretrained(model_name)
# _model.config.hidden_dropout_prob = 0.2
# _model.config.attention_probs_dropout_prob = 0.2
# tokenizer = AutoTokenizer.from_pretrained(model_name)
model_name = "intfloat/multilingual-e5-large-instruct"
model = SentenceTransformer(model_name, device="cuda", trust_remote_code=True)
# model._first_module().auto_model = _model

# Loss function
# cosine_loss = CoSENTLoss(model)
multiple_negative_loss = MultipleNegativesRankingLoss(model)

print_gpu_memory()

# Train
# Training arguments
base="d4-embeddings-v1.0-MultipleNegativesRankingLoss"
output_dir =  f".models/finetuned--{base}"
log_dir = f"{output_dir}/logs"
num_epochs = 10
batch_size = 100
lr = 2e-5
logging_steps=5

train_args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    learning_rate=lr,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=150,
    logging_steps=logging_steps,
    save_strategy="steps",
    save_steps=150,
    max_grad_norm=0.5,
    weight_decay=0.01,
    seed=42,
    lr_scheduler_type="cosine",
    save_total_limit = 1,
    report_to="wandb",
    dataloader_num_workers=4,
    fp16=True 
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=multiple_negative_loss,
)

checkpoint_dir=f"/home/azureuser/projects/embeddings-train/.models/finetuned--measuring-embeddings-v5.2/checkpoint-"
checkpoint_dir="/home/azureuser/projects/embeddings-train/.models/finetuned--measuring-embeddings-v5.3/checkpoint-300"

trainer.train()
