from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformer, SentenceTransformerTrainer
from huggingface_hub import login
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss, CoSENTLoss
import torch


model_name = "/home/azureuser/projects/embeddings-train/.models/finetuned--d4-embeddings-v1.0-ContrastiveLoss/checkpoint-580"

model = SentenceTransformer(model_name, device="cuda", trust_remote_code=True)

# Push to hub
login(token="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")

hub_name=f"d4-embeddings-v2.0"
model.push_to_hub(f"Lauther/{hub_name}")
