from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformer, SentenceTransformerTrainer
from huggingface_hub import login
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss, CoSENTLoss
import torch


model_name = "/home/azureuser/projects/embeddings-train/.models/finetuned--measuring-embeddings-v5.3/checkpoint-1120"

model = SentenceTransformer(model_name, device="cuda", trust_remote_code=True)

# Push to hub
login(token="xxx")

hub_name=f"measuring-embeddings-v5.1"
model.push_to_hub(f"Lauther/{hub_name}")
