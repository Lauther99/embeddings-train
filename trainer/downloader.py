from sentence_transformers.losses import CosineSimilarityLoss
from huggingface_hub import login
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformer, SentenceTransformerTrainer
from datasets import load_dataset


model_name = "intfloat/e5-mistral-7b-instruct"
model = SentenceTransformer(model_name)