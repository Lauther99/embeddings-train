from sentence_transformers import SentenceTransformer
from datasets import load_dataset

path="/home/azureuser/projects/embeddings-train/.models/finetuned--d4-embeddings-v2.0-ContrastiveLoss/checkpoint-880"
path="/home/azureuser/projects/embeddings-train/.models/finetuned--d4-embeddings-v1.0-ContrastiveLoss/checkpoint-580"

model = SentenceTransformer(path, device="cuda", trust_remote_code=True)

eval_dataset = load_dataset("Lauther/d4-embeddings", split="test")

queries = {}
corpus = {}
relevant_docs = {}

for idx, row in enumerate(eval_dataset):
    qid = f"q{idx}"
    did = f"d{idx}"

    queries[qid] = row["sentence1"]
    corpus[did] = row["sentence2"]

    relevant_docs[qid] = {did}

from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
)

evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    show_progress_bar=True,
    name="ir_eval",
)

res = evaluator(model)

print(res)