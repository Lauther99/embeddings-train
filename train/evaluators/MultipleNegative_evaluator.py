from sentence_transformers import SentenceTransformer
from datasets import load_dataset

path="/home/azureuser/projects/embeddings-train/.models/finetuned--d4-embeddings-v1.0-MultipleNegativesRankingLoss/checkpoint-370"

model = SentenceTransformer(path, device="cuda", trust_remote_code=True)

eval_dataset = load_dataset("Lauther/d4-embeddings", split="validation")

from collections import defaultdict
queries = dict()
corpus = dict()
relevant_docs = defaultdict(set)

query_id_counter = 0
doc_id_counter = 0

query_to_id = {}
doc_to_id = {}

for example in eval_dataset:
    query = example["sentence1"]
    doc = example["sentence2"]
    label = int(example["posneg"])  # o "posneg", si aún no lo has renombrado

    # Asigna un ID único si no existe
    if query not in query_to_id:
        query_to_id[query] = f"q{query_id_counter}"
        queries[query_to_id[query]] = query
        query_id_counter += 1

    if doc not in doc_to_id:
        doc_to_id[doc] = f"d{doc_id_counter}"
        corpus[doc_to_id[doc]] = doc
        doc_id_counter += 1

    # Solo los pares positivos se consideran relevantes
    if label == 1:
        qid = query_to_id[query]
        did = doc_to_id[doc]
        relevant_docs[qid].add(did)

from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
)

evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="ir_eval",
    show_progress_bar=True
)


res = evaluator(model)

print(res)