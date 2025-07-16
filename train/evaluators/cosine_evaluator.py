from sentence_transformers import SentenceTransformer
from datasets import load_dataset

eval_dataset = load_dataset("Lauther/measuring-embeddings-v5", split="validation")


model_name = "intfloat/e5-mistral-7b-instruct"
model_name = "sentence-transformers/all-mpnet-base-v2"
model_name = "sentence-transformers/all-distilroberta-v1"
model_name = "NovaSearch/stella_en_1.5B_v5"
model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
model_name = "NovaSearch/stella_en_400M_v5"
model_name = "jxm/cde-small-v2"
model_name = "Alibaba-NLP/gte-large-en-v1.5"
model_name = "intfloat/multilingual-e5-large-instruct"


path="/home/azureuser/projects/embeddings-train/.models/finetuned--measuring-embeddings-v5.3/checkpoint-1120"

model = SentenceTransformer(path, device="cuda", trust_remote_code=True)

from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
)

dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    name="sts_dev",
    show_progress_bar=True
)

results = dev_evaluator(model)

print("\n\n", model_name)
print(dev_evaluator.primary_metric)
# => "sts_dev_pearson_cosine"
print(results[dev_evaluator.primary_metric])
# => 0.881019449484294
