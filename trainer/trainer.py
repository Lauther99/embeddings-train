from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.trainer import BatchSamplers
from huggingface_hub import login
from datasets import load_dataset

num_epochs = 1
batch_size = 16
lr = 2e-5
finetuned_model_name =  "e5-mistral-embeddings"

login(token="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")

train_args = SentenceTransformerTrainingArguments(
    output_dir=f"models/{finetuned_model_name}",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_gpu_eval_batch_size=batch_size,
    learning_rate=lr,
    warmup_ratio=0.1,
    # batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=10,
)

model_name = "intfloat/e5-mistral-7b-instruct"
model = SentenceTransformer(model_name)

dataset = load_dataset("Lauther/embeddings-train-semantic")
train_dataset = dataset["train"]
eval_dataset = dataset["test"]


trainer = SentenceTransformerTrainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    # evaluator=evaluator_valid,
)

trainer.train()

