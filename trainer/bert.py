from datasets import load_dataset

# Cargar el dataset IMDB
dataset = load_dataset("imdb")
print(dataset)

from transformers import BertTokenizer

# Cargar el tokenizador de BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Función para tokenizar los textos
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Aplicar tokenización a todo el dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)


from transformers import BertForSequenceClassification

# Cargar el modelo BERT para clasificación de texto (2 etiquetas: positivo/negativo)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

from transformers import Trainer, TrainingArguments

# Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",          # Directorio para guardar los resultados
    evaluation_strategy="epoch",    # Evaluar al final de cada época
    learning_rate=2e-5,             # Tasa de aprendizaje
    per_device_train_batch_size=8,  # Tamaño del batch por dispositivo
    per_device_eval_batch_size=8,   # Tamaño del batch para evaluación
    num_train_epochs=3,             # Número de épocas
    weight_decay=0.01,              # Regularización (weight decay)
    save_steps=10_000,              # Guardar el modelo cada 10,000 pasos
    save_total_limit=2,             # Mantener solo los últimos 2 modelos guardados
)

# Crear el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo y el tokenizador
model.save_pretrained("./mi_modelo_finetune")
tokenizer.save_pretrained("./mi_modelo_finetune")

# Evaluar el modelo
eval_results = trainer.evaluate()
print(f"Precisión:\n {eval_results}")