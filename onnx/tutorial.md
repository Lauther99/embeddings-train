# HOW TO GET ONNX FILES?
Debes correr este codigo en la consola:

```cmd

optimum-cli export onnx --model Lauther/d4-embeddings-v3.0 --task feature-extraction --opset 14 onnx_model_d4_v3_tripletloss/

```

lo unico que cambia es el nombre del modelo y la ruta y listo!