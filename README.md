# ResNet34 Fine-tuning

## Descripción
Este proyecto realiza un fine-tuning del modelo ResNet-34 utilizando el dataset de Animal Faces. El objetivo es clasificar imágenes en tres clases diferentes.

## Requisitos
- Python 3.12
- PyTorch
- TensorBoard

## Configuración del entorno
1. Crear un entorno virtual con Python:
    - `python -m venv venv`
    - `source venv/bin/activate`  *(En Windows usar `venv\Scripts\activate`)*

2. Instalar PyTorch:
    - PyTorch se puede descargar desde su [página oficial](https://pytorch.org/get-started/locally/).

3. Instalar TensorBoard:
    - `pip install tensorboard`

## Descarga de datos
Descargar el dataset de Animal Faces desde [aquí](https://www.kaggle.com/datasets/andrewmvd/animal-faces) y descomprimirlo en una carpeta de tu elección.

## Preparación de datos
1. Copiar los datos a la estructura de carpetas deseada utilizando la función `copy_data` de `utils.py`:
    - `utils.copy_data(source_path="ruta/a/dataset", dest_folder="ruta/a/destino")`

2. Crear el archivo de metadatos utilizando la función `create_metadata` de `utils.py`:
    - `utils.create_metadata(dataset_folder_path="ruta/a/destino")`

3. Dividir los datos en entrenamiento, validación y prueba utilizando la función `split_dataset` de `utils.py`:
    - `utils.split_dataset(config)`

4. Modificar el archivo de configuración JSON para definir los porcentajes de división y las rutas de los archivos de metadatos.

## Pruebas
1. Probar la clase `ImageDataset` en `dataset.py`:
    - `python dataset.py`

2. Probar la función `test` en `split_data.py`:
    - `python split_data.py`

## Configuración
Modificar los archivos de configuración JSON en la carpeta `configs` siguiendo la misma estructura de los archivos existentes (ej. `exp01_config.json` y `exp2_config.json`).

## Entrenamiento y Evaluación
1. Entrenar el modelo:
    - `python main.py --config exp01_config.json --train`

2. Evaluar el modelo:
    - `python main.py --config exp01_config.json --test --epoch <número_de_epoch>`

## Resultados
Los resultados del entrenamiento y evaluación se guardarán en las carpetas `checkpoints` y `logs` dentro de la carpeta del experimento correspondiente.
