# Image Deblurring App

## Getting Started
1. Clone the repository into your local system
```bash
git clone https://github.com/agarnung/image-deblurring-exploration.git
```
2. Install required dependencies (recommended in a venv):
```bash
pip install pipreqs
```
(generated with `$ pipreqs`)
3. Specify your custom configuration for training and testing the model:
```bash
nano ./config.yaml 
```
It is worth mentioning that the model that will be used is exactly /```outputs/model.pth```. You can rename any pretrained model to this name and specify in the config which architecture it corresponds to.
4. To run the application first you have to train the model:
```py
python train.py
```
5. After completion of execution, you can now test the model running an application:
```py
python test.py
```

That's it! Now test the model to deblur any image.

# Troubleshooting

* To monitorize GPU in real time:
```bash
watch -n 1 nvidia-smi
```

# TODO
* Pensar cómo usar el config.yaml en realtivo, sin falta de asumir que se ejecuta desde ./src el código para encontrarlo
* Poner más parámetros por configuración (batchsize [nº o auto, con lo de ultralytics], numWorkers [nº o auto, la mitad del máximo], etc.)
* Hacer bien la conversión directa para mostrar deblurred_image en la app, sin falta de guardarla en archivo

