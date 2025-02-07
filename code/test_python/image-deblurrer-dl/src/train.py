"""
Script to train the model
"""

# Importing the required dependencies
import os
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import time 
import models
import yaml
import multiprocessing
import utils

# Importing the required packages
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

# https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch
import gc
gc.collect()

torch.cuda.memory_reserved()
torch.cuda.memory_allocated()
torch.cuda.empty_cache()
print('Memory in use:')
print(torch.cuda.memory_summary(device=None, abbreviated=False))

# torch.backends.cudnn.benchmark = True # Esto permite que PyTorch seleccione la mejor implementación de CUDA para la arquitectura de tu GPU (False para desactivar)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Esto ayuda a reutilizar la memoria de la GPU en vez de fragmentarla: https://github.com/pytorch/pytorch/issues/16417

# Read parameters
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

epochs = config.get("epochs", 60) 
path = config.get("path", "C:/Users/ANIRUDH/OneDrive/Desktop/Image Deblurring App")
dataset_path = config.get("dataset_path", "/home/alejandro/Descargas/deblurring_datasets/the-deblurring-dataset")
model_name = config.get("model", "SimpleAE")
num_workers = config.get("num_workers", "auto")
if num_workers == "auto":
    num_workers = max(1, int(multiprocessing.cpu_count() / 2)) 
else:
    num_workers = int(num_workers)
pin_memory = config.get("pin_memory", True)

# helper functions
image_dir = path + '/outputs/saved_images'
os.makedirs(image_dir, exist_ok=True)

# Functions for viewing the image in size of 224 x 224
def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

# To check for availability of GPU memory on the machine
first_gpu_available = 'cuda:0'
device = first_gpu_available if torch.cuda.is_available() else 'cpu'

# Directories for training images and CNN, Autoencoders models
gauss_blur = os.listdir(os.path.join(dataset_path, 'blurred'))
gauss_blur.sort()
sharp = os.listdir(os.path.join(dataset_path, 'sharp'))
sharp.sort()

# This is used for checking that whether the blur image is regarding to the corresponding sharp image.
x_blur = []
for i in range(len(gauss_blur)):
    x_blur.append(gauss_blur[i])

y_sharp = []
for i in range(len(sharp)):
    y_sharp.append(sharp[i])

# print(x_blur[10])
# print(y_sharp[10])

# Train and Test split with 20% to be used as test dataset
(x_train, x_val, y_train, y_val) = train_test_split(x_blur, y_sharp, test_size=0.20)

print("Train images: ", len(x_train))
print("Test images: ", len(x_val))

# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), # ResNet and VGG input size standard
    transforms.ToTensor()
])

# Deblurring transformations
class DeblurDataset(Dataset):
    def __init__(self, blur_paths, sharp_paths=None, transforms=None):
        self.X = blur_paths
        self.y = sharp_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        blur_image = cv2.imread(os.path.join(dataset_path, "blurred", self.X[i]))
        
        if self.transforms:
            blur_image = self.transforms(blur_image)  # Transformar imagen de entrada a 224x224

        if self.y is not None:
            sharp_image = cv2.imread(os.path.join(dataset_path, "sharp", self.y[i]))
            sharp_image = self.transforms(sharp_image)
            return (blur_image, sharp_image)
        else:
            return blur_image

# Used to load and generate the image into tensors and arrays of size 224x224
train_data = DeblurDataset(x_train, y_train, transform)
val_data = DeblurDataset(x_val, y_val, transform)

print(f'Number of available workers: {multiprocessing.cpu_count()}')

# Model to be used 
if model_name == "SimpleAE":
    model = models.SimpleAE().to(device)
elif model_name == "CNN":
    model = models.CNN().to(device)
elif model_name == "AutoCNN":
    model = models.AutoCNN().to(device)
elif model_name == "DeblurNet":
    model = models.DeblurNet().to(device)
else:
    raise ValueError(f"Model '{model_name}' not recognized in config.yaml")

print(f"Usando el modelo: {model_name}")
print(model)

print("¿CUDA disponible?", torch.cuda.is_available())
print("Dispositivo en uso:", device)
print("Modelo en:", next(model.parameters()).device)

# Batch Size of images
# batch_size = 32
batch_size = utils.find_max_batch_size(model, train_data)

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode = 'min',
    patience=5,
    factor=0.1
)

# optimizer function
def fit(model, dataloader, epoch):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        blur_image, sharp_image = data # Extraemos imágenes borrosas y nítidas
        
        # Carga de datos iterativa en CUDA, evitando mover todo a la GPU de golpe
        if torch.cuda.is_available():
            blur_image = blur_image.to(device, non_blocking=True)
            sharp_image = sharp_image.to(device, non_blocking=True)

        # print(f"Batch {i}: blur_image en {blur_image.device}, sharp_image en {sharp_image.device}")
        
        optimizer.zero_grad()
        outputs = model(blur_image)
        loss = criterion(outputs, sharp_image)

        # backpropagation
        loss.backward()

        # update the parameters
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss/len(dataloader.dataset)
    print(f"Train Loss: {train_loss:.5f}")

    return train_loss

# the training function
def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            blur_image, sharp_image = data  # Extraemos imágenes
            
            # Carga de datos iterativa en CUDA, evitando mover todo a la GPU de golpe
            if torch.cuda.is_available():
                blur_image = blur_image.to(device, non_blocking=True)
                sharp_image = sharp_image.to(device, non_blocking=True)

            # print(f"Validación - Batch {i}: blur_image en {blur_image.device}, sharp_image en {sharp_image.device}")

            outputs = model(blur_image)
            loss = criterion(outputs, sharp_image)
            running_loss += loss.item()

            # based on the epoch number used for training and evaluation
            if epoch == 0 and i == (len(val_data) / dataloader.batch_size) - 1:
                save_decoded_image(sharp_image.cpu(), name=os.path.join(path, "outputs", "saved_images", f"sharp{epoch}.jpg"))
                save_decoded_image(blur_image.cpu(), name=os.path.join(path, "outputs", "saved_images", f"blur{epoch}.jpg"))

        val_loss = running_loss/len(dataloader.dataset)
        print(f"Val Loss: {val_loss:.5f}")

        save_decoded_image(outputs.cpu().data, name=os.path.join(path, "outputs", "saved_images", f"val_deblurred{epoch}.jpg"))

        return val_loss

# Evaluating and Plotting loss function
train_loss = []
val_loss = []
start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, trainloader, epoch)
    val_epoch_loss = validate(model, valloader, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    scheduler.step(val_epoch_loss)
end = time.time()

print(f"Took {((end-start)/60):.3f} minutes to train")

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(path + '/outputs/loss.png')
plt.show()

# Save the model to disk in PyTorch format
print('Saving model...')
torch.save(model.state_dict(), path + '/outputs/model.pth')
