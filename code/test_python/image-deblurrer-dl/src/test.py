import sys
import cv2
import torch
import models
import yaml
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt
from torchvision.transforms import transforms
from torchvision.utils import save_image
import os 

# https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch
import gc
gc.collect()

torch.cuda.memory_reserved()
torch.cuda.memory_allocated()
torch.cuda.empty_cache()
print('Memory in use:')
print(torch.cuda.memory_summary(device=None, abbreviated=False))

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

path = config.get("path", "/home/")
model_name = config.get("model", "SimpleAE")
usual_images_folder = config.get("usual_images_folder", "/home/")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

class DeblurringApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Deblurring App")
        self.setFixedSize(1300, 700)  
        self.setStyleSheet("background-color: #f5f7fa;")

        self.image_file_path = None

        self.title_label = QLabel("Image Deblurring App", self)
        self.title_label.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.original_label = QLabel("Original Image", self)
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setStyleSheet("border: 2px solid #ddd; background-color: #eee; padding: 5px;")

        self.deblurred_label = QLabel("Deblurred Image", self)
        self.deblurred_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.deblurred_label.setStyleSheet("border: 2px solid #ddd; background-color: #eee; padding: 5px;")

        self.select_button = QPushButton("Select Image")
        self.select_button.setFont(QFont("Arial", 12))
        self.select_button.setStyleSheet(self.get_button_style())
        self.select_button.clicked.connect(self.choose_image_file)

        self.deblur_button = QPushButton("Deblur Image")
        self.deblur_button.setFont(QFont("Arial", 12))
        self.deblur_button.setStyleSheet(self.get_button_style())
        self.deblur_button.clicked.connect(self.deblur_image)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.original_label)
        image_layout.addWidget(self.deblurred_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.deblur_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title_label)
        main_layout.addLayout(image_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def get_button_style(self):
        return """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """

    def choose_image_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", usual_images_folder, "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_file_path = file_path
            self.display_image(file_path, self.original_label)

    def display_image(self, file_path, label):
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (600, 450)) 
        height, width, ch = img.shape
        bytes_per_line = ch * width
        qt_img = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_img))
        label.setScaledContents(True)

    def deblur_image(self):
        if not self.image_file_path:
            QMessageBox.warning(self, "Error", "Please select an image first.")
            return

        image = cv2.imread(self.image_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        if model_name == "SimpleAE":
            model = models.SimpleAE().to(device).eval()
        elif model_name == "CNN":
            model = models.CNN().to(device).eval()
        elif model_name == "AutoCNN":
            model = models.AutoCNN().to(device).eval()
        elif model_name == "DeblurNet":
            model = models.DeblurNet().to(device)
        else:
            QMessageBox.critical(self, "Error", f"Model '{model_name}' not recognized in config.yaml")
            return

        model.load_state_dict(torch.load(path + '/outputs/model.pth'))

        with torch.no_grad():
            outputs = model(image_tensor)
            save_decoded_image(outputs.cpu().data, name="deblurred_image.jpg")

        self.display_image("deblurred_image.jpg", self.deblurred_label)

        os.remove("deblurred_image.jpg")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DeblurringApp()
    window.show()
    sys.exit(app.exec())
