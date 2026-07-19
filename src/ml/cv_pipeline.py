import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Dummy classes for PlantVillage
DISEASE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def get_cv_model(num_classes=len(DISEASE_CLASSES)):
    """
    Initializes a pre-trained ResNet50 model and modifies the final layer.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def process_image(image_bytes):
    """
    Applies necessary transforms to the uploaded image for ResNet inference.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_bytes).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    return tensor

def predict_disease(image_tensor, model=None):
    """
    Runs inference on the image tensor.
    If no model is provided, it simulates a prediction for demonstration purposes.
    """
    if model is None:
        # Simulate prediction
        import random
        import time
        time.sleep(1) # simulate compute time
        
        # Bias simulation slightly towards healthy for better UX unless it's obviously bad
        pred_idx = random.randint(0, len(DISEASE_CLASSES) - 1)
        confidence = random.uniform(0.75, 0.99)
        return DISEASE_CLASSES[pred_idx], confidence
        
    # Real inference block
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    return DISEASE_CLASSES[predicted_idx.item()], confidence.item()

def train_model(data_dir, epochs=10, batch_size=32, learning_rate=0.0001, model_save_path="cv_model_best.pth"):
    """
    Trains the ResNet model on the dataset located at data_dir.
    Assumes standard ImageFolder structure: data_dir/train and data_dir/val
    """
    import os
    import time
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    import torch.optim as optim

    print(f"Starting training loop on {data_dir} for {epochs} epochs...")
    
    # Transformations (Advanced Augmentation)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)), # Added Random Zoom
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets and Loaders
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Missing train/val splits in {data_dir}. Run the data downloader first.")
        
    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    val_dataset = ImageFolder(val_dir, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = get_cv_model(num_classes=len(train_dataset.classes))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{epochs} (LR: {current_lr})')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
                dataset_size = len(train_dataset)
            else:
                model.eval()
                dataloader = val_loader
                dataset_size = len(val_dataset)
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val':
                # Step the scheduler based on validation loss
                scheduler.step(epoch_loss)
                
                # Deep copy the model if accuracy improved
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Saved new best model with accuracy: {best_acc:.4f}")
                
        print()
        
    print(f'Training complete. Best val Acc: {best_acc:4f}')
    return model
