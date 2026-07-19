import os
import shutil
import numpy as np
from PIL import Image

def generate_dummy_dataset(base_dir="data/PlantVillage", samples_per_class=10):
    """
    Generates a small synthetic dataset for testing the CV training pipeline
    without needing to download the massive 2GB dataset.
    """
    from src.ml.cv_pipeline import DISEASE_CLASSES
    
    print(f"Generating synthetic PlantVillage dataset in {base_dir}...")
    
    # Create train and val directories
    for split in ['train', 'val']:
        for disease in DISEASE_CLASSES:
            path = os.path.join(base_dir, split, disease)
            os.makedirs(path, exist_ok=True)
            
            # Generate dummy images (random noise)
            for i in range(samples_per_class):
                # Create a 224x224 RGB image with some random noise
                # We'll make it slightly different per class just to let the model learn *something*
                class_idx = DISEASE_CLASSES.index(disease)
                color = (
                    (class_idx * 50) % 255, 
                    (class_idx * 100) % 255, 
                    (class_idx * 150) % 255
                )
                
                img_array = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
                img_array[:, :, 0] = np.clip(img_array[:, :, 0] + color[0], 0, 255)
                img_array[:, :, 1] = np.clip(img_array[:, :, 1] + color[1], 0, 255)
                img_array[:, :, 2] = np.clip(img_array[:, :, 2] + color[2], 0, 255)
                
                img = Image.fromarray(img_array)
                img.save(os.path.join(path, f"dummy_{i}.jpg"))
                
    print("Synthetic dataset generated successfully.")

def download_from_kaggle():
    """
    Instructions/placeholder for real Kaggle download.
    """
    print("To download the real dataset, ensure you have the Kaggle API installed:")
    print("  pip install kaggle")
    print("And your kaggle.json configured.")
    print("Then run: kaggle datasets download -d emmarex/plantdisease -p data/ --unzip")

if __name__ == "__main__":
    # By default, we generate a dummy dataset for testing
    generate_dummy_dataset()
