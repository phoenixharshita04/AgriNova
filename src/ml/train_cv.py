import argparse
import os
from src.data.download_plantvillage import generate_dummy_dataset
from src.ml.cv_pipeline import train_model

def main():
    parser = argparse.ArgumentParser(description="Train the Computer Vision model for Leaf Disease Detection.")
    parser.add_argument('--data-dir', type=str, default='data/PlantVillage', help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--dummy-data', action='store_true', help='Generate and use a dummy dataset if the real one is missing')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(os.path.join(args.data_dir, 'train')):
        if args.dummy_data:
            print("Dataset not found. Generating dummy data for testing...")
            generate_dummy_dataset(base_dir=args.data_dir, samples_per_class=5)
        else:
            print(f"Error: Dataset not found at {args.data_dir}.")
            print("Please run the data downloader script or use the --dummy-data flag to test the pipeline.")
            return

    print("=== Starting CV Model Training ===")
    trained_model = train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print("=== Training Completed ===")
    print(f"Best model weights saved to cv_model_best.pth")

if __name__ == '__main__':
    main()
