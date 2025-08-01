import torch
import torch.nn as nn
import os
import glob

# Import the final model class and architectural parameters from train.py
from train import PolicyValueNet, NUM_FILTERS, NUM_RESIDUAL_BLOCKS

# --- Configuration ---
DEVICE = torch.device("cpu") # Exporting should always be done on the CPU
EXPORTED_MODEL_NAME = "reversi_model.onnx"

def get_latest_model_path():
    """Finds the path of the model with the highest iteration number."""
    list_of_files = glob.glob('models/reversi_model_iter_*.pt')
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return latest_file

def main():
    # Automatically find the latest trained model
    latest_model_path = get_latest_model_path()
    
    if not latest_model_path:
        print("Error: No trained models found in the 'models' directory.")
        print("Please run train.py first to generate a model file.")
        return
        
    print(f"Loading latest model for export: {os.path.basename(latest_model_path)}")
    
    # Instantiate the correct model architecture
    model = PolicyValueNet(
        num_residual_blocks=NUM_RESIDUAL_BLOCKS,
        num_filters=NUM_FILTERS
    ).to(DEVICE)
    
    # Load the trained weights
    model.load_state_dict(torch.load(latest_model_path, map_location=DEVICE))
    model.eval() # Set to evaluation mode

    # Create a dummy input tensor with the correct shape (Batch Size=1, Channels=6, Height=8, Width=8)
    # The batch size of 1 is important for tracing the model for export.
    dummy_input = torch.randn(1, 6, 8, 8, device=DEVICE)

    print(f"\nExporting model to ONNX format as '{EXPORTED_MODEL_NAME}'...")

    # Export the model
    torch.onnx.export(
        model,                          # The model to export
        dummy_input,                    # A sample input to trace the model's operations
        EXPORTED_MODEL_NAME,            # The name of the output file
        export_params=True,             # Store the trained weights in the model file
        opset_version=11,               # The ONNX version to use
        do_constant_folding=True,       # A common optimization
        input_names=['input'],          # A name for the model's input
        output_names=['policy', 'value']# Names for the model's outputs
    )
    
    print("\nExport successful!")
    print(f"Your model is ready. Place '{EXPORTED_MODEL_NAME}' in your web project folder.")

if __name__ == '__main__':
    main()