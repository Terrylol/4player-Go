
import torch
import torch.onnx
import os
from network import AlphaZeroNet

def export_5x5():
    board_size = 5
    model_dir = "models_5x5"
    model_path = os.path.join(model_dir, "latest_model.pth")
    output_path = "../public/models/model_5x5.onnx"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load model
    model = AlphaZeroNet(board_size=board_size)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded weights from {model_path}")
    except FileNotFoundError:
        print(f"Warning: {model_path} not found. Exporting random initialized model.")
    
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 5, board_size, board_size)

    # Export
    torch.onnx.export(model, 
                      dummy_input, 
                      output_path, 
                      verbose=False,
                      input_names=['input'], 
                      output_names=['policy', 'value'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'policy': {0: 'batch_size'},
                                    'value': {0: 'batch_size'}})
    print(f"Exported to {output_path}")

if __name__ == "__main__":
    export_5x5()
