import torch
import torch.onnx
from network import AlphaZeroNet

def export(model_path="latest_model.pth", board_size=9):
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
                      "model.onnx", 
                      verbose=False,
                      input_names=['input'], 
                      output_names=['policy', 'value'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'policy': {0: 'batch_size'},
                                    'value': {0: 'batch_size'}})
    print("Exported to model.onnx")

if __name__ == "__main__":
    export()
