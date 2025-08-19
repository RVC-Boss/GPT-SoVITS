import os
import sys
import torch
import torchaudio
import onnxruntime as ort
import numpy as np
import argparse
from transformers import HubertModel, HubertConfig


class HubertONNXExporter:
    """Export and test HuBERT model to ONNX format"""
    
    def __init__(self, model_path="GPT_SoVITS/pretrained_models/chinese-hubert-base", output_path="playground/hubert/chinese-hubert-base.onnx"):
        self.model_path = model_path
        self.onnx_path = output_path
        self.model = None
        self.config = None
        
    def setup_model(self):
        """Configure and load the HuBERT model for ONNX export"""
        # Configure for better ONNX compatibility
        self.config = HubertConfig.from_pretrained(self.model_path)
        self.config._attn_implementation = "eager"  # Use standard attention
        self.config.apply_spec_augment = False      # Disable masking for inference
        self.config.layerdrop = 0.0                 # Disable layer dropout
        
        # Load the model
        self.model = HubertModel.from_pretrained(
            self.model_path, 
            config=self.config, 
            local_files_only=True
        )
        self.model.eval()
        
    def export_to_onnx(self, dummy_length=16000):
        """Export the model to ONNX format"""
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
            
        # Create dummy input (1 second at 16kHz)
        dummy_input = torch.rand(1, dummy_length, dtype=torch.float32) - 0.5
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            self.onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['audio16k'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'audio16k': {0: 'batch_size', 1: 'sequence_length'},
                'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        print(f"[Success] Model exported to {self.onnx_path}")
        
    def test_onnx_export_exists(self):
        """Test that the ONNX model file was created"""
        if os.path.exists(self.onnx_path):
            print(f"[Success] ONNX model file exists at {self.onnx_path}")
            return True
        else:
            print(f"[Error] ONNX model not found at {self.onnx_path}")
            return False
            
    def _load_and_preprocess_audio(self, audio_path, max_length=160000):
        """Load and preprocess audio file"""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Take first channel
        if waveform.shape[0] > 1:
            waveform = waveform[0:1]
        
        # Limit length for testing (10 seconds at 16kHz)
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]

        # make a zero tensor that has length 3200*0.3
        zero_tensor = torch.zeros((1, 9600), dtype=torch.float32)

        print("waveform shape and zero wave shape", waveform.shape, zero_tensor.shape)

        # concate zero_tensor with waveform
        waveform = torch.cat([waveform, zero_tensor], dim=1)

        return waveform
        
    def test_torch_vs_onnx(self, audio_path="playground/ref/audio.wav"):
        """Test that ONNX model outputs match PyTorch model outputs"""
        if not os.path.exists(audio_path):
            print(f"[Skip] Test audio file not found at {audio_path}")
            return False
            
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
            
        # Load and preprocess audio
        waveform = self._load_and_preprocess_audio(audio_path)
        
        # PyTorch inference
        with torch.no_grad():
            torch_output = self.model(waveform)
            torch_hidden_states = torch_output.last_hidden_state
        
        # ONNX inference
        ort_session = ort.InferenceSession(self.onnx_path)
        input_values = waveform.numpy().astype(np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: input_values}
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_hidden_states = ort_outputs[0]
        
        # Compare outputs
        torch_numpy = torch_hidden_states.numpy()
        diff = np.abs(torch_numpy - onnx_hidden_states).mean()
        
        success = diff <= 1e-5
        status = "[Success]" if success else "[Fail]"
        
        print(f"{status} ONNX vs PyTorch comparison")
        print(f" > mean_difference={diff}")
        print(f" > torch_shape={torch_numpy.shape}")
        print(f" > onnx_shape={onnx_hidden_states.shape}")
        
        return success
        
    def run_full_export_and_test(self):
        """Run the complete export and testing pipeline"""
        print("Starting HuBERT ONNX export and testing...")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.onnx_path), exist_ok=True)
        
        # Setup model
        self.setup_model()
        
        # Export to ONNX
        self.export_to_onnx()
        
        # Test export
        self.test_onnx_export_exists()
        self.test_torch_vs_onnx()
        
        print("Export and testing complete!")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Export HuBERT model to ONNX format")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="GPT_SoVITS/pretrained_models/chinese-hubert-base",
        help="Path to the HuBERT model directory (default: GPT_SoVITS/pretrained_models/chinese-hubert-base)"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="playground/hubert/chinese-hubert-base.onnx",
        help="Output path for the ONNX model (default: playground/hubert/chinese-hubert-base.onnx)"
    )
    
    args = parser.parse_args()
    
    exporter = HubertONNXExporter(model_path=args.model_path, output_path=args.output_path)
    exporter.run_full_export_and_test()


if __name__ == "__main__":
    main()