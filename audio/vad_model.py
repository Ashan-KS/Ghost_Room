"""
audio/vad_model.py
===================
Silero VAD Adaptation & Compression Script for RPi 4.

This script implements the "Compress-and-Train" workflow:
1. Loads the pre-trained Silero VAD model.
2. Provides a DataLoader template for your 2-second audio chunks.
3. Prunes the model (removes 20% of weights).
4. (Placeholder) Fine-tunes the pruned model.
5. Performs Post-Training Quantization (PTQ) to INT8 using qnnpack.
"""

import os
import torch
import torch.nn.utils.prune as prune
import logging

log = logging.getLogger(__name__)

# Settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0  # seconds
NUM_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)


def get_vad_frame_samples(sample_rate: int) -> int:
    """Silero VAD accepts fixed frame sizes: 512@16kHz or 256@8kHz."""
    if sample_rate == 16000:
        return 512
    if sample_rate == 8000:
        return 256
    raise ValueError(f"Unsupported sample rate for Silero VAD: {sample_rate}")


def load_base_silero():
    """Loads the pre-trained Silero VAD model from PyTorch Hub."""
    log.info("Loading base Silero VAD model...")
    # Using torchaudio's cached hub to avoid re-downloading constantly
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    # The utils tuple contains helpful functions if we need them later
    return model


def get_calibration_dataloader(dataset_path: str, batch_size=16):
    """
    Loads the 2-second dataset for calibration/fine-tuning.
    Occupied = Room in use (Meeting noises, voices)
    Empty = Silent room / Baseline noise
    """
    from torch.utils.data import DataLoader, Dataset
    import torchaudio
    import glob
    
    class RoomAudioDataset(Dataset):
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.files = []
            
            # Load occupied (label = 1)
            occupied_files = glob.glob(os.path.join(root_dir, "occupied", "*.wav"))
            for f in occupied_files:
                self.files.append((f, 1))
                
            # Load empty (label = 0)
            empty_files = glob.glob(os.path.join(root_dir, "empty", "*.wav"))
            for f in empty_files:
                self.files.append((f, 0))
                
            log.info(f"Found {len(occupied_files)} occupied and {len(empty_files)} empty audios in {root_dir}")
            
        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            file_path, label = self.files[idx]
            try:
                waveform, sr = torchaudio.load(file_path)
                
                # Resample if needed
                if sr != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                    waveform = resampler(waveform)
                
                # Convert stereo to mono if needed
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Pad or truncate to exactly NUM_SAMPLES
                if waveform.shape[1] > NUM_SAMPLES:
                    waveform = waveform[:, :NUM_SAMPLES]
                elif waveform.shape[1] < NUM_SAMPLES:
                    pad = NUM_SAMPLES - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad))
                    
                # Silero expects shape (samples,) for individual items before batching, 
                # wait, DataLoader batches it to (batch, samples) if we squeeze the channel dimension
                waveform = waveform.squeeze(0)
                return waveform, label
            except Exception as e:
                log.error(f"Error loading {file_path}: {e}")
                return torch.zeros(NUM_SAMPLES), label
            
    log.info(f"Creating DataLoader from {dataset_path}")
    dataset = RoomAudioDataset(dataset_path)
    
    # Check if dataset is empty to gracefully handle testing
    if len(dataset) == 0:
        log.warning("Dataset is empty. Returning dummy dataloader fallback for testing.")
        return [(torch.randn(batch_size, NUM_SAMPLES), torch.randint(0, 2, (batch_size,))) for _ in range(5)]
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def prune_model(model, amount=0.2):
    """
    Applies L1 unstructured pruning to Convolutional layers.
    """
    log.info(f"Pruning {amount*100}% of Conv layer weights...")
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight') # Make permanent
    return model

def fine_tune_model(model, train_loader, epochs=3):
    """
    Fine-tunes the PyTorch model to recover accuracy lost from pruning.
    """
    log.info(f"Starting fine-tuning for {epochs} epochs...")
    import torch.nn as nn
    import torch.optim as optim
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    frame_samples = get_vad_frame_samples(SAMPLE_RATE)
    
    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0
        for i, (audio_batch, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            if hasattr(model, 'reset_states'):
                model.reset_states()
                
            frames = audio_batch.unfold(dimension=-1, size=frame_samples, step=frame_samples)
            batch_size = audio_batch.size(0)
            outputs = torch.zeros(batch_size, frames.shape[1], device=audio_batch.device)
            
            for frame_idx in range(frames.shape[1]):
                out = model(frames[:, frame_idx, :], SAMPLE_RATE)
                outputs[:, frame_idx] = out.squeeze(-1)
                
            chunk_prob = outputs.max(dim=1)[0]
            loss = criterion(chunk_prob, labels.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        avg_loss = total_loss / max(1, batches)
        log.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    return model


def quantize_model(model, calibration_loader):
    """
    Performs Post-Training Quantization (PTQ) to INT8.
    Prefers qnnpack (ARM/RPi) but falls back to available backends on dev machines.
    """
    log.info("Preparing model for PTQ...")
    
    # Must be in eval mode for quantization
    model.eval()
    
    # Pick the best available backend for this runtime.
    supported_engines = {str(e) for e in torch.backends.quantized.supported_engines}
    preferred_order = ("qnnpack", "x86", "fbgemm")
    selected_engine = next((e for e in preferred_order if e in supported_engines), None)

    if selected_engine is None:
        log.warning(
            "No supported quantized engine found (%s). Skipping PTQ and returning FP32 model.",
            sorted(supported_engines),
        )
        return model

    torch.backends.quantized.engine = selected_engine
    model.qconfig = torch.quantization.get_default_qconfig(selected_engine)
    log.info("Using quantized backend: %s", selected_engine)

    if selected_engine != "qnnpack":
        log.warning(
            "qnnpack is unavailable on this machine. PTQ will use %s instead. "
            "For Raspberry Pi deployment, quantize in an environment where qnnpack is supported.",
            selected_engine,
        )
    
    # Fusion (Optional but recommended if model architecture allows, not done here for simplicity)
    
    torch.quantization.prepare(model, inplace=True)
    
    log.info("Calibrating model with dataset...")
    frame_samples = get_vad_frame_samples(SAMPLE_RATE)
    with torch.no_grad():
        for i, (audio_batch, _) in enumerate(calibration_loader):
            # Silero VAD expects short frames: 512 (16kHz) / 256 (8kHz).
            # Convert each long calibration sample into non-overlapping valid frames.
            frames = audio_batch.unfold(dimension=-1, size=frame_samples, step=frame_samples)
            # frames shape: (batch, num_frames, frame_samples)
            for frame_idx in range(frames.shape[1]):
                model(frames[:, frame_idx, :], SAMPLE_RATE)
            if i > 10: # Just run a few batches for calibration
                break
                
    log.info("Converting model to INT8...")
    torch.quantization.convert(model, inplace=True)
    
    return model


def main():
    # 1. Load base Silero VAD
    print("Loading base Silero VAD model...")
    model = load_base_silero()
    
    # 2. Prune
    print("Pruning the model...")
    model = prune_model(model, amount=0.2)
    
    # 3. Load dataset for training and calibration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(current_dir, "processed_audio")
    data_loader = get_calibration_dataloader(DATASET_PATH)

    # 4. Fine-tune the model here using your dataset to recover accuracy
    print("Fine-tuning the model...")
    model = fine_tune_model(model, data_loader, epochs=3)
    
    # 5. Quantize to INT8
    # NOTE: PyTorch quantized models sometimes require the input to be `torch.quantize_per_tensor`
    # We will handle this in `vad_processor.py`.
    print("Quantizing the model to INT8...")
    quantized_model = quantize_model(model, data_loader)
    
    # 6. Save the optimized model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    save_path = os.path.join(base_dir, "models", "silero_vad_int8.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Using TorchScript (JIT) is highly recommended for edge deployment
    log.info(f"Saving quantized model to {save_path} using TorchScript...")
    
    # We trace it with a dummy input
    dummy_input = torch.randn(1, get_vad_frame_samples(SAMPLE_RATE))
    traced_model = torch.jit.trace(quantized_model, (dummy_input, SAMPLE_RATE))
    traced_model.save(save_path)
    
    log.info("Done! Model is ready for Edge Deployment on RPi 4.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
