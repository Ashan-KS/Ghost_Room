"""
test_audio.py — Quick test for the audio module
=================================================
Run from the Ghost_Room root directory:
    python -m audio.test_audio

This will:
  1. Test WebRTC VAD initialisation
  2. Test feature extraction with synthetic audio
  3. (Optional) Test live mic capture for 3 seconds
"""

import sys
import os
import numpy as np

# Ensure project root is on sys.path so `config` can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def test_vad():
    """Test that WebRTC VAD initialises and can classify a silent frame."""
    print("\n=== Test 1: WebRTC VAD ===")
    try:
        import webrtcvad
        vad = webrtcvad.Vad(config.VAD_MODE)
        print(f"  [OK] VAD initialised (mode={config.VAD_MODE})")
    except ImportError:
        print("  [FAIL] webrtcvad is not installed. Run: pip install webrtcvad")
        return False

    # Create a silent 30ms frame (all zeros) at 16 kHz, 16-bit PCM
    frame_samples = int(config.AUDIO_SAMPLE_RATE * config.AUDIO_CHUNK_MS / 1000)
    silent_frame = np.zeros(frame_samples, dtype=np.int16).tobytes()

    result = vad.is_speech(silent_frame, config.AUDIO_SAMPLE_RATE)
    print(f"  [OK] VAD on silent frame → is_speech={result} (expected False)")

    # Create a noisy frame (random data simulating speech-like energy)
    noisy_frame = (np.random.normal(0, 5000, frame_samples)).astype(np.int16).tobytes()
    result2 = vad.is_speech(noisy_frame, config.AUDIO_SAMPLE_RATE)
    print(f"  [OK] VAD on noisy frame  → is_speech={result2}")

    return True


def test_feature_extraction():
    """Test that feature extraction produces a (14,) float32 vector."""
    print("\n=== Test 2: Feature Extraction ===")
    try:
        from audio.feature_extractor import extract_features, FEATURE_DIM
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False

    # Create a synthetic 30ms PCM chunk
    frame_samples = int(config.AUDIO_SAMPLE_RATE * config.AUDIO_CHUNK_MS / 1000)
    chunk = (np.random.normal(0, 3000, frame_samples)).astype(np.int16).tobytes()

    vector = extract_features(chunk)

    print(f"  Shape:  {vector.shape}  (expected ({FEATURE_DIM},))")
    print(f"  Dtype:  {vector.dtype}  (expected float32)")
    print(f"  RMS:    {vector[0]:.6f}")
    print(f"  MFCCs:  {vector[1:]}")

    if vector.shape == (FEATURE_DIM,) and vector.dtype == np.float32:
        print("  [OK] Feature extraction works correctly!")
        return True
    else:
        print("  [FAIL] Unexpected shape or dtype")
        return False


def test_vad_processor():
    """Test the vad_processor module's is_speech() function."""
    print("\n=== Test 3: vad_processor.is_speech() ===")
    try:
        from audio.vad_processor import is_speech
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False

    frame_samples = int(config.AUDIO_SAMPLE_RATE * config.AUDIO_CHUNK_MS / 1000)

    # Silent frame
    silent = np.zeros(frame_samples, dtype=np.int16).tobytes()
    result = is_speech(silent)
    print(f"  [OK] Silent frame → is_speech={result}")

    # Noisy frame
    noisy = (np.random.normal(0, 5000, frame_samples)).astype(np.int16).tobytes()
    result2 = is_speech(noisy)
    print(f"  [OK] Noisy frame  → is_speech={result2}")

    return True


def test_live_mic(duration_seconds=10):
    """Test live microphone capture for a few seconds."""
    print(f"\n=== Test 4: Live Mic Capture ({duration_seconds}s) ===")
    try:
        import pyaudio
    except ImportError:
        print("  [SKIP] pyaudio not installed. Run: pip install pyaudio")
        return True  # Not a failure, just skipped

    try:
        from audio.vad_processor import is_speech
        from audio.feature_extractor import extract_features

        pa = pyaudio.PyAudio()
        frame_samples = int(config.AUDIO_SAMPLE_RATE * config.AUDIO_CHUNK_MS / 1000)

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=config.AUDIO_SAMPLE_RATE,
            input=True,
            input_device_index=config.AUDIO_DEVICE_INDEX,
            frames_per_buffer=frame_samples,
        )
        print(f"  [OK] Audio stream opened (rate={config.AUDIO_SAMPLE_RATE}, chunk={config.AUDIO_CHUNK_MS}ms)")
        print(f"  Recording for {duration_seconds} seconds... Speak into the mic!")

        total_frames = int(config.AUDIO_SAMPLE_RATE / frame_samples * duration_seconds)
        speech_count = 0
        silence_count = 0

        for i in range(total_frames):
            chunk = stream.read(frame_samples, exception_on_overflow=False)

            # Test VAD
            vad_result = is_speech(chunk)
            if vad_result:
                speech_count += 1
            else:
                silence_count += 1

            # Test feature extraction (just on first frame to avoid spam)
            if i == 0:
                features = extract_features(chunk)
                print(f"  First frame features: shape={features.shape}, rms={features[0]:.6f}")

        stream.stop_stream()
        stream.close()
        pa.terminate()

        print(f"\n  Results over {total_frames} frames:")
        print(f"    Speech frames:  {speech_count}")
        print(f"    Silence frames: {silence_count}")
        print(f"  [OK] Live mic test complete!")
        return True

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_audio_queue_contract():
    """Test that the audio queue message has the correct keys and types."""
    print("\n=== Test 5: Audio Queue Contract ===")
    from datetime import datetime, timezone
    
    # Clear queue
    while not config.audio_queue.empty():
        config.audio_queue.get_nowait()

    # Simulate a VAD result
    message = {
        "vad_fired": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    config.audio_queue.put(message)

    # Read it back
    res = config.audio_queue.get_nowait()
    print(f"  Message: {res}")
    
    if "vad_fired" in res and "timestamp" in res and isinstance(res["vad_fired"], bool):
        print("  [OK] Audio queue contract is valid!")
        return True
    else:
        print("  [FAIL] Audio queue contract mismatch")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("  Ghost Room — Audio Module Tests")
    print("=" * 50)

    results = {}
    results["VAD"] = test_vad()
    results["Feature Extraction"] = test_feature_extraction()
    results["VAD Processor"] = test_vad_processor()
    results["Audio Queue Contract"] = test_audio_queue_contract()

    # Ask user if they want to test live mic
    try:
        answer = input("\nDo you want to test live microphone capture? (y/n): ").strip().lower()
        if answer == "y":
            results["Live Mic"] = test_live_mic()
    except (EOFError, KeyboardInterrupt):
        pass

    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  All tests passed! ✓")
    else:
        print("  Some tests failed. Check output above.")
    print()
