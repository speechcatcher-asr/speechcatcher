"""Tests for ESPnetASRModel."""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from speechcatcher.model import ESPnetASRModel


def test_build_model():
    """Test building an ESPnet ASR model."""
    print("\n=== Testing Model Building ===")

    vocab_size = 1000
    model = ESPnetASRModel.build_model(
        vocab_size=vocab_size,
        input_size=80,
        encoder_output_size=128,
        encoder_attention_heads=4,
        encoder_num_blocks=2,
        decoder_attention_heads=4,
        decoder_num_blocks=2,
        use_frontend=False,  # Skip frontend for simplicity
        use_ctc=True,
        use_decoder=True,
    )

    assert model.vocab_size == vocab_size
    assert model.encoder is not None
    assert model.decoder is not None
    assert model.ctc is not None

    print(f"✓ Vocab size: {model.vocab_size}")
    print(f"✓ Encoder: {type(model.encoder).__name__}")
    print(f"✓ Decoder: {type(model.decoder).__name__}")
    print(f"✓ CTC: {type(model.ctc).__name__}")
    print("✓ Model building passed")


def test_forward_inference():
    """Test forward pass in inference mode (no targets)."""
    print("\n=== Testing Forward Inference ===")

    batch_size = 2
    time_steps = 100
    feat_dim = 80
    vocab_size = 100

    model = ESPnetASRModel.build_model(
        vocab_size=vocab_size,
        input_size=feat_dim,
        encoder_output_size=128,
        encoder_num_blocks=2,
        decoder_num_blocks=2,
        use_frontend=False,
    )
    model.eval()

    # Input features
    speech = torch.randn(batch_size, time_steps, feat_dim)
    speech_lengths = torch.tensor([time_steps, time_steps // 2])

    with torch.no_grad():
        results = model(speech, speech_lengths)

    assert "encoder_out" in results
    assert "encoder_out_lens" in results
    assert "ctc_logits" in results
    assert results["encoder_out"].shape[0] == batch_size
    assert results["encoder_out"].shape[2] == 128  # encoder_output_size

    print(f"✓ Input shape: {speech.shape}")
    print(f"✓ Encoder output shape: {results['encoder_out'].shape}")
    print(f"✓ CTC logits shape: {results['ctc_logits'].shape}")
    print("✓ Forward inference passed")


def test_forward_training():
    """Test forward pass in training mode (with targets and loss)."""
    print("\n=== Testing Forward Training ===")

    batch_size = 2
    time_steps = 100
    feat_dim = 80
    vocab_size = 100
    target_len = 20

    model = ESPnetASRModel.build_model(
        vocab_size=vocab_size,
        input_size=feat_dim,
        encoder_output_size=128,
        encoder_num_blocks=2,
        decoder_num_blocks=2,
        use_frontend=False,
    )
    model.train()

    # Input features and targets
    speech = torch.randn(batch_size, time_steps, feat_dim)
    speech_lengths = torch.tensor([time_steps, time_steps // 2])
    text = torch.randint(1, vocab_size, (batch_size, target_len))
    text_lengths = torch.tensor([target_len, target_len // 2])

    results = model(speech, speech_lengths, text, text_lengths)

    assert "loss" in results, "Loss should be computed in training mode"
    assert "ctc_loss" in results
    assert "att_loss" in results
    assert "decoder_logits" in results

    print(f"✓ Input shape: {speech.shape}")
    print(f"✓ Target shape: {text.shape}")
    print(f"✓ Total loss: {results['loss'].item():.4f}")
    print(f"✓ CTC loss: {results['ctc_loss'].item():.4f}")
    print(f"✓ Attention loss: {results['att_loss'].item():.4f}")
    print(f"✓ Decoder logits shape: {results['decoder_logits'].shape}")
    print("✓ Forward training passed")


def test_encode_streaming():
    """Test encoder streaming interface."""
    print("\n=== Testing Encode Streaming ===")

    batch_size = 1
    chunk_size = 50
    feat_dim = 80
    vocab_size = 100

    model = ESPnetASRModel.build_model(
        vocab_size=vocab_size,
        input_size=feat_dim,
        encoder_output_size=128,
        encoder_num_blocks=2,
        use_frontend=False,
    )
    model.eval()

    # Simulate streaming with multiple chunks
    chunks = [torch.randn(batch_size, chunk_size, feat_dim) for _ in range(3)]
    chunk_lengths = torch.tensor([chunk_size])

    states = None
    outputs = []

    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            is_final = (i == len(chunks) - 1)
            encoder_out, encoder_out_lens, states = model.encode(
                chunk, chunk_lengths, prev_states=states, is_final=is_final
            )
            outputs.append(encoder_out)
            out_len = encoder_out_lens[0].item() if encoder_out_lens is not None else encoder_out.shape[1]
            print(f"✓ Chunk {i+1}: encoder_out shape {encoder_out.shape}, output_len {out_len}")

    print(f"✓ Streaming with {len(chunks)} chunks completed")
    print("✓ Encode streaming passed")


def test_model_only_ctc():
    """Test model with only CTC (no decoder)."""
    print("\n=== Testing CTC-only Model ===")

    batch_size = 2
    time_steps = 100
    feat_dim = 80
    vocab_size = 100
    target_len = 20

    model = ESPnetASRModel.build_model(
        vocab_size=vocab_size,
        input_size=feat_dim,
        encoder_output_size=128,
        encoder_num_blocks=2,
        use_frontend=False,
        use_ctc=True,
        use_decoder=False,  # No decoder
    )
    model.train()

    # Input and targets
    speech = torch.randn(batch_size, time_steps, feat_dim)
    speech_lengths = torch.tensor([time_steps, time_steps // 2])
    text = torch.randint(1, vocab_size, (batch_size, target_len))
    text_lengths = torch.tensor([target_len, target_len // 2])

    results = model(speech, speech_lengths, text, text_lengths)

    assert "ctc_loss" in results
    assert "loss" in results
    assert "att_loss" not in results, "Should not have attention loss without decoder"
    assert "decoder_logits" not in results, "Should not have decoder logits without decoder"

    print(f"✓ Loss: {results['loss'].item():.4f}")
    print(f"✓ CTC loss: {results['ctc_loss'].item():.4f}")
    print("✓ CTC-only model passed")


def test_model_gradient_flow():
    """Test gradient flow through the entire model."""
    print("\n=== Testing Gradient Flow ===")

    batch_size = 2
    time_steps = 50
    feat_dim = 80
    vocab_size = 50
    target_len = 10

    model = ESPnetASRModel.build_model(
        vocab_size=vocab_size,
        input_size=feat_dim,
        encoder_output_size=64,
        encoder_num_blocks=1,
        decoder_num_blocks=1,
        use_frontend=False,
    )

    # Input and targets
    speech = torch.randn(batch_size, time_steps, feat_dim, requires_grad=True)
    speech_lengths = torch.tensor([time_steps, time_steps // 2])
    text = torch.randint(1, vocab_size, (batch_size, target_len))
    text_lengths = torch.tensor([target_len, target_len // 2])

    # Forward
    results = model(speech, speech_lengths, text, text_lengths)
    loss = results["loss"]

    # Backward
    loss.backward()

    # Check gradients
    assert speech.grad is not None, "Input should have gradients"
    assert not torch.isnan(speech.grad).any(), "Gradients should not be NaN"

    # Check model parameters have gradients
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)

    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Parameters with gradients: {params_with_grad}/{total_params}")
    print(f"✓ Input gradient norm: {speech.grad.norm().item():.4f}")
    print("✓ Gradient flow passed")


def main():
    """Run all ESPnet ASR model tests."""
    print("=" * 80)
    print("Running ESPnet ASR Model Tests")
    print("=" * 80)

    try:
        test_build_model()
        test_forward_inference()
        test_forward_training()
        test_encode_streaming()
        test_model_only_ctc()
        test_model_gradient_flow()

        print("\n" + "=" * 80)
        print("ALL ESPNET ASR MODEL TESTS PASSED!")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
