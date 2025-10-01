#!/usr/bin/env python3
"""
Test script for embedding vector conversion in LLaMA model.

Usage:
    python test_embedding_conversion.py
"""

import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath('.'))

from tsllm.models.llama_new import LLAMAmodel
from omegaconf import OmegaConf

def create_test_config():
    """Create a test configuration for the LLaMA model."""
    config = OmegaConf.create({
        'experiment': {
            'task': 'forecast'
        },
        'model': {
            'name': 'llama-7b-chat',
            'model_name': '7b-chat',
            'use_embeddings': True,
            'embedding_dim': 768,  # Smaller for testing
            'max_sequence_length': 32,
            'temp': 1.0,
            'settings': {
                'time_sep': ', ',
                'base': 10,
                'prec': 2
            }
        },
        'debug_mode': True
    })
    return config

def test_embedding_conversion():
    """Test the embedding conversion functionality."""
    print("Testing LLaMA embedding conversion...")

    # Create test config
    config = create_test_config()

    # Initialize model
    model = LLAMAmodel(config)

    # Test data (simplified time series strings)
    input_str = "627, 661, 739, 723, 681, 652, 690, 743, 725, 710"
    input_trend_str = "124, 126, 128, 130, 132, 134, 136, 138, 140, 142"
    input_season_str = "12, -5, 8, -2, 15, -8, 6, -3, 11, -7"
    input_resid_str = "3, -1, 2, 0, -4, 1, -2, 3, -1, 2"
    description = "Daily temperature measurements"

    print(f"Input string: {input_str}")
    print(f"Trend string: {input_trend_str}")
    print(f"Season string: {input_season_str}")
    print(f"Residual string: {input_resid_str}")

    # Test embedding conversion
    try:
        embeddings = model.convert_time_series_to_embeddings(
            input_str, input_trend_str, input_season_str, input_resid_str, description
        )

        print(f"\nEmbedding conversion successful!")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding dtype: {embeddings.dtype}")
        print(f"Embedding range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

        # Test parsing individual components
        main_values = model._parse_time_series_string(input_str)
        trend_values = model._parse_time_series_string(input_trend_str)
        print(f"\nParsed main values: {main_values[:5]}...")  # First 5 values
        print(f"Parsed trend values: {trend_values[:5]}...")

        # Test positional encoding
        seq_len = len(main_values)
        dummy_embeddings = torch.randn(seq_len, model.embedding_dim)
        pos_encoded = model._add_positional_encoding(dummy_embeddings)
        print(f"\nPositional encoding added successfully")
        print(f"Original shape: {dummy_embeddings.shape}")
        print(f"Encoded shape: {pos_encoded.shape}")

        return True

    except Exception as e:
        print(f"Error during embedding conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_loading():
    """Test that the configuration is properly loaded."""
    print("\nTesting configuration loading...")

    config = create_test_config()
    model = LLAMAmodel(config)

    print(f"use_embeddings: {model.use_embeddings}")
    print(f"embedding_dim: {model.embedding_dim}")
    print(f"max_sequence_length: {model.max_sequence_length}")

    # Test with embedding disabled
    config.model.use_embeddings = False
    model_disabled = LLAMAmodel(config)
    print(f"use_embeddings (disabled): {model_disabled.use_embeddings}")

if __name__ == "__main__":
    print("=" * 60)
    print("LLaMA Embedding Conversion Test")
    print("=" * 60)

    # Test configuration loading
    test_configuration_loading()

    # Test embedding conversion
    success = test_embedding_conversion()

    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
        print("\nTo enable embedding mode, set 'use_embeddings: True' in your model config:")
        print("tsllm/config/model/llama-2.yaml")
    else:
        print("❌ Tests failed!")
    print("=" * 60)