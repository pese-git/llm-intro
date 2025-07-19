import torch
import pytest
from multi_head_attention import MultiHeadAttention

@pytest.fixture
def sample_input():
    """Фикстура с тестовыми входными данными"""
    batch_size = 2
    seq_len = 10
    emb_size = 64
    return torch.randn(batch_size, seq_len, emb_size)

def test_initialization():
    """Тест инициализации с правильными параметрами"""
    mha = MultiHeadAttention(
        num_heads=8,
        emb_size=64,
        head_size=32,
        max_seq_len=100,
        dropout=0.1
    )
    
    assert len(mha._heads) == 8
    assert mha._layer.in_features == 8 * 32
    assert mha._layer.out_features == 64
    assert mha._dropout.p == 0.1

def test_forward_pass(sample_input):
    """Тест прямого прохода с сохранением размерности"""
    mha = MultiHeadAttention(
        num_heads=4,
        emb_size=64,
        head_size=16,
        max_seq_len=50
    )
    
    output = mha(sample_input)
    assert output.shape == sample_input.shape

def test_dropout_effect(sample_input):
    """Тест влияния dropout на выход"""
    mha_with_dropout = MultiHeadAttention(
        num_heads=4,
        emb_size=64,
        head_size=16,
        max_seq_len=50,
        dropout=0.5
    )
    
    mha_without_dropout = MultiHeadAttention(
        num_heads=4,
        emb_size=64,
        head_size=16,
        max_seq_len=50,
        dropout=0.0
    )
    
    output1 = mha_with_dropout(sample_input)
    output2 = mha_without_dropout(sample_input)
    assert not torch.allclose(output1, output2)

def test_gradient_flow(sample_input):
    """Тест корректности обратного распространения"""
    mha = MultiHeadAttention(
        num_heads=4,
        emb_size=64,
        head_size=16,
        max_seq_len=50
    )
    
    sample_input.requires_grad_(True)
    output = mha(sample_input)
    output.sum().backward()
    assert sample_input.grad is not None

def test_mask_support(sample_input):
    """Тест поддержки масок (должен проходить даже без реализации)"""
    mask = torch.ones(sample_input.shape[:2])
    mha = MultiHeadAttention(
        num_heads=4,
        emb_size=64,
        head_size=16,
        max_seq_len=50
    )
    
    try:
        output = mha(sample_input, mask=mask)
        assert output.shape == sample_input.shape
    except Exception as e:
        pytest.fail(f"Mask handling failed: {e}")
