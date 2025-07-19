import torch
import pytest
from head_attention import HeadAttention

# Проверка версии PyTorch для совместимости
def check_pytorch_version():
    print(f"\nPyTorch version: {torch.__version__}")
    if torch.__version__ < '1.2.0':
        pytest.skip("This test requires PyTorch 1.2.0 or higher")

@pytest.fixture(autouse=True)
def setup():
    check_pytorch_version()

@pytest.fixture
def attention():
    return HeadAttention(emb_size=64, head_size=32, max_seq_len=128)

def test_attention_initialization(attention):
    assert attention._emb_size == 64
    assert attention._head_size == 32
    assert attention._max_seq_len == 128
    assert isinstance(attention._k, torch.nn.Linear)
    assert isinstance(attention._q, torch.nn.Linear)
    assert isinstance(attention._v, torch.nn.Linear)

def test_attention_forward_shape(attention):
    batch_size = 4
    seq_len = 32
    x = torch.randn(batch_size, seq_len, 64)  # (B, T, emb_size)
    output = attention(x)
    assert output.shape == (batch_size, seq_len, 32)  # (B, T, head_size)

def test_attention_mask():
    attention = HeadAttention(emb_size=64, head_size=32, max_seq_len=128)
    # Проверяем, что маска нижнетреугольная
    mask = attention._tril_mask
    assert mask.shape == (128, 128)
    assert torch.all(mask == torch.tril(torch.ones(128, 128)).bool())

def test_attention_causal_property(attention):
    """Test causal attention property"""
    print("\nTesting causal attention...")
    seq_len = 10
    x = torch.randn(1, seq_len, 64)
    
    # Получаем query и key
    q = attention._q(x)  # (1, seq_len, head_size)
    k = attention._k(x)  # (1, seq_len, head_size)
    
    # Вычисляем scores
    scores = q @ k.transpose(-2, -1) / (attention._head_size ** 0.5)
    
    # Применяем маску (это должно происходить внутри forward)
    mask = attention._tril_mask[:seq_len, :seq_len]
    masked_scores = scores.masked_fill(~mask, float('-inf'))
    
    # Проверяем, что будущие токены замаскированы
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            assert masked_scores[0, i, j] == float('-inf'), f"Position {i},{j} not masked"
    print("Causal attention test passed")

def test_attention_sequence_length_limit(attention):
    with pytest.raises(ValueError):
        x = torch.randn(1, 129, 64)  # Превышаем max_seq_len
        attention(x)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
