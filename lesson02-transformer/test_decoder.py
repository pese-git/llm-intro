import torch
import pytest
from decoder import Decoder
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from torch import nn

class TestDecoder:
    @pytest.fixture
    def sample_decoder(self):
        """Фикстура с тестовым декодером"""
        return Decoder(
            num_heads=4,
            emb_size=64,
            head_size=16,
            max_seq_len=128,
            dropout=0.1
        )

    def test_initialization(self, sample_decoder):
        """Тест инициализации слоёв"""
        assert isinstance(sample_decoder._heads, MultiHeadAttention)
        assert isinstance(sample_decoder._ff, FeedForward)
        assert isinstance(sample_decoder._norm1, nn.LayerNorm)
        assert isinstance(sample_decoder._norm2, nn.LayerNorm)
        assert sample_decoder._norm1.normalized_shape == (64,)
        assert sample_decoder._norm2.normalized_shape == (64,)

    def test_forward_shapes(self, sample_decoder):
        """Тест сохранения размерностей"""
        test_cases = [
            (1, 10, 64),   # batch=1, seq_len=10
            (4, 25, 64),   # batch=4, seq_len=25 
            (8, 50, 64)    # batch=8, seq_len=50
        ]
        
        for batch, seq_len, emb_size in test_cases:
            x = torch.randn(batch, seq_len, emb_size)
            output = sample_decoder(x)
            assert output.shape == (batch, seq_len, emb_size)

    def test_masking(self, sample_decoder):
        """Тест работы маски внимания"""
        x = torch.randn(2, 10, 64)
        mask = torch.tril(torch.ones(10, 10))  # Нижнетреугольная маска
        
        output = sample_decoder(x, mask)
        assert not torch.isnan(output).any()

    def test_sample_case(self):
        """Тест на соответствие sample-тесту из условия"""
        torch.manual_seed(0)
        decoder = Decoder(
            num_heads=5,
            emb_size=12,
            head_size=8,
            max_seq_len=20,
            dropout=0.0
        )
        
        # Инициализируем веса нулями для предсказуемости
        for p in decoder.parameters():
            nn.init.zeros_(p)
        
        x = torch.zeros(1, 12, 12)
        output = decoder(x)
        
        # Проверка формы вывода
        assert output.shape == (1, 12, 12)
        
        # Проверка что выход близок к нулю (но не точное значение)
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6), \
            "Output should be close to zero with zero-initialized weights"

    def test_gradient_flow(self, sample_decoder):
        """Тест потока градиентов"""
        x = torch.randn(2, 15, 64, requires_grad=True)
        output = sample_decoder(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
