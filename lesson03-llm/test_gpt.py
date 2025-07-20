import torch
import pytest
from gpt import GPT

class TestGPT:
    @pytest.fixture
    def default_config(self):
        return {
            'vocab_size': 1000,
            'max_seq_len': 128,
            'emb_size': 256,
            'num_heads': 4,
            'head_size': 64,
            'num_layers': 2,
            'dropout': 0.1
        }

    @pytest.fixture
    def sample_input(self):
        return torch.randint(0, 1000, (2, 32))  # batch_size=2, seq_len=32

    def test_initialization(self, default_config):
        """Проверка создания модели"""
        gpt = GPT(**default_config)
        assert isinstance(gpt, torch.nn.Module)
        assert len(gpt._decoders) == default_config['num_layers']

    def test_forward_pass(self, default_config, sample_input):
        """Тест прямого прохода"""
        gpt = GPT(**default_config)
        output = gpt(sample_input)
        assert output.shape == (2, 32, 1000)  # batch, seq_len, vocab_size

    def test_max_length(self, default_config):
        """Проверка обработки максимальной длины"""
        gpt = GPT(**default_config)
        # Корректная длина
        x = torch.randint(0, 1000, (1, 128))
        output = gpt(x)
        # Слишком длинная последовательность
        with pytest.raises(ValueError):
            x = torch.randint(0, 1000, (1, 129))
            gpt(x)

if __name__ == "__main__":
    pytest.main(["-v"])