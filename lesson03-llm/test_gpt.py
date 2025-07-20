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

    def test_generate_basic(self, default_config, sample_input):
        """Тест базовой генерации"""
        gpt = GPT(**default_config)
        generated = gpt.generate(sample_input, max_new_tokens=10)
        assert generated.shape == (2, 42)  # Исходные 32 + 10 новых токенов

    def test_generate_empty(self, default_config):
        """Тест генерации с пустым входом"""
        gpt = GPT(**default_config)
        empty_input = torch.randint(0, 1000, (2, 0))
        with pytest.raises(IndexError):
            gpt.generate(empty_input, max_new_tokens=10)

    def test_generate_max_length(self, default_config):
        """Тест генерации с максимальной длиной последовательности"""
        gpt = GPT(**default_config)
        # Вход с максимальной длиной
        max_len_input = torch.randint(0, 1000, (2, 128))
        generated = gpt.generate(max_len_input, max_new_tokens=1)
        assert generated.shape == (2, 129)

    @pytest.mark.skip(reason="Требуется доработка генерации для поддержки детерминированности")
    def test_generate_deterministic(self, default_config):
        """Тест детерминированности генерации (при одинаковом seed)"""
        # Фиксируем seed для входа
        torch.manual_seed(42)
        gpt = GPT(**default_config)
        input_tensor = torch.randint(0, 1000, (1, 10))
        
        # Два вызова generate с одинаковым seed
        out1 = gpt.generate(input_tensor.clone(), max_new_tokens=5)
        out2 = gpt.generate(input_tensor.clone(), max_new_tokens=5)
        
        assert torch.equal(out1, out2), "Результаты генерации должны быть идентичными при одинаковых seed"

if __name__ == "__main__":
    pytest.main(["-v"])