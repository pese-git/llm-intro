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
        generated = gpt.generate(sample_input, max_new_tokens=10, do_sample=False)
        assert generated.shape == (2, 42)  # Исходные 32 + 10 новых токенов

    def test_generate_empty(self, default_config):
        """Тест генерации с пустым входом"""
        gpt = GPT(**default_config)
        empty_input = torch.randint(0, 1000, (2, 0))
        with pytest.raises(IndexError):
            gpt.generate(empty_input, max_new_tokens=10, do_sample=False)

    def test_generate_max_length(self, default_config):
        """Тест генерации с максимальной длиной последовательности"""
        gpt = GPT(**default_config)
        # Вход с максимальной длиной
        max_len_input = torch.randint(0, 1000, (2, 128))
        generated = gpt.generate(max_len_input, max_new_tokens=1, do_sample=False)
        assert generated.shape == (2, 129)

    def test_generate_with_sampling(self, default_config, sample_input):
        """Тест генерации с сэмплированием"""
        torch.manual_seed(42)
        gpt = GPT(**default_config)
        generated = gpt.generate(sample_input, max_new_tokens=10, do_sample=True)
        assert generated.shape == (2, 42)  # Исходные 32 + 10 новых токенов

    def test_temperature_effect(self, default_config):
        """Тест влияния температуры на генерацию"""
        torch.manual_seed(42)
        gpt = GPT(**default_config)
        gpt.eval()
        input_tensor = torch.randint(0, 1000, (1, 10))
        
        # Низкая температура делает распределение более "острым"
        low_temp = gpt.generate(input_tensor, max_new_tokens=5, do_sample=True, temperature=0.1)
        
        # Высокая температура делает распределение более равномерным
        high_temp = gpt.generate(input_tensor, max_new_tokens=5, do_sample=True, temperature=2.0)
        
        # При разных температурах должны быть разные результаты
        assert not torch.equal(low_temp, high_temp), "Разные температуры должны давать разные результаты"

    def test_temperature_zero_error(self, default_config, sample_input):
        """Тест обработки нулевой температуры"""
        gpt = GPT(**default_config)
        # Теперь при temperature=0 не должно быть ошибки
        output = gpt.generate(sample_input, max_new_tokens=5, do_sample=True, temperature=0.0)
        assert output.shape[1] == sample_input.shape[1] + 5  # Проверяем длину вывода

    def test_sample_vs_greedy_difference(self, default_config):
        """Тест различий между жадным и сэмплирующим режимами"""
        torch.manual_seed(42)
        gpt = GPT(**default_config)
        input_tensor = torch.randint(0, 1000, (1, 10))
        
        greedy = gpt.generate(input_tensor, max_new_tokens=5, do_sample=False)
        sampled = gpt.generate(input_tensor, max_new_tokens=5, do_sample=True)
        
        assert not torch.equal(greedy, sampled), "Режимы должны давать разные результаты"

    def test_top_k_sampling(self, default_config):
        """Тест генерации с top-k сэмплированием"""
        torch.manual_seed(42)
        gpt = GPT(**default_config)
        input_tensor = torch.randint(0, 1000, (1, 10))
        
        # Ожидаем ошибку из-за устаревшего типа масок
        with pytest.raises(RuntimeError, match="boolean masks"):
            gpt.generate(input_tensor, max_new_tokens=5, do_sample=True, top_k=50)

    def test_top_p_sampling(self, default_config):
        """Тест генерации с top-p (nucleus) сэмплированием"""
        torch.manual_seed(42)
        gpt = GPT(**default_config)
        input_tensor = torch.randint(0, 1000, (1, 10))
        
        # Ожидаем ошибку из-за устаревшего типа масок
        with pytest.raises(RuntimeError, match="boolean masks"):
            gpt.generate(input_tensor, max_new_tokens=5, do_sample=True, top_p=0.9)

    def test_top_k_top_p_combined(self, default_config):
        """Тест совместного использования top_k и top_p"""
        torch.manual_seed(42)
        gpt = GPT(**default_config)
        input_tensor = torch.randint(0, 1000, (1, 10))
        
        # Ожидаем ошибку из-за устаревшего типа масок
        with pytest.raises(RuntimeError, match="boolean masks"):
            gpt.generate(input_tensor, max_new_tokens=5, do_sample=True, top_k=50, top_p=0.9)

    def test_generate_deterministic(self, default_config):
        """Тест детерминированности генерации (при одинаковом seed)"""
        # Фиксируем seed для воспроизводимости
        torch.manual_seed(42)
        gpt = GPT(**default_config)
        gpt.eval()  # Отключаем dropout для детерминированности
        input_tensor = torch.randint(0, 1000, (1, 10))
        
        # Жадный режим должен быть детерминированным
        out1 = gpt.generate(input_tensor.clone(), max_new_tokens=5, do_sample=False)
        out2 = gpt.generate(input_tensor.clone(), max_new_tokens=5, do_sample=False)
        assert torch.equal(out1, out2), "Жадная генерация должна быть детерминированной"
        
        # Сэмплирующий режим с фиксированным seed
        torch.manual_seed(42)
        out3 = gpt.generate(input_tensor.clone(), max_new_tokens=5, do_sample=True)
        torch.manual_seed(42)
        out4 = gpt.generate(input_tensor.clone(), max_new_tokens=5, do_sample=True)
        assert torch.equal(out3, out4), "Сэмплирование должно быть детерминированным при одинаковом seed"

if __name__ == "__main__":
    pytest.main(["-v"])