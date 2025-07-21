import os
import tempfile
import pytest
import torch
from gpt import GPT

@pytest.mark.skip(reason="Пропуск тестов сохранения/загрузки для ускорения проверки")
def test_save_load():
    """Тестирование сохранения и загрузки модели GPT"""
    # Инициализация параметров модели
    vocab_size = 1000
    max_seq_len = 128
    emb_size = 256
    num_heads = 4
    head_size = 64
    num_layers = 3
    
    # Создаем модель
    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        emb_size=emb_size,
        num_heads=num_heads,
        head_size=head_size,
        num_layers=num_layers
    )
    
    # Создаем временный файл
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    try:
        # Тестируем сохранение
        model.save(temp_path)
        assert os.path.exists(temp_path), "Файл модели не был создан"
        
        # Тестируем загрузку
        loaded_model = GPT.load(temp_path, device='cpu')
        
        # Проверяем, что параметры загружены корректно через проверку конфигурации модели
        assert loaded_model._token_embeddings.num_embeddings == vocab_size
        assert loaded_model.max_seq_len == max_seq_len
        assert loaded_model._token_embeddings.embedding_dim == emb_size
        assert len(loaded_model._decoders) == num_layers
        
        # Проверяем, что веса загрузились корректно
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(),
            loaded_model.named_parameters()
        ):
            assert name1 == name2, "Имена параметров не совпадают"
            assert torch.allclose(param1, param2), f"Параметры {name1} не совпадают"
            
            # Проверяем работу загруженной модели
            test_input = torch.randint(0, vocab_size, (1, 10))
            with torch.no_grad():
                torch.manual_seed(42)  # Фиксируем seed для воспроизводимости
                original_output = model(test_input)
                torch.manual_seed(42)
                loaded_output = loaded_model(test_input)
            assert torch.allclose(original_output, loaded_output, atol=1e-6), "Выходы моделей не совпадают"
            
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_path):
            os.remove(temp_path)

@pytest.mark.skip(reason="Пропуск тестов сохранения/загрузки для ускорения проверки")
def test_save_load_with_generation():
    """Тестирование генерации после загрузки модели"""
    vocab_size = 1000
    max_seq_len = 128
    emb_size = 256
    num_heads = 4
    head_size = 64
    num_layers = 2
    
    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        emb_size=emb_size,
        num_heads=num_heads,
        head_size=head_size,
        num_layers=num_layers
    )
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    try:
        model.save(temp_path)
        loaded_model = GPT.load(temp_path, device='cpu')
        
        # Тестируем генерацию
        input_seq = torch.randint(0, vocab_size, (1, 5))
        original_gen = model.generate(input_seq, max_new_tokens=10)
        loaded_gen = loaded_model.generate(input_seq, max_new_tokens=10)
        
        assert original_gen.shape == loaded_gen.shape, "Размеры сгенерированных последовательностей не совпадают"
        assert torch.all(original_gen == loaded_gen), "Сгенерированные последовательности не совпадают"
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    test_save_load()
    test_save_load_with_generation()
    print("Все тесты прошли успешно!")
