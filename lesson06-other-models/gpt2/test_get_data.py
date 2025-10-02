import pytest
import torch
import sys
import os

# Добавляем путь к модулям в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from get_data import GetData

class TestGetData:
    """Набор тестов для проверки класса GetData"""
    
    @pytest.fixture
    def sample_data(self):
        """Фикстура с тестовыми данными: последовательность чисел 0-99"""
        return list(range(100))

    def test_initialization(self, sample_data):
        """Тест корректности инициализации класса"""
        seq_len = 10
        dataset = GetData(data=sample_data, seq_len=seq_len)
        
        assert dataset._data == sample_data
        assert dataset._seq_len == seq_len
        assert dataset._device == "cpu"
        
        # Проверка инициализации с явным указанием устройства
        dataset_gpu = GetData(data=sample_data, seq_len=seq_len, device="cuda")
        assert dataset_gpu._device == "cuda"

    def test_dataset_length(self, sample_data):
        """Тест корректного вычисления длины датасета"""
        test_cases = [
            (10, 89),   # seq_len=10 → len=100-10-1=89
            (50, 49),    # seq_len=50 → len=100-50-1=49
            (99, 0)      # seq_len=99 → len=100-99-1=0
        ]
        
        for seq_len, expected_len in test_cases:
            dataset = GetData(data=sample_data, seq_len=seq_len)
            assert len(dataset) == expected_len

    def test_item_retrieval(self, sample_data):
        """Тест получения элементов датасета"""
        seq_len = 5
        dataset = GetData(data=sample_data, seq_len=seq_len)
        
        # Проверка первых элементов
        x, y = dataset[0]
        assert torch.equal(x, torch.tensor([0, 1, 2, 3, 4]))
        assert torch.equal(y, torch.tensor([1, 2, 3, 4, 5]))
        
        # Проверка элементов из середины
        x, y = dataset[50]
        assert torch.equal(x, torch.tensor([50, 51, 52, 53, 54]))
        assert torch.equal(y, torch.tensor([51, 52, 53, 54, 55]))
        
        # Проверка последнего элемента
        last_idx = len(dataset) - 1
        x, y = dataset[last_idx]
        expected_x = sample_data[last_idx:last_idx+seq_len]
        expected_y = sample_data[last_idx+1:last_idx+seq_len+1]
        assert torch.equal(x, torch.tensor(expected_x))
        assert torch.equal(y, torch.tensor(expected_y))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Требуется GPU")
    def test_gpu_support(self, sample_data):
        """Тест работы с GPU (только если доступен CUDA)"""
        seq_len = 10
        dataset = GetData(data=sample_data, seq_len=seq_len, device="cuda")
        x, y = dataset[0]
        
        assert x.is_cuda
        assert y.is_cuda
        assert x.device == torch.device("cuda")
        assert y.device == torch.device("cuda")

    def test_edge_cases(self):
        """Тест обработки граничных случаев"""
        # Слишком длинная последовательность
        with pytest.raises(ValueError):
            GetData(data=[1, 2, 3], seq_len=4)
            
        # Отрицательная длина последовательности
        with pytest.raises(ValueError):
            GetData(data=[1, 2, 3], seq_len=-1)
            
        # Пустые входные данные
        with pytest.raises(ValueError):
            GetData(data=[], seq_len=1)

    def test_tensor_conversion(self, sample_data):
        """Тест корректности преобразования в тензоры"""
        seq_len = 3
        dataset = GetData(data=sample_data, seq_len=seq_len)
        x, y = dataset[10]
        
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.dtype == torch.int64
        assert y.dtype == torch.int64

if __name__ == "__main__":
    pytest.main(["-v", "--tb=native"])