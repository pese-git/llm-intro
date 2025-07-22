import torch
from torch.utils.data import Dataset

class GetData(Dataset):
    """
    Класс для создания датасета последовательных данных для обучения языковых моделей.
    
    Наследуется от torch.utils.data.Dataset и реализует:
    - Скользящее окно по последовательности данных
    - Автоматическое разделение на входные и целевые последовательности
    - Поддержку работы на CPU/GPU
    - Проверку корректности параметров
    
    Args:
        data (List): Обучающая последовательность (список чисел или токенов)
        seq_len (int): Длина одной обучающей последовательности (в элементах). 
                      Должна быть положительной и меньше длины данных.
        device (str, optional): Устройство для тензоров ('cpu' или 'cuda'). По умолчанию 'cpu'.
    
    Raises:
        ValueError: Если seq_len <= 0 или seq_len >= len(data)
    
    Attributes:
        _data (List): Хранит входную последовательность
        _seq_len (int): Длина последовательности для обучения
        _device (str): Устройство для вычислений
    
    Examples:
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> dataset = GetData(data, seq_len=3)
        >>> len(dataset)
        6
        >>> dataset[0]
        (tensor([1, 2, 3]), tensor([2, 3, 4]))
        
        # Некорректные параметры
        >>> GetData(data=[1, 2, 3], seq_len=4)  # Вызовет ValueError
        >>> GetData(data=[1, 2, 3], seq_len=-1)  # Вызовет ValueError
    """
    
    def __init__(self, data: list, seq_len: int, device: str = "cpu") -> None:
        """Инициализация датасета с последовательными данными."""
        if seq_len <= 0:
            raise ValueError(f"Sequence length must be positive, got {seq_len}")
        if seq_len >= len(data):
            raise ValueError(f"Sequence length {seq_len} must be less than data length {len(data)}")
        self._data = data
        self._seq_len = seq_len
        self._device = device

    def __len__(self) -> int:
        """
        Возвращает количество обучающих примеров в датасете.
        
        Формула:
            N - seq_len - 1
        где N - длина всей последовательности
        
        Returns:
            int: Количество доступных последовательностей
        """
        return len(self._data) - self._seq_len - 1
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает один обучающий пример по индексу.
        
        Args:
            idx (int): Позиция начала последовательности
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Пара (входная_последовательность, целевая_последовательность)
            где целевая последовательность сдвинута на 1 элемент вперед
        """
        x = torch.tensor(self._data[idx:idx+self._seq_len]).to(self._device)
        y = torch.tensor(self._data[idx+1:idx+self._seq_len+1]).to(self._device)
        return (x, y)