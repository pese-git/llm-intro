from torch import nn
import torch
from gelu import GELU

class FeedForward(nn.Module):
    """
    Слой прямой связи (Feed Forward Network) для архитектуры трансформеров.
    
    Этот слой состоит из двух линейных преобразований с расширением внутренней размерности
    в 4 раза и механизмом dropout для регуляризации. Между линейными слоями применяется
    активация GELU (Gaussian Error Linear Unit), которая является стандартной для GPT-2.

    Алгоритм работы:
    1. Входной тензор x (размерность: [batch_size, seq_len, emb_size])
    2. Линейное преобразование: emb_size -> 4*emb_size
    3. Активация GELU
    4. Линейное преобразование: 4*emb_size -> emb_size
    5. Применение dropout
    6. Возврат результата (размерность: [batch_size, seq_len, emb_size])

    Предназначение:
    - Добавляет нелинейность в архитектуру трансформера
    - Обеспечивает взаимодействие между различными размерностями эмбеддингов
    - Работает независимо для каждого токена в последовательности
    - Использует GELU активацию для лучшей производительности в NLP задачах

    Особенности реализации GPT-2:
    - Используется GELU вместо ReLU для более плавной активации
    - Коэффициент расширения 4x соответствует оригинальной архитектуре
    - Dropout применяется только к выходу второго линейного слоя

    Примеры использования:
    
    >>> # Инициализация слоя
    >>> ff = FeedForward(emb_size=512, dropout=0.1)
    >>>
    >>> # Прямой проход
    >>> x = torch.randn(32, 10, 512)  # [batch_size, seq_len, emb_size]
    >>> output = ff(x)
    >>> print(output.shape)  # torch.Size([32, 10, 512])
    >>>
    >>> # Работа с разными типами данных
    >>> x_double = torch.randn(32, 10, 512, dtype=torch.float64)
    >>> output_double = ff(x_double)
    >>> print(output_double.dtype)  # torch.float64
    """
    def __init__(self, emb_size: int, dropout: float = 0.1):
        """
        Инициализация слоя Feed Forward Network.
        
        Args:
            emb_size: Размерность входных эмбеддингов
            dropout: Вероятность dropout для регуляризации (по умолчанию: 0.1)
        """
        super().__init__()
        # Первый линейный слой (расширение размерности)
        self._layer1 = nn.Linear(emb_size, emb_size * 4)
        # GELU активация (используется в GPT-2 вместо ReLU)
        self._gelu = GELU()
        # Второй линейный слой (сжатие обратно)
        self._layer2 = nn.Linear(emb_size * 4, emb_size)
        # Dropout для регуляризации
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Прямой проход через слой Feed Forward Network.
        
        Args:
            x: Входной тензор размерности [batch_size, seq_len, emb_size]
            
        Returns:
            Тензор той же размерности, что и входной
        """
        # Сохраняем dtype входных данных
        input_dtype = x.dtype
        
        # Приводим веса к нужному типу если необходимо
        if input_dtype != self._layer1.weight.dtype:
            self._layer1 = self._layer1.to(dtype=input_dtype)
            self._layer2 = self._layer2.to(dtype=input_dtype)
            
        # Пропустим тензор x по очереди через все созданные слои
        x = self._layer1(x)
        x = self._gelu(x)
        x = self._layer2(x)
        return self._dropout(x)