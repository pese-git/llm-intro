from torch import nn
import torch

class FeedForward(nn.Module):
    """
    Слой прямой связи (Feed Forward Network) для архитектуры трансформеров.
    
    Этот слой состоит из двух линейных преобразований с расширением внутренней размерности
    в 4 раза и механизмом dropout для регуляризации. Между линейными слоями применяется
    активация ReLU.

    Алгоритм работы:
    1. Входной тензор x (размерность: [batch_size, seq_len, emb_size])
    2. Линейное преобразование: emb_size -> 4*emb_size
    3. Активация ReLU
    4. Линейное преобразование: 4*emb_size -> emb_size
    5. Применение dropout
    6. Возврат результата (размерность: [batch_size, seq_len, emb_size])

    Предназначение:
    - Добавляет нелинейность в архитектуру трансформера
    - Обеспечивает взаимодействие между различными размерностями эмбеддингов
    - Работает независимо для каждого токена в последовательности

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
        self.net = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.ReLU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        """
        Прямой проход через слой Feed Forward Network.
        
        Args:
            x: Входной тензор размерности [batch_size, seq_len, emb_size]
            
        Returns:
            Тензор той же размерности, что и входной
        """
        # Приводим все параметры сети к типу входного тензора
        self.net = self.net.to(x.dtype)
        return self.net(x)