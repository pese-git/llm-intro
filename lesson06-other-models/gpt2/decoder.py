from torch import nn
import torch
from feed_forward import FeedForward
from multi_head_attention import MultiHeadAttention

class Decoder(nn.Module):
    """
    Декодер трансформера в стиле GPT-2 - ключевой компонент архитектуры Transformer.
    
    Предназначен для:
    - Обработки последовательностей с учетом контекста (самовнимание)
    - Постепенного генерирования выходной последовательности
    - Учета causal масок для предотвращения "заглядывания в будущее"

    Алгоритм работы (Pre-norm архитектура как в GPT-2):
    1. Входной тензор (batch_size, seq_len, emb_size)
    2. LayerNorm -> Multi-Head Attention -> Residual Connection
    3. LayerNorm -> FeedForward Network -> Residual Connection
    4. Выходной тензор (batch_size, seq_len, emb_size)

    Основные характеристики:
    - Поддержка causal масок внимания
    - Residual connections для стабилизации градиентов
    - Layer Normalization ДО каждого sub-layer (Pre-norm как в GPT-2)
    - Конфигурируемые параметры внимания

    Примеры использования:

    1. Базовый случай:
    >>> decoder = Decoder(num_heads=8, emb_size=512, head_size=64, max_seq_len=1024)
    >>> x = torch.randn(1, 10, 512)  # [batch, seq_len, emb_size]
    >>> output = decoder(x)
    >>> print(output.shape)
    torch.Size([1, 10, 512])

    2. С маской внимания:
    >>> mask = torch.tril(torch.ones(10, 10))  # Нижнетреугольная маска
    >>> output = decoder(x, mask)

    3. Инкрементальное декодирование:
    >>> for i in range(10):
    >>>     output = decoder(x[:, :i+1, :], mask[:i+1, :i+1])
    """
    def __init__(self, 
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        dropout: float = 0.1
    ):
        """
        Инициализация декодера.

        Параметры:
            num_heads: int - количество голов внимания
            emb_size: int - размерность эмбеддингов
            head_size: int - размерность каждой головы внимания
            max_seq_len: int - максимальная длина последовательности
            dropout: float (default=0.1) - вероятность dropout
        """
        super().__init__()
        self._heads = MultiHeadAttention(
            num_heads=num_heads, 
            emb_size=emb_size, 
            head_size=head_size, 
            max_seq_len=max_seq_len, 
            dropout=dropout
        )
        self._ff = FeedForward(emb_size=emb_size, dropout=dropout)
        self._norm1 = nn.LayerNorm(emb_size)
        self._norm2 = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Прямой проход через декодер в стиле GPT-2 (Pre-norm архитектура).

        Вход:
            x: torch.Tensor - входной тензор [batch_size, seq_len, emb_size]
            mask: torch.Tensor (optional) - маска внимания [seq_len, seq_len]

        Возвращает:
            torch.Tensor - выходной тензор [batch_size, seq_len, emb_size]

        Алгоритм forward (Pre-norm как в GPT-2):
        1. Применяем LayerNorm -> MultiHeadAttention -> Residual Connection
        2. Применяем LayerNorm -> FeedForward Network -> Residual Connection
        """
        # Self-Attention блок
        norm1_out = self._norm1(x)
        attention = self._heads(norm1_out, mask)
        out = attention + x
        
        # FeedForward блок
        norm2_out = self._norm2(out)
        ffn_out = self._ff(norm2_out)
        return ffn_out + out