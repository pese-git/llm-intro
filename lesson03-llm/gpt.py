from torch import nn
import torch
import torch.nn.functional as F
from token_embeddings import TokenEmbeddings
from positional_embeddings import PositionalEmbeddings
from decoder import Decoder

class GPT(nn.Module):
    """GPT-like трансформер для генерации текста
    
    Args:
        vocab_size: Размер словаря
        max_seq_len: Макс. длина последовательности
        emb_size: Размерность эмбеддингов
        num_heads: Количество голов внимания
        head_size: Размерность голов внимания
        num_layers: Количество слоёв декодера
        dropout: Вероятность dropout (default=0.1)
        device: Устройство (default='cpu')
    """
    def __init__(self,
        vocab_size: int,
        max_seq_len: int,
        emb_size: int,
        num_heads: int,
        head_size: int,
        num_layers: int,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.max_seq_len = max_seq_len
        
        # Инициализация слоев
        self._token_embeddings = TokenEmbeddings(
            vocab_size=vocab_size, 
            emb_size=emb_size
        )
        self._position_embeddings = PositionalEmbeddings(
            max_seq_len=max_seq_len, 
            emb_size=emb_size
        )
        self._dropout = nn.Dropout(dropout)
        self._decoders = nn.ModuleList([Decoder(
            num_heads=num_heads,
            emb_size=emb_size,
            head_size=head_size,
            max_seq_len=max_seq_len,
            dropout=dropout 
        ) for _ in range(num_layers)])
        self._linear = nn.Linear(emb_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через GPT
        
        Args:
            x: Входной тензор [batch_size, seq_len]
            
        Returns:
            Тензор логитов [batch_size, seq_len, vocab_size]
        """
        # Проверка длины последовательности
        if x.size(1) > self.max_seq_len:
            raise ValueError(f"Длина последовательности {x.size(1)} превышает максимальную {self.max_seq_len}")
        
        # Эмбеддинги токенов и позиций
        tok_out = self._token_embeddings(x)  # [batch, seq_len, emb_size]
        pos_out = self._position_embeddings(x.size(1))  # [seq_len, emb_size]
        
        # Комбинирование
        out = self._dropout(tok_out + pos_out.unsqueeze(0))  # [batch, seq_len, emb_size]
        
        # Стек декодеров
        for decoder in self._decoders:
            out = decoder(out)
            
        return self._linear(out)  # [batch, seq_len, vocab_size]

    def generate(self, x: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Авторегрессивная генерация текста
        
        Args:
            x: Входной тензор с индексами токенов [batch_size, seq_len]
            max_new_tokens: Максимальное количество новых токенов для генерации
            
        Returns:
            Тензор с расширенной последовательностью токенов [batch_size, seq_len + max_new_tokens]
            
        Алгоритм работы:
        1. На каждом шаге берется последний фрагмент последовательности (не длиннее max_seq_len)
        2. Вычисляются логиты для следующего токена
        3. Выбирается токен с максимальной вероятностью (жадный алгоритм)
        4. Токен добавляется к последовательности
        5. Процесс повторяется пока не сгенерируется max_new_tokens токенов
        """
        for _ in range(max_new_tokens):
            # 1. Обрезаем вход, если последовательность слишком длинная
            x_cond = x[:, -self.max_seq_len:]

            # 2. Передаем последовательность в метод forward класса GPT и полуаем логиты.
            logits = self.forward(x_cond)

            # 3. Берем логиты для последнего токена
            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # 4. Применяем Softmax
            probs = F.softmax(last_logits, dim=-1)  # [batch_size, vocab_size]

            # 5. Выбираем токен с максимальной вероятностью
            next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [batch_size, 1]

            # 6. Добавляем его к последовательности
            x = torch.cat([x, next_token], dim=1)  # [batch_size, seq_len+1]     
        return x