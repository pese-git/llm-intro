from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
        self._vocab_size = vocab_size
        self._max_seq_len = max_seq_len
        self._emb_size = emb_size
        self._num_heads = num_heads
        self._head_size = head_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._device = device
        
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
        if x.size(1) > self._max_seq_len:
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

    def generate(self,
        x: torch.Tensor, 
        max_new_tokens: int, 
        do_sample: bool,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ) -> torch.Tensor:
        """Авторегрессивная генерация текста.
        
        Параметры:
            x: Входной тензор с индексами токенов формы [batch_size, seq_len],
               где batch_size - размер батча, seq_len - длина последовательности.
            max_new_tokens: Максимальное количество новых токенов для генерации.
            do_sample: Флаг выбора режима генерации:
                - True: вероятностное сэмплирование
                - False: жадный поиск (argmax)
            temperature: Параметр температуры для сэмплирования:
                - >1.0 - более случайные результаты
                - 1.0 - нейтральное значение
                - <1.0 - более предсказуемые результаты
                Должна быть > 0 (по умолчанию: 1.0)
            top_k: Если задан (и do_sample=True), используется top-k сэмплирование:
                - Выбираются только top_k самых вероятных токенов
                - Остальным токенам устанавливается вероятность 0
                - None: отключено (по умолчанию)
            top_p: Если задан (и do_sample=True), используется nucleus (top-p) сэмплирование:
                - Выбираются токены с кумулятивной вероятностью ≤ top_p
                - Гарантируется, что хотя бы один токен остаётся (даже если его вероятность > top_p)
                - None: отключено (по умолчанию)
                - Должен быть в диапазоне (0, 1]
        
        Возвращает:
            torch.Tensor: Тензор с расширенной последовательностью токенов формы 
                          [batch_size, seq_len + max_new_tokens]

        Исключения:
            ValueError: Если входная последовательность длиннее max_seq_len
            ValueError: Если temperature <= 0
            ValueError: Если одновременно заданы top_k и top_p
            ValueError: Если top_k задан и ≤ 0
            ValueError: Если top_p задан и не в диапазоне (0, 1]

        Примеры:
            >>> # Жадная генерация
            >>> output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
            >>> 
            >>> # Вероятностная генерация с top-k
            >>> output = model.generate(input_ids, max_new_tokens=10, do_sample=True, top_k=50)
            >>>
            >>> # Nucleus sampling (top-p)
            >>> output = model.generate(input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
            >>>
            >>> # Комбинация температуры и top-k
            >>> output = model.generate(input_ids, max_new_tokens=10, do_sample=True, 
            ...                        temperature=0.7, top_k=50)

        Примечания:
            1. Для детерминированных результатов в режиме сэмплирования 
               зафиксируйте random seed (torch.manual_seed).
            2. Температура влияет только на режим сэмплирования (do_sample=True).
            3. Одновременное использование top_k и top_p запрещено.
            4. При do_sample=False параметры top_k, top_p и temperature игнорируются.

        Args:
            x (torch.Tensor): Входной тензор с индексами токенов формы [batch_size, seq_len],
                              где batch_size - размер батча, seq_len - длина последовательности.
            max_new_tokens (int): Максимальное количество новых токенов для генерации.
            do_sample (bool): Флаг выбора режима генерации:
                              - True: вероятностное сэмплирование
                              - False: жадный поиск (argmax)
            temperature (float): Параметр температуры для сэмплирования:
                              - >1.0 - более случайные результаты
                              - 1.0 - нейтральное значение
                              - <1.0 - более предсказуемые результаты
                              Должна быть > 0 (по умолчанию: 1.0)

        Returns:
            torch.Tensor: Тензор с расширенной последовательностью токенов формы 
                          [batch_size, seq_len + max_new_tokens]

        Raises:
            ValueError: Если входная последовательность длиннее max_seq_len
            ValueError: Если temperature <= 0

        Examples:
            >>> # Жадная генерация
            >>> output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
            >>>
            >>> # Вероятностная генерация с температурой
            >>> output = model.generate(input_ids, max_new_tokens=10, do_sample=True, temperature=0.7)
            >>>
            >>> # Более случайная генерация
            >>> output = model.generate(input_ids, max_new_tokens=10, do_sample=True, temperature=1.5)

        Note:
            Для детерминированных результатов в режиме сэмплирования 
            зафиксируйте random seed (torch.manual_seed).
            Температура влияет только на режим сэмплирования (do_sample=True).
        """
        for _ in range(max_new_tokens):
            # 1. Обрезаем вход, если последовательность слишком длинная
            x_cond = x[:, -self.max_seq_len:]

            # 2. Передаем последовательность в метод forward класса GPT и полуаем логиты.
            logits = self.forward(x_cond)

            # 3. Берем логиты для последнего токена
            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # Масштабируем логиты температурой
            if temperature > 0:
                logits_scaled = last_logits / temperature
            else:
                logits_scaled = last_logits

            if do_sample == True and top_k != None:
                _, topk_indices = torch.topk(logits_scaled, top_k, dim=-1)

                # # Заменим все НЕ top-k логиты на -inf
                masked_logits = logits_scaled.clone()
                vocab_size = logits_scaled.size(-1)

                # создаём маску: 1, если токен НЕ в topk_indices
                mask = torch.ones_like(logits_scaled, dtype=torch.uint8)
                mask.scatter_(1, topk_indices, 0)  # 0 там, где top-k индексы
                masked_logits[mask.byte()] = float('-inf')

                logits_scaled = masked_logits

            if do_sample == True and top_p != None:
                # 1. Применим softmax, чтобы получить вероятности:
                probs = F.softmax(logits_scaled, dim=-1)  # [B, vocab_size]
                # 2. Отсортируем токены по убыванию вероятностей:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                # 3. Посчитаем кумулятивную сумму вероятностей:
                cum_probs = torch.cumsum(sorted_probs, dim=-1)  # [B, vocab_size]
                # 4. Определим маску: оставить токены, пока сумма < top_p
                sorted_mask = (cum_probs <= top_p).byte()  # [B, vocab_size]
                # Гарантируем, что хотя бы первый токен останется
                sorted_mask[:, 0] = 1
                # 5. Преобразуем маску обратно в оригинальный порядок:
                # Создаём полную маску из 0
                mask = torch.zeros_like(probs, dtype=torch.uint8)
                # Устанавливаем 1 в местах нужных токенов
                mask.scatter_(dim=1, index=sorted_indices, src=sorted_mask)
                # 6. Зануляем логиты токенов вне топ-p:
                logits_scaled[~mask] = float('-inf')

            # 4. Применяем Softmax
            probs = F.softmax(logits_scaled, dim=-1)  # [batch_size, vocab_size]


            if do_sample == True:
                # 5. Если do_sample равен True, то отбираем токен случайно с помощью torch.multinomial
                next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            else:
                # 5. Если do_sample равен False, то выбираем токен с максимальной вероятностью
                next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [batch_size, 1]
            
            # 6. Добавляем его к последовательности
            x = torch.cat([x, next_token], dim=1)  # [batch_size, seq_len+1]
        return x

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self._vocab_size,
            'max_seq_len': self._max_seq_len,
            'emb_size': self._emb_size,
            'num_heads': self._num_heads,
            'head_size': self._head_size,
            'num_layers': self._num_layers
        }, path)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_seq_len=checkpoint['max_seq_len'],
            emb_size=checkpoint['emb_size'],
            num_heads=checkpoint['num_heads'],
            head_size=checkpoint['head_size'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model

    @property
    def max_seq_len(self) -> int:
        """Возвращает максимальную длину последовательности"""
        return self._max_seq_len


    def fit(self,
        train_loader: DataLoader,
        valid_loader: DataLoader = None,
        num_epoch: int = 1,
        learning_rate: float = 0.001
    ):
        """Обучает модель GPT на предоставленных данных.
        
        Процесс обучения включает:
        - Прямой проход для получения предсказаний
        - Вычисление потерь с помощью кросс-энтропии
        - Обратное распространение ошибки
        - Обновление весов через оптимизатор Adam
        
        Args:
            train_loader (DataLoader): Загрузчик обучающих данных, возвращающий пары (inputs, targets).
                inputs - тензор индексов токенов формы [batch_size, seq_len]
                targets - тензор индексов следующих токенов формы [batch_size, seq_len]
            valid_loader (DataLoader, optional): Загрузчик валидационных данных. Если None, 
                валидация не выполняется. По умолчанию None.
            num_epoch (int, optional): Количество эпох обучения. По умолчанию 1.
            learning_rate (float, optional): Скорость обучения для оптимизатора. По умолчанию 0.001.
        
        Returns:
            None
        
        Raises:
            ValueError: Если train_loader равен None
            ValueError: Если num_epoch ≤ 0
            ValueError: Если learning_rate ≤ 0
        
        Side Effects:
            - Обновляет параметры модели (self.parameters())
            - Устанавливает атрибуты:
                self.train_loss: Средние потери на обучении за последнюю эпоху
                self.validation_loss: Средние потери на валидации за последнюю эпоху (если valid_loader передан)
        
        Examples:
            >>> from torch.utils.data import DataLoader, TensorDataset
            >>> # Создаем тестовые данные
            >>> inputs = torch.randint(0, 100, (100, 10))
            >>> targets = torch.randint(0, 100, (100, 10))
            >>> dataset = TensorDataset(inputs, targets)
            >>> loader = DataLoader(dataset, batch_size=32)
            >>> # Инициализируем модель
            >>> model = GPT(vocab_size=100, max_seq_len=20, emb_size=64, 
            ...             num_heads=4, head_size=16, num_layers=2)
            >>> # Обучаем модель
            >>> model.fit(loader, num_epoch=5, learning_rate=0.001)
        """
        if train_loader is None:
            raise ValueError("train_loader не может быть None")
        if num_epoch <= 0:
            raise ValueError("num_epoch должен быть > 0")
        if learning_rate <= 0:
            raise ValueError("learning_rate должен быть > 0")
    
        device = torch.device(self._device)
        self.to(device)
    
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
        for epoch in range(num_epoch):
            self.train()
            epoch_loss = 0.0
    
            #for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}"):
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
    
                logits = self(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
    
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                epoch_loss += loss.item()
    
            self.train_loss = epoch_loss / len(train_loader)
            #print(f"[{epoch+1}/{num_epoch}] Train Loss: {self.train_loss:.4f}", end='')
    
            if valid_loader is not None:
                self.eval()
                valid_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in valid_loader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
    
                        logits = self(inputs)
                        logits = logits.view(-1, logits.size(-1))
                        targets = targets.view(-1)
    
                        loss = F.cross_entropy(logits, targets)
                        valid_loss += loss.item()
    
                self.validation_loss = valid_loss / len(valid_loader)
                #print(f" | Val Loss: {self.validation_loss:.4f}")