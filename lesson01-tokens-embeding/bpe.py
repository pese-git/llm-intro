class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.id2token = {}
        self.token2id = {}

    def fit(self, text: str):
        # 1. Получаем уникальные токены (символы)
        unique_tokens = sorted(set(text))
        tokens = unique_tokens.copy()

        # 2. Разбиваем текст на токены-символы
        sequence = list(text)

        # 3. Объединяем токены до достижения нужного размера словаря
        while len(tokens) < self.vocab_size:
            #print(f'len={len(tokens)} < {self.vocab_size}')
            # Считаем частоты пар
            pair_freq = {}
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                #print(f'pair = {pair}')
                if pair not in pair_freq:
                    pair_freq[pair] = 0
                pair_freq[pair] += 1


            #print(f'pair_freq = {pair_freq}')  
            if not pair_freq:
                break  # нет пар — выходим

            #for x in pair_freq.items():
            #    self.debug(x, sequence)

            # Находим самую частую пару (в случае равенства — та, что встретилась первой)
            most_frequent_pair = max(pair_freq.items(), key=lambda x: (x[1], -self._pair_first_index(sequence, x[0])))[0]
            #print(most_frequent_pair)
            # Создаем новый токен
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            #print(f"new token={new_token}")
            tokens.append(new_token)
            #print(f"tokens={tokens}")

            i = 0
            new_sequence = []

            while i < len(sequence):
                if i < len(sequence) - 1 and (sequence[i], sequence[i + 1]) == most_frequent_pair:
                    new_sequence.append(new_token)
                    i += 2  # пропускаем два символа — заменённую пару
                else:
                    new_sequence.append(sequence[i])
                    i += 1
            sequence = new_sequence
            #break
        
        # 4. Создаем словари
        self.vocab = tokens.copy()
        self.token2id = dict(zip(tokens, range(self.vocab_size)))
        self.id2token = dict(zip(range(self.vocab_size), tokens))

    def _pair_first_index(self, sequence, pair):
        for i in range(len(sequence) - 1):
            if (sequence[i], sequence[i + 1]) == pair:
                return i
        return float('inf')  # если пара не найдена (в теории не должно случиться)


    def encode(self, text: str):
        # 1. Разбиваем текст на токены-символы
        sequence = list(text)
        # 2. Инициализация пустого списка токенов
        tokens = []
        # 3. Установить i = 0
        i = 0
        while i < len(text):
            # 3.1 Найти все токены в словаре, начинающиеся с text[i]
            start_char = text[i]
            result = [token for token in self.vocab if token.startswith(start_char)]
            # 3.2 Выбрать самый длинный подходящий токен
            find_token = self._find_max_matching_token(text[i:], result)
            if find_token is None:
                # Обработка неизвестного символа
                tokens.append(text[i])  # Добавляем сам символ как токен
                i += 1
            else:
                # 3.3 Добавить токен в результат
                tokens.append(find_token)
                # 3.4 Увеличить i на длину токена
                i += len(find_token)

        # 4. Заменить токены на их ID
        return self._tokens_to_ids(tokens)

    def _find_max_matching_token(self, text: str, tokens: list):
        """Находит самый длинный токен из списка, с которого начинается текст"""
        matching = [token for token in tokens if text.startswith(token)]
        return max(matching, key=len) if matching else None

    def _tokens_to_ids(self, tokens):
        """Конвертирует список токенов в их ID с обработкой неизвестных токенов"""
        ids = []
        for token in tokens:
            if token in self.token2id:
                ids.append(self.token2id[token])
            else:
                ids.append(-1)  # Специальное значение
        return ids


    def decode(self, ids: list) -> str:
        return ''.join(self._ids_to_tokens(ids))

    def _ids_to_tokens(self, ids: list) -> list:
        """Конвертирует список Ids в их tokens"""
        tokens = []
        for id in ids:
            if id in self.id2token:
                tokens.append(self.id2token[id])
            else:
                tokens.append('')  # Специальное значение
        return tokens