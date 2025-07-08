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

    #def debug(self, x, sequence):
    #    token_pair = x[0]
    #    frequency = x[1]
    #    first_token = token_pair[0]
    #    first_index = sequence.index(first_token)
    #    key_value = (frequency, -first_index)
    #    print(f"x = {x}")
    #    print(f"  token_pair = {token_pair}")
    #    print(f"  frequency = {frequency}")
    #    print(f"  first_token = {first_token}")
    #    print(f"  first_index = {first_index}")
    #    print(f"  key = {key_value}")
    #    print()
