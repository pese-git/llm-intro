import pytest
from bpe import BPE

class TestBPEEncode:
    @pytest.fixture
    def trained_bpe(self):
        """Фикстура с обученным BPE на тестовом словаре"""
        bpe = BPE(vocab_size=10)
        training_text = "абра кадабра абра швабра"
        bpe.fit(training_text)
        print(bpe.vocab)
        return bpe

    def test_encode_simple_tokens(self, trained_bpe):
        # Проверка базовой токенизации
        text = "абра кадабра"
        expected_ids = [
            trained_bpe.token2id["абр"],
            trained_bpe.token2id["а"],
            trained_bpe.token2id[" "],
            trained_bpe.token2id["к"],
            trained_bpe.token2id["а"],
            trained_bpe.token2id["д"],
            trained_bpe.token2id["абр"],
            trained_bpe.token2id["а"]
        ]
        assert trained_bpe.encode(text) == expected_ids

    def test_encode_with_unknown_chars(self, trained_bpe):
        # Проверка обработки неизвестных символов
        text = "абра xyz кадабра"
        expected_ids = [
             trained_bpe.token2id["абр"],
            trained_bpe.token2id["а"],
            trained_bpe.token2id[" "],
            -1,  # x
            -1,  # y
            -1,  # z
            trained_bpe.token2id[" "],
            trained_bpe.token2id["к"],
            trained_bpe.token2id["а"],
            trained_bpe.token2id["д"],
            trained_bpe.token2id["абр"],
            trained_bpe.token2id["а"]
        ]
        assert trained_bpe.encode(text) == expected_ids

    def test_empty_string(self, trained_bpe):
        # Проверка пустой строки
        assert trained_bpe.encode("") == []

    def test_max_matching(self, trained_bpe):
        # Проверка жадного поиска максимального совпадения
        # Добавим специальный токен в словарь
        trained_bpe.vocab.append("абра")
        trained_bpe.token2id["абра"] = len(trained_bpe.vocab) - 1
        trained_bpe.max_token_len = max(len(t) for t in trained_bpe.vocab)
        
        text = "абракадабра"
        # Должен найти "абра" вместо "а"+"б"+"ра"
        assert trained_bpe.encode(text)[:1] == [trained_bpe.token2id["абра"]]

    def test_special_characters(self, trained_bpe):
        # Проверка специальных символов
        text = "\n\t"
        result = trained_bpe.encode(text)
        assert len(result) == 2
        assert all(id_ == -1 for id_ in result)  # Неизвестные символы