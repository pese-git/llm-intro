import pytest
from bpe import BPE

class TestBPEDecode:
    @pytest.fixture
    def trained_bpe(self):
        """Фикстура с обученным BPE на тестовом словаре"""
        bpe = BPE(vocab_size=10)
        training_text = "абра кадабра абра швабра"
        bpe.fit(training_text)
        print(bpe.vocab)
        return bpe

    def test_decode_basic(self, trained_bpe):
        """Тест базового декодирования"""
        # Подготовка тестовых ID
        test_ids = [
            trained_bpe.token2id["абр"],
            trained_bpe.token2id["а"],
            trained_bpe.token2id[" "],
            trained_bpe.token2id["к"],
            trained_bpe.token2id["а"],
            trained_bpe.token2id["д"],
            trained_bpe.token2id["абр"],
            trained_bpe.token2id["а"]
        ]
        
        # Проверка декодирования
        result = trained_bpe.decode(test_ids)
        assert result == "абра кадабра"

    def test_decode_with_unknown_ids(self, trained_bpe):
        """Тест с неизвестными ID токенов"""
        test_ids = [
            trained_bpe.token2id["а"],
            -1,  # Неизвестный ID
            trained_bpe.token2id[" "],
            9999  # Несуществующий ID
        ]
        
        result = trained_bpe.decode(test_ids)
        assert result.startswith("а")  # Первый известный символ
        assert len(result) == 2  # 2 символа (неизвестные коды скипаем)

    def test_decode_empty_list(self, trained_bpe):
        """Тест с пустым списком ID"""
        assert trained_bpe.decode([]) == ""

    def test_decode_special_chars(self, trained_bpe):
        """Тест со специальными символами"""
        # Добавим специальные символы в словарь
        special_token = "\n"
        trained_bpe.vocab.append(special_token)
        trained_bpe.token2id[special_token] = len(trained_bpe.vocab) - 1
        trained_bpe.id2token[len(trained_bpe.vocab) - 1] = special_token
        
        test_ids = [
            trained_bpe.token2id["а"],
            trained_bpe.token2id[special_token],
            trained_bpe.token2id[" "]
        ]
        
        assert trained_bpe.decode(test_ids) == "а\n "


    def test_roundtrip(self, trained_bpe):
        """Тест кодирования-декодирования (roundtrip)"""
        original_text = "абра кадабра"
        encoded = trained_bpe.encode(original_text)
        decoded = trained_bpe.decode(encoded)
        assert decoded == original_text