import ast
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from bpe import BPE

def parse_examples(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    examples = []
    for section in content.split('Пример №'):
        if not section.strip():
            continue
        lines = section.strip().split('\n')
        vocab_line = [line for line in lines if line.startswith('vocab_size')][0]
        text_line = [line for line in lines if line.startswith('текст')][0]
        tokens_line = [line for line in lines if line.startswith('токены')][0]
        vocab_size = int(vocab_line.split(':')[1].strip())
        text = text_line.split(':', 1)[1].strip().strip('\'"')
        tokens = ast.literal_eval(tokens_line.split(':', 1)[1].strip())
        examples.append((text, vocab_size, tokens))
    return examples

def test_bpe_against_examples():
    examples = parse_examples(os.path.join(os.path.dirname(__file__), "BPE.txt"))
    for text, vocab_size, expected_tokens in examples:
        bpe = BPE(vocab_size)
        bpe.fit(text)
        # Сравнение по полю vocab - предполагается что результат BPE будет в bpe.vocab
        assert getattr(bpe, "vocab", None) == expected_tokens, (
            f"Failed for vocab_size={vocab_size}, text={text}. "
            f"Expected tokens: {expected_tokens}, got: {getattr(bpe, 'vocab', None)}"
        )


def test_save_load_dill(tmp_path):
    """Тестирование сохранения и загрузки с использованием dill"""
    # 1. Подготовка тестовой модели
    original = BPE(vocab_size=30)
    original.fit("абра кадабра абра швабра")
    
    # 2. Сохранение модели
    test_file = tmp_path / "bpe_model.dill"
    original.save(test_file)
    
    # 3. Проверка файла
    assert test_file.exists()
    assert test_file.stat().st_size > 0
    
    # 4. Загрузка модели
    loaded = BPE.load(test_file)
    
    # 5. Проверка целостности
    assert loaded.vocab_size == original.vocab_size
    assert loaded.vocab == original.vocab
    assert loaded.token2id == original.token2id
    
    # 6. Проверка работоспособности
    assert loaded.encode("кадабра") == original.encode("кадабра")