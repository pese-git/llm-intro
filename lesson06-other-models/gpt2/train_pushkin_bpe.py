import glob
import json
import time
from pathlib import Path
from bpe import BPE  # Импортируем наш класс BPE из lesson04-training

def load_corpus():
    """Загрузка и объединение всех текстов Пушкина"""
    texts_dir = Path('./text/pushkin_poetry')
    all_text = []
    
    print("Загрузка текстов...")
    for file_path in texts_dir.glob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:  # Пропускаем пустые файлы
                all_text.append(text)
    
    corpus = '\n\n\n'.join(all_text)
    print(f"Загружено {len(all_text)} файлов, всего символов: {len(corpus):,}")
    return corpus

def train_tokenizer(corpus, vocab_size=2000):
    """Обучение BPE токенизатора"""
    print("\nОбучение BPE токенизатора...")
    start_time = time.time()
    
    tokenizer = BPE(vocab_size=vocab_size)
    tokenizer.fit(
        text=corpus
    )
    
    training_time = time.time() - start_time
    print(f"\nОбучение завершено за {training_time:.2f} секунд")
    return tokenizer

def save_artifacts(tokenizer, tokens):
    """Сохранение токенизатора и токенов"""
    print("\nСохранение артефактов...")
    
    # Сохраняем токенизатор
    tokenizer.save("pushkin_bpe_tokenizer.json")
    
    # Сохраняем токены в бинарном формате
    import numpy as np
    np.save("pushkin_tokens.npy", np.array(tokens, dtype=np.int32))
    
    print("Артефакты сохранены:")
    print("- pushkin_bpe_tokenizer.json (токенизатор)")
    print("- pushkin_tokens.npy (токенизированный корпус)")

def main():
    # 1. Загрузка корпуса
    corpus = load_corpus()
    
    # 2. Обучение токенизатора (можно уменьшить vocab_size для ускорения)
    tokenizer = train_tokenizer(corpus, vocab_size=2000)
    
    # 3. Токенизация всего корпуса
    print("\nТокенизация корпуса...")
    tokens = tokenizer.encode(corpus)
    print(f"Получено {len(tokens):,} токенов")
    
    # 4. Сохранение результатов
    save_artifacts(tokenizer, tokens)

if __name__ == "__main__":
    main()