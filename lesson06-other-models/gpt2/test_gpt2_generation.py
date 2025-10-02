import torch
import torch.nn.functional as F
from gpt2 import GPT2

def test_gpt2_generation():
    """
    Тестирование генерации GPT-2 с заданными параметрами
    """
    # Параметры из тестового примера
    params = {
        "batch_size": 1,
        "seq_len": 12, 
        "emb_size": 12,
        "num_heads": 5,
        "head_size": 8,
        "vocab_size": 15,
        "max_seq_len": 40,
        "num_layers": 5,
        "max_new_tokens": 5,
        "do_sample": False
    }
    
    print("Параметры модели:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Создаем модель
    model = GPT2(
        vocab_size=params["vocab_size"],
        max_seq_len=params["max_seq_len"],
        emb_size=params["emb_size"],
        num_heads=params["num_heads"],
        head_size=params["head_size"],
        num_layers=params["num_layers"],
        device='cpu'
    )
    
    print(f"\nСоздана модель GPT-2:")
    print(f"  Общее количество параметров: {sum(p.numel() for p in model.parameters())}")
    
    # Создаем тестовый вход
    torch.manual_seed(42)  # Для воспроизводимости
    input_tokens = torch.randint(0, params["vocab_size"], (params["batch_size"], params["seq_len"]))
    
    print(f"\nВходные данные:")
    print(f"  Форма: {input_tokens.shape}")
    print(f"  Токены: {input_tokens.tolist()}")
    
    # Тестируем прямой проход
    with torch.no_grad():
        logits = model(input_tokens)
        print(f"\nПрямой проход:")
        print(f"  Выходная форма: {logits.shape}")
        print(f"  Диапазон логитов: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Тестируем генерацию
    print(f"\nГенерация (do_sample={params['do_sample']}):")
    generated = model.generate(
        x=input_tokens,
        max_new_tokens=params["max_new_tokens"],
        do_sample=params["do_sample"]
    )
    
    print(f"  Выходная форма: {generated.shape}")
    print(f"  Исходная последовательность: {input_tokens.tolist()[0]}")
    print(f"  Сгенерированная последовательность: {generated.tolist()[0]}")
    
    # Проверяем соответствие ожидаемому формату вывода
    expected_shape = torch.Size([params["batch_size"], params["seq_len"] + params["max_new_tokens"]])
    assert generated.shape == expected_shape, f"Ожидалась форма {expected_shape}, получена {generated.shape}"
    
    # Вычисляем сумму всех токенов (для сравнения с ожидаемым значением 122)
    total_sum = generated.sum().item()
    print(f"  Сумма всех токенов: {total_sum}")
    
    print(f"\nФинальный результат:")
    print(f"  {generated.shape} | {total_sum}")
    
    return generated, total_sum

if __name__ == "__main__":
    test_gpt2_generation()