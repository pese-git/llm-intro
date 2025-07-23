import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from gpt import GPT


class DummyDataset(Dataset):
    """Тестовый датасет для проверки обучения"""
    def __init__(self, vocab_size=100, seq_len=10, num_samples=100):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.targets = torch.roll(self.data, -1, dims=1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@pytest.fixture
def gpt_model():
    """Фикстура для создания тестовой модели GPT"""
    return GPT(
        vocab_size=100,
        max_seq_len=10,
        emb_size=64,
        num_heads=4,
        head_size=16,
        num_layers=2
    )


@pytest.fixture
def data_loaders():
    """Фикстура для создания тестовых DataLoader"""
    dataset = DummyDataset()
    train_loader = DataLoader(dataset, batch_size=10)
    valid_loader = DataLoader(dataset, batch_size=10)
    return train_loader, valid_loader


def test_fit_basic(gpt_model, data_loaders):
    """Тестирование базовой работы метода fit"""
    train_loader, valid_loader = data_loaders
    
    # Запускаем обучение
    gpt_model.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epoch=2,
        learning_rate=0.001
    )
    
    # Проверяем что атрибуты loss были установлены
    assert hasattr(gpt_model, 'train_loss'), "train_loss не был сохранен"
    assert hasattr(gpt_model, 'validation_loss'), "validation_loss не был сохранен"
    assert isinstance(gpt_model.train_loss, float), "train_loss должен быть числом"
    assert isinstance(gpt_model.validation_loss, float), "validation_loss должен быть числом"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fit_device(gpt_model, data_loaders, device):
    """Тестирование работы fit на разных устройствах"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA не доступен для тестирования")
    
    train_loader, valid_loader = data_loaders
    
    # Переносим модель на нужное устройство
    gpt_model._device = device
    gpt_model.to(device)
    
    gpt_model.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epoch=1,
        learning_rate=0.001
    )
    
    # Проверяем что модель действительно на нужном устройстве
    assert next(gpt_model.parameters()).device.type == device


def test_fit_loss_decrease(gpt_model, data_loaders):
    """Тестирование что веса изменяются в процессе обучения"""
    train_loader, valid_loader = data_loaders
    
    # Сохраняем начальные веса
    initial_state = {k: v.clone() for k, v in gpt_model.state_dict().items()}
    
    gpt_model.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epoch=3,
        learning_rate=0.01
    )
    
    # Проверяем что веса изменились
    changed = False
    for k in initial_state:
        if not torch.allclose(initial_state[k], gpt_model.state_dict()[k]):
            changed = True
            break
            
    assert changed, "Веса модели не изменились после обучения"


def test_fit_without_validation(gpt_model, data_loaders):
    """Тестирование работы без валидационного набора"""
    train_loader, _ = data_loaders
    
    gpt_model.fit(
        train_loader=train_loader,
        valid_loader=None,
        num_epoch=1,
        learning_rate=0.001
    )
    
    assert hasattr(gpt_model, 'train_loss')
    assert gpt_model.validation_loss is None

def test_fit_with_invalid_train_data(gpt_model):
    """Тестирование обработки невалидных train данных"""
    with pytest.raises(ValueError, match="train_loader не может быть None"):
        gpt_model.fit(
            train_loader=None,
            valid_loader=None,
            num_epoch=1,
            learning_rate=0.001
        )
