import torch
import pytest
import math
from gelu import GELU

class TestGELU:
    """Тесты для функции активации GELU"""
    
    @pytest.fixture
    def gelu(self):
        """Фикстура для создания экземпляра GELU"""
        return GELU()
    
    @pytest.fixture
    def sample_input_3d(self):
        """Фикстура для создания 3D тензора (batch_size × seq_len × emb_size)"""
        return torch.randn(2, 10, 64)  # batch_size=2, seq_len=10, emb_size=64
    
    @pytest.fixture
    def sample_input_2d(self):
        """Фикстура для создания 2D тензора"""
        return torch.randn(4, 32)  # batch_size=4, seq_len=32
    
    @pytest.fixture
    def sample_input_1d(self):
        """Фикстура для создания 1D тензора"""
        return torch.randn(16)  # вектор из 16 элементов

    def test_gelu_initialization(self, gelu):
        """Тест инициализации класса GELU"""
        assert isinstance(gelu, torch.nn.Module)
        assert hasattr(gelu, 'sqrt_2_over_pi')
        assert isinstance(gelu.sqrt_2_over_pi, torch.Tensor)
        
        # Проверка вычисления константы
        expected_value = math.sqrt(2 / math.pi)
        assert torch.allclose(gelu.sqrt_2_over_pi, torch.tensor(expected_value), rtol=1e-6)

    def test_gelu_forward_3d(self, gelu, sample_input_3d):
        """Тест прямого прохода для 3D тензора"""
        output = gelu(sample_input_3d)
        
        # Проверка размерности
        assert output.shape == sample_input_3d.shape
        
        # Проверка типа данных
        assert output.dtype == torch.float32
        
        # Проверка, что выходные значения в разумных пределах
        assert torch.all(output >= -0.5)  # GELU(x) >= -0.17 для всех x, но с учетом аппроксимации
        assert torch.all(output <= torch.max(sample_input_3d) * 1.1)  # GELU(x) <= x для x > 0

    def test_gelu_forward_2d(self, gelu, sample_input_2d):
        """Тест прямого прохода для 2D тензора"""
        output = gelu(sample_input_2d)
        
        # Проверка размерности
        assert output.shape == sample_input_2d.shape
        
        # Проверка типа данных
        assert output.dtype == torch.float32

    def test_gelu_forward_1d(self, gelu, sample_input_1d):
        """Тест прямого прохода для 1D тензора"""
        output = gelu(sample_input_1d)
        
        # Проверка размерности
        assert output.shape == sample_input_1d.shape
        
        # Проверка типа данных
        assert output.dtype == torch.float32

    def test_gelu_specific_values(self, gelu):
        """Тест конкретных значений GELU"""
        # Тестовые точки
        test_cases = [
            (0.0, 0.0),           # GELU(0) = 0
            (1.0, 0.8413),        # GELU(1) ≈ 0.8413
            (-1.0, -0.1587),      # GELU(-1) ≈ -0.1587
            (2.0, 1.9540),        # GELU(2) ≈ 1.9540
            (-2.0, -0.0454),      # GELU(-2) ≈ -0.0454
        ]
        
        for input_val, expected_val in test_cases:
            input_tensor = torch.tensor([input_val], dtype=torch.float32)
            output = gelu(input_tensor)
            expected_tensor = torch.tensor([expected_val], dtype=torch.float32)
            
            # Допустимая погрешность для аппроксимации
            assert torch.allclose(output, expected_tensor, rtol=1e-3, atol=1e-3)

    def test_gelu_positive_values(self, gelu):
        """Тест, что положительные значения почти сохраняются"""
        positive_input = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float32)
        output = gelu(positive_input)
        
        # Для положительных x: 0 < GELU(x) < x
        assert torch.all(output > 0)
        assert torch.all(output < positive_input)

    def test_gelu_negative_values(self, gelu):
        """Тест, что отрицательные значения подавляются"""
        negative_input = torch.tensor([-0.5, -1.0, -2.0, -5.0], dtype=torch.float32)
        output = gelu(negative_input)
        
        # Для отрицательных x: x < GELU(x) < 0
        assert torch.all(output < 0)
        assert torch.all(output > negative_input)

    def test_gelu_zero_values(self, gelu):
        """Тест нулевых значений"""
        zero_input = torch.zeros(5, dtype=torch.float32)
        output = gelu(zero_input)
        
        # GELU(0) = 0
        assert torch.all(output == 0)

    def test_gelu_differentiability(self, gelu, sample_input_3d):
        """Тест дифференцируемости функции"""
        sample_input_3d.requires_grad_(True)
        output = gelu(sample_input_3d)
        
        # Проверка, что градиенты могут быть вычислены
        loss = output.sum()
        loss.backward()
        
        assert sample_input_3d.grad is not None
        assert sample_input_3d.grad.shape == sample_input_3d.shape

    def test_gelu_gradient_at_zero(self, gelu):
        """Тест градиента в нуле"""
        x = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
        y = gelu(x)
        y.backward()
        
        # Теоретический градиент GELU в нуле: 0.5
        expected_grad = 0.5
        assert torch.allclose(x.grad, torch.tensor([expected_grad]), rtol=1e-3)

    def test_gelu_key_properties(self, gelu):
        """Тест ключевых свойств функции GELU"""
        # Проверяем основные свойства функции
        
        # 1. GELU(0) = 0
        assert torch.allclose(gelu(torch.tensor([0.0])), torch.tensor([0.0]))
        
        # 2. Для положительных x: 0 < GELU(x) < x
        positive_x = torch.tensor([0.5, 1.0, 2.0, 5.0])
        positive_y = gelu(positive_x)
        assert torch.all(positive_y > 0)
        assert torch.all(positive_y < positive_x)
        
        # 3. Для отрицательных x: x < GELU(x) < 0
        negative_x = torch.tensor([-0.5, -1.0, -2.0, -5.0])
        negative_y = gelu(negative_x)
        assert torch.all(negative_y < 0)
        assert torch.all(negative_y > negative_x)
        
        # 4. При больших положительных x: GELU(x) ≈ x
        large_positive = torch.tensor([10.0, 20.0, 50.0])
        large_positive_y = gelu(large_positive)
        assert torch.allclose(large_positive_y, large_positive, rtol=0.01)
        
        # 5. При больших отрицательных x: GELU(x) ≈ 0
        large_negative = torch.tensor([-10.0, -20.0, -50.0])
        large_negative_y = gelu(large_negative)
        assert torch.all(torch.abs(large_negative_y) < 0.01)

    def test_gelu_batch_independence(self, gelu):
        """Тест независимости обработки батчей"""
        batch1 = torch.randn(2, 5, 8)
        batch2 = torch.randn(2, 5, 8)
        
        # Объединяем в один батч
        combined_batch = torch.cat([batch1, batch2], dim=0)
        combined_output = gelu(combined_batch)
        
        # Обрабатываем отдельно
        output1 = gelu(batch1)
        output2 = gelu(batch2)
        separate_output = torch.cat([output1, output2], dim=0)
        
        # Результаты должны совпадать
        assert torch.allclose(combined_output, separate_output)

    def test_gelu_device_compatibility(self, gelu):
        """Тест совместимости с разными устройствами (если доступен CUDA)"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gelu_cuda = GELU().to(device)
            input_cpu = torch.randn(2, 10, 64)
            input_cuda = input_cpu.to(device)
            
            output_cpu = gelu(input_cpu)
            output_cuda = gelu_cuda(input_cuda)
            
            # Результаты должны совпадать с учетом погрешности
            assert torch.allclose(output_cpu, output_cuda.cpu(), rtol=1e-6)
