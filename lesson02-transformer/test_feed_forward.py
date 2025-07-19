import torch
import pytest
from feed_forward import FeedForward

class TestFeedForward:
    @pytest.fixture
    def ff_layer(self):
        return FeedForward(emb_size=512)

    def test_initialization(self, ff_layer):
        assert isinstance(ff_layer._layer1, torch.nn.Linear)
        assert isinstance(ff_layer._layer2, torch.nn.Linear)
        assert isinstance(ff_layer._relu, torch.nn.ReLU)
        assert isinstance(ff_layer._dropout, torch.nn.Dropout)
        
        assert ff_layer._layer1.in_features == 512
        assert ff_layer._layer1.out_features == 2048
        assert ff_layer._layer2.in_features == 2048
        assert ff_layer._layer2.out_features == 512

    def test_forward_pass_shape(self, ff_layer):
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 512)
        output = ff_layer(x)
        
        assert output.shape == (batch_size, seq_len, 512)

    def test_dropout_training(self):
        ff_layer = FeedForward(512, dropout=0.5)
        ff_layer.train()
        x = torch.randn(2, 5, 512)
        output = ff_layer(x)
        
        # Проверяем, что dropout действительно работает в режиме обучения
        assert not torch.allclose(output, ff_layer._layer2(ff_layer._relu(ff_layer._layer1(x))))

    def test_dropout_eval(self):
        ff_layer = FeedForward(512, dropout=0.5)
        ff_layer.eval()
        x = torch.randn(2, 5, 512)
        output = ff_layer(x)
        
        # В eval режиме dropout не должен работать
        expected = ff_layer._layer2(ff_layer._relu(ff_layer._layer1(x)))
        assert torch.allclose(output, expected)

    def test_dtype_preservation(self, ff_layer):
        x = torch.randn(2, 5, 512, dtype=torch.float64)
        output = ff_layer(x)
        assert output.dtype == torch.float64
