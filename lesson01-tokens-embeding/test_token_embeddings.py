import torch
import pytest
from token_embedings import TokenEmbeddings

class TestTokenEmbeddings:
    """Unit tests for TokenEmbeddings class"""
    
    @pytest.fixture
    def embedding_layer(self):
        return TokenEmbeddings(vocab_size=100, emb_size=32)
    
    def test_initialization(self, embedding_layer):
        """Test layer initialization"""
        assert isinstance(embedding_layer, torch.nn.Module)
        assert embedding_layer._embedding.num_embeddings == 100
        assert embedding_layer._embedding.embedding_dim == 32
        
    def test_forward_shape(self, embedding_layer):
        """Test output shape of forward pass"""
        test_input = torch.tensor([
            [1, 2, 3],
            [4, 5, 6]
        ])
        output = embedding_layer(test_input)
        assert output.shape == (2, 3, 32)  # batch_size=2, seq_len=3, emb_size=32
        
    def test_embedding_values(self, embedding_layer):
        """Test that embeddings are trainable"""
        input_tensor = torch.tensor([[1]])
        before = embedding_layer(input_tensor).clone()
        
        # Simulate training step
        optimizer = torch.optim.SGD(embedding_layer.parameters(), lr=0.1)
        loss = embedding_layer(input_tensor).sum()
        loss.backward()
        optimizer.step()
        
        after = embedding_layer(input_tensor)
        assert not torch.allclose(before, after), "Embeddings should change after training"
        
    def test_out_of_vocab(self, embedding_layer):
        """Test handling of out-of-vocabulary indices"""
        with pytest.raises(IndexError):
            embedding_layer(torch.tensor([[100]]))  # vocab_size=100
            
if __name__ == "__main__":
    pytest.main(["-v", __file__])
