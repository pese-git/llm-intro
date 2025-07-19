import torch
import pytest
from positional_embeddings import PositionalEmbeddings

class TestPositionalEmbeddings:
    @pytest.fixture
    def pos_encoder(self):
        return PositionalEmbeddings(max_seq_len=100, emb_size=64)

    def test_output_shape(self, pos_encoder):
        output = pos_encoder(10)
        assert output.shape == (10, 64)

    def test_embedding_layer(self, pos_encoder):
        assert isinstance(pos_encoder.embedding, torch.nn.Embedding)
        assert pos_encoder.embedding.num_embeddings == 100
        assert pos_encoder.embedding.embedding_dim == 64

    def test_out_of_range(self, pos_encoder):
        with pytest.raises(IndexError):
            pos_encoder(101)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
