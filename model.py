import torch
import torch.utils.data
import numpy as np
import main
from gensim.models import KeyedVectors

#word_vectors = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', binary=False)

class SentimentAnalysisModel(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, word_vectors):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.word_vectors = word_vectors
        #self.embeddings = torch.nn.Embedding(vocab_size, embedding_size)
        self.hidden = torch.nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.out = torch.nn.Linear(hidden_size, 1)        
        
    def forward(self, inputs):
        input_pad_lengths = (torch.tensor((inputs.numpy() == self.word_vectors['PAD']).astype(int)).sum(dim=2)).sum(dim=1) / self.embedding_size 
        output, hidden = self.hidden(inputs)
        hiddens = output[np.arange(inputs.shape[0]),19 - input_pad_lengths,:]
        sentiments = self.out(hiddens)        
        return sentiments