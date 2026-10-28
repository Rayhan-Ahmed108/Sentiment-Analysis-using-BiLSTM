import torch 
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim, attention_dim = 64):
        super().__init__()
        self.linear = nn.Linear(encoder_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias = False)

    def forward(self, outputs, mask):
        energy = torch.tanh(self.linear(outputs))#convert 128 dimension to 64 dimension
        scores = self.score(energy).squeeze(-1)

        #ignore padding tokens
        scores = scores.masked_fill(~mask, -1e9)

        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)

        return context, weights

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 100, hidden_dim = 64, attention_dim = 64, dropout = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)

        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, batch_first = True, bidirectional = True)

        encoder_dim = hidden_dim * 2

        self.attention = AdditiveAttention(encoder_dim, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder_dim, 2)
    def forward(self, input_ids, lengths):
        #input_ids = (batch, seq_len)
        embedded = self.embedding(input_ids)

        #ignore padding while processing sequences
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first = True, enforce_sorted = False)

        packed_outputs, _ = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first = True, total_length = input_ids.size(1))

        positions = torch.arange(input_ids.size(1), device = input_ids.device).unsqueeze(0)
        mask = positions < lengths.to(input_ids.device).unsqueeze(1)

        context, weigths = self.attention(outputs, mask)
        logits = self.classifier(self.dropout(context))

        return logits, weigths