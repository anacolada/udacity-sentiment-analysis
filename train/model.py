import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)
        self.gru = nn.GRU(embedding_dim+2*hidden_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)
        self.dense1 = nn.Linear(in_features=hidden_dim*2, out_features=hidden_dim)  # *2 because bidirectional=True
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dense2 = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)  # map the review words to embeddings
        lstm_out, _ = self.lstm(embeds)  # pass the embeddings through an LSTM module
        embeds_lstm = torch.cat([embeds, lstm_out], 2)  # concatenate the LSTM output with the embedded input
        gru, _ = self.gru(embeds_lstm)  # pass the concatenation through a GRU module
        lstm_out = lstm_out + gru  # summing up the initial LSTM output and the previous GRU output
        lstm_out, _ = lstm_out.max(dim=0, keepdim=False)  # compression to a 2D array of size batchsize x 2*hidden_dim
        out = self.dense1(lstm_out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dense2(out)

        return self.sig(out.squeeze())