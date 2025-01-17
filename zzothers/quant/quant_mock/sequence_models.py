import sys,os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# sys.path.append(os.getcwd())


seq_len = 20
epochs = 40

def create_data(seq_len, num_samples=1000):
    X = []
    y = []
    for _ in range(num_samples):
        start = np.random.rand() * 2 * np.pi
        seq_x = np.sin(np.linspace(start, start + seq_len * 0.1, seq_len))
        seq_y = np.sin(start + seq_len * 0.1+seq_len * 0.1/(seq_len-1))
        X.append(seq_x)
        y.append(seq_y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def generate_data(seq,seq_len):
    X,y = [],[]
    i = seq_len
    while i<len(seq):
        X.append(seq[i-seq_len:i])
        y.append(seq[i])
    return torch.tensor(X, dtype=torch.float32).flatten(), torch.tensor(y, dtype=torch.float32)


class TransformerTimeSeries(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, output_size=1):
        super(TransformerTimeSeries, self).__init__()
        self.input_layer = nn.Linear(input_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.output_layer = nn.Linear(d_model, output_size)
    
    def forward(self, src):
        src = self.input_layer(src)
        src = src.permute(1, 0, 2)
        transformer_output = self.transformer(src, src)  # __call__()
        output = self.output_layer(transformer_output[-1])
        return output

def train_tf():
    X, y = create_data(seq_len)
    model = TransformerTimeSeries()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X.unsqueeze(-1))  # 添加特征维度
        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    torch.save(model.state_dict(), 'data/models/transformer20.pth')

def test_tf():
    model = TransformerTimeSeries()
    model.load_state_dict(torch.load('data/models/transformer20.pth'))
    model.eval()

    benifit,principal = 0,100
    X, y = create_data(seq_len)
    for i in range(20):
        with torch.no_grad():
            sample_input = X[i].unsqueeze(0).unsqueeze(-1)
            prediction = model(sample_input)
            benifit += principal*(y[i].item()-X[i][-1])/abs(X[i][-1])
            print(f"True value: {y[i].item()}, Predicted value: {prediction.item()}",X[i][-1],principal*(y[i].item()-X[i][-1])/abs(X[i][-1]))
    
    print(benifit/principal)

if __name__=='__main__':
    pass
    # train_tf()
    test_tf()
