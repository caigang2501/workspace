class EncoderLayer(nn.Module):
    def __init__(self, feature_size, num_heads, feedforward_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # Self-Attention 层
        self.self_attention = nn.MultiheadAttention(feature_size, num_heads, dropout=dropout)

        # 前馈神经网络层
        self.feedforward = nn.Sequential(
            nn.Linear(feature_size, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, feature_size)
        )

        # 残差连接和层归一化
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 多头自注意力
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈神经网络
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x
