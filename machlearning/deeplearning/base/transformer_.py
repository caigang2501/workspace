import numpy as np

# 多头注意力机制
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.wq = np.random.randn(d_model, d_model)
        self.wk = np.random.randn(d_model, d_model)
        self.wv = np.random.randn(d_model, d_model)
        self.dense = np.random.randn(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = np.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return np.transpose(x, (0, 2, 1, 3))

    def call(self, q, k, v):
        batch_size = q.shape[0]
        q = np.dot(q, self.wq)
        k = np.dot(k, self.wk)
        v = np.dot(v, self.wv)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention_logits = np.matmul(q, k.T) / np.sqrt(self.depth)
        attention_weights = np.softmax(scaled_attention_logits, axis=-1)
        output = np.matmul(attention_weights, v)
        output = np.transpose(output, (0, 2, 1, 3))
        output = np.reshape(output, (batch_size, -1, self.d_model))
        output = np.dot(output, self.dense)
        return output


# Transformer 编码器
class EncoderLayer:
    def __init__(self, d_model, num_heads, dff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = np.random.randn(d_model, dff)
        self.layernorm1 = np.random.randn(d_model)
        self.layernorm2 = np.random.randn(d_model)

    def call(self, x):
        attn_output = self.mha.call(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = np.dot(out1, self.ffn)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


# Transformer 模型
class Transformer:
    def __init__(self, num_layers, d_model, num_heads, dff):
        self.num_layers = num_layers
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff) for _ in range(num_layers)]

    def call(self, x):
        for i in range(self.num_layers):
            x = self.encoder_layers[i].call(x)
        return x

def test():
    num_layers = 2
    d_model = 256
    num_heads = 8
    dff = 512
    input_seq_length = 20
    batch_size = 64

    transformer = Transformer(num_layers, d_model, num_heads, dff)
    inputs = np.random.randn(batch_size, input_seq_length, d_model)
    outputs = transformer.call(inputs)
    print(outputs.shape)  # 应该输出 (64, 20, 256)


if __name__=='__main__':
    test()
