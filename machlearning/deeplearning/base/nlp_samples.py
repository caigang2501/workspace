import numpy as np

class TwoLayerRNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        # 初始化权重
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        
        self.Whh1 = np.random.randn(hidden_size1, hidden_size1) * 0.01  # 第一隐藏层的隐藏状态权重
        self.Wxh1 = np.random.randn(hidden_size1, input_size) * 0.01    # 输入到第一隐藏层权重
        self.bh1 = np.zeros((hidden_size1, 1))                          # 第一隐藏层偏置
        
        self.Whh2 = np.random.randn(hidden_size2, hidden_size1) * 0.01  # 第二隐藏层的隐藏状态权重
        self.bh2 = np.zeros((hidden_size2, 1))                          # 第二隐藏层偏置
        
        self.Why = np.random.randn(output_size, hidden_size2) * 0.01    # 第二隐藏层到输出权重
        self.by = np.zeros((output_size, 1))                            # 输出偏置

    def forward(self, inputs):
        h1_prev = np.zeros((self.hidden_size1, 1))  # 初始第一隐藏状态
        h2_prev = np.zeros((self.hidden_size2, 1))  # 初始第二隐藏状态
        self.last_inputs = inputs
        self.last_h1s = {0: h1_prev}
        self.last_h2s = {0: h2_prev}

        # 前向传播
        for t, x in enumerate(inputs):
            x = x.reshape(-1, 1)
            h1_next = np.tanh(self.Wxh1 @ x + self.Whh1 @ h1_prev + self.bh1)
            self.last_h1s[t + 1] = h1_next
            h1_prev = h1_next
            
            h2_next = np.tanh(self.Whh2 @ h1_next + self.bh2)
            self.last_h2s[t + 1] = h2_next
            h2_prev = h2_next

        # 计算输出
        y = self.Why @ h2_next + self.by
        return y, h2_next

    def backward(self, dL_dy, learning_rate=0.001):
        n = len(self.last_inputs)

        # 初始化梯度
        dWhy = np.zeros_like(self.Why)
        dWhh1 = np.zeros_like(self.Whh1)
        dWxh1 = np.zeros_like(self.Wxh1)
        dWhh2 = np.zeros_like(self.Whh2)
        dbh1 = np.zeros_like(self.bh1)
        dbh2 = np.zeros_like(self.bh2)
        dby = np.zeros_like(self.by)

        dh2_next = np.zeros_like(self.last_h2s[0])
        dh1_next = np.zeros_like(self.last_h1s[0])

        # 反向传播
        for t in reversed(range(n)):
            x = self.last_inputs[t].reshape(-1, 1)
            h1 = self.last_h1s[t + 1]
            h1_prev = self.last_h1s[t]
            h2 = self.last_h2s[t + 1]
            h2_prev = self.last_h2s[t]

            # 输出层梯度
            dWhy += dL_dy @ h2.T
            dby += dL_dy

            # 第二隐藏层梯度
            dh2 = self.Why.T @ dL_dy + dh2_next
            dh2_raw = (1 - h2 * h2) * dh2

            dbh2 += dh2_raw
            dWhh2 += dh2_raw @ h1.T
            dh1 = self.Whh2.T @ dh2_raw + dh1_next

            # 第一隐藏层梯度
            dh1_raw = (1 - h1 * h1) * dh1

            dbh1 += dh1_raw
            dWxh1 += dh1_raw @ x.T
            dWhh1 += dh1_raw @ h1_prev.T

            dh1_next = self.Whh1.T @ dh1_raw
            dh2_next = self.Whh2.T @ dh2_raw

        # 梯度下降更新权重
        for dparam in [dWxh1, dWhh1, dWhh2, dWhy, dbh1, dbh2, dby]:
            np.clip(dparam, -1, 1, out=dparam)  # 防止梯度爆炸

        self.Whh1 -= learning_rate * dWhh1
        self.Wxh1 -= learning_rate * dWxh1
        self.Whh2 -= learning_rate * dWhh2
        self.Why -= learning_rate * dWhy
        self.bh1 -= learning_rate * dbh1
        self.bh2 -= learning_rate * dbh2
        self.by -= learning_rate * dby

def test_rnn():
    input_size = 5
    hidden_size1 = 10
    hidden_size2 = 10
    output_size = 1

    rnn = TwoLayerRNN(input_size, hidden_size1, hidden_size2, output_size)

    inputs = [np.random.randn(input_size) for _ in range(3)]

    output, hidden = rnn.forward(inputs)
    print('Output:', output)

    dL_dy = output - 1

    rnn.backward(dL_dy)

    print('Updated Weights:')
    print('Whh1:', rnn.Whh1)
    print('Wxh1:', rnn.Wxh1)
    print('Whh2:', rnn.Whh2)
    print('Why:', rnn.Why)
    print('bh1:', rnn.bh1)
    print('bh2:', rnn.bh2)
    print('by:', rnn.by)


class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size

        # 遗忘门参数
        self.Wf = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))

        # 输入门参数
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))

        # 候选记忆细胞状态参数
        self.Wc = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))

        # 输出门参数
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.bo = np.zeros((hidden_size, 1))

        # 输出层参数
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h_prev = np.zeros((self.hidden_size, 1))
        c_prev = np.zeros((self.hidden_size, 1))

        self.last_inputs = inputs
        self.last_hs = {0: h_prev}
        self.last_cs = {0: c_prev}

        for t, x in enumerate(inputs):
            x = x.reshape(-1, 1)
            concat = np.vstack((h_prev, x))

            ft = self._sigmoid(np.dot(self.Wf, concat) + self.bf)
            it = self._sigmoid(np.dot(self.Wi, concat) + self.bi)
            cct = np.tanh(np.dot(self.Wc, concat) + self.bc)
            c_next = ft * c_prev + it * cct
            ot = self._sigmoid(np.dot(self.Wo, concat) + self.bo)
            h_next = ot * np.tanh(c_next)

            self.last_hs[t + 1] = h_next
            self.last_cs[t + 1] = c_next

            h_prev = h_next
            c_prev = c_next

        y = np.dot(self.Wy, h_next) + self.by

        return y, h_next, c_next

    def backward(self, dL_dy, learning_rate=0.001):
        n = len(self.last_inputs)

        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)

        dWhh = np.zeros_like(self.Wf)
        dWxh = np.zeros_like(self.Wf)
        dbh = np.zeros_like(self.bf)

        dh_next = np.zeros_like(self.last_hs[0])
        dc_next = np.zeros_like(self.last_cs[0])

        for t in reversed(range(n)):
            x = self.last_inputs[t].reshape(-1, 1)
            h = self.last_hs[t + 1]
            c = self.last_cs[t + 1]
            c_prev = self.last_cs[t]
            h_prev = self.last_hs[t]

            concat = np.vstack((h_prev, x))

            dy = dL_dy
            dWy += dy @ h.T
            dby += dy

            dh = self.Wy.T @ dy + dh_next

            do = dh * np.tanh(c)
            do = self._sigmoid_deriv(do)
            dWo = do @ concat.T
            dbo = do

            dc = dh * self._sigmoid(h) * (1 - np.tanh(c) ** 2) + dc_next
            dcct = dc * self._sigmoid(c)
            dcct = (1 - np.tanh(dcct) ** 2)
            dWc = dcct @ concat.T
            dbc = dcct

            di = dc * np.tanh(c)
            di = self._sigmoid_deriv(di)
            dWi = di @ concat.T
            dbi = di

            df = dc * c_prev
            df = self._sigmoid_deriv(df)
            dWf = df @ concat.T
            dbf = df

            dconcat = np.dot(self.Wf.T, df) + np.dot(self.Wi.T, di) + np.dot(self.Wc.T, dcct) + np.dot(self.Wo.T, do)

            dh_next = dconcat[:self.hidden_size, :]
            dc_next = dc * self._sigmoid(c)

            for dparam in [dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo]:
                np.clip(dparam, -1, 1, out=dparam)  # 防止梯度爆炸

            self.Wf -= learning_rate * dWf
            self.Wi -= learning_rate * dWi
            self.Wc -= learning_rate * dWc
            self.Wo -= learning_rate * dWo
            self.bf -= learning_rate * dbf
            self.bi -= learning_rate * dbi
            self.bc -= learning_rate * dbc
            self.bo -= learning_rate * dbo

        self.Wy -= learning_rate * dWy
        self.by -= learning_rate * dby

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _sigmoid_deriv(x):
        return x * (1 - x)

def test_lstm():
    input_size = 5
    hidden_size = 10
    output_size = 1

    # 创建一个LSTM实例
    lstm = LSTM(input_size, hidden_size, output_size)

    # 假设输入序列有3个时间步，每个时间步有5个特征
    inputs = [np.random.randn(input_size) for _ in range(3)]

    # 前向传播
    output, hidden, cell = lstm.forward(inputs)
    print('Output:', output)

    # 计算损失函数对输出的梯度（假设损失是1）
    dL_dy = output - 1

    # 反向传播
    lstm.backward(dL_dy)

    print('Updated Weights:')
    print('Wf:', lstm.Wf)
    print('Wi:', lstm.Wi)
    print('Wc:', lstm.Wc)
    print('Wo:', lstm.Wo)
    print('Wy:', lstm.Wy)
    print('bf:', lstm.bf)
    print('bi:', lstm.bi)
    print('bc:', lstm.bc)
    print('bo:', lstm.bo)
    print('by:', lstm.by)


if __name__=='__main__':
    test_rnn()
