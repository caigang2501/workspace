import numpy as np
import matplotlib.pyplot as plt
import random

fs = 1000  # 采样频率
t = np.arange(0, 1, 1/fs)  # 从0到1秒，步长为1/fs

frequency = 2  # 信号频率 5 Hz
A = 1  # 振幅
# noise = 0.5 * np.random.normal(size=t.shape)  # 添加噪声
noise = 0
f = A*np.sin(2*np.pi*frequency*t)
for n in [5,11,17]:
    At = 1
    frequency_t = n
    f += np.sin(2*np.pi*frequency_t*(t+1/n/3*random.randint(1,3)))
signal = f + noise

F = np.fft.fft(signal)
F_magnitude = np.abs(F)  # 幅度谱
F_frequency = np.fft.fftfreq(len(F), d=1/fs)  # 频率

# 绘制原始信号和傅里叶变换
plt.figure(figsize=(12, 8))

# 绘制原始信号
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='orignal')
plt.title('time signal')
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.grid()
plt.legend()

# 绘制傅里叶变换结果
plt.subplot(2, 1, 2)
plt.plot(F_frequency[:len(F_frequency)//2], F_magnitude[:len(F_magnitude)//2], label='amplitude', color='orange')
plt.title('frequency signal (fourier_transform)')
plt.xlabel('frequency (Hz)')
plt.ylabel('amplitude')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
