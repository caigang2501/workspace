import numpy as np
import matplotlib.pyplot as plt

# 生成一个示例信号
fs = 1000  # 采样频率
t = np.arange(0, 1, 1/fs)  # 时间数组
signal = 5 * np.sin(2 * np.pi * 50 * t) + 3 * np.sin(2 * np.pi * 120 * t)

# 计算傅里叶变换
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), 1/fs)  # 获取频率轴

# 振幅谱
amplitude_spectrum = np.abs(fft_result)

# 相位谱
phase_spectrum = np.angle(fft_result)

# 绘制振幅谱
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(frequencies, amplitude_spectrum)
plt.title('Amplitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

# 绘制相位谱
plt.subplot(2, 1, 2)
plt.plot(frequencies, phase_spectrum)
plt.title('Phase Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')

plt.tight_layout()
plt.show()
