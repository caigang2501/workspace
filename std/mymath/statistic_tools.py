import matplotlib.pyplot as plt
import random


def plot_distribution(getrandom,*args,num_samples=100000):
    samples = [getrandom(*args) for _ in range(num_samples)]
    print(min(samples),max(samples))
    plt.hist(samples, bins=30, density=True, alpha=0.7, color='b')
    plt.title('Random Number Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def getrandom(min_val, max_val):
    return random.normalvariate(min_val, max_val)
if __name__=='__main__':
    plot_distribution(getrandom, 0, 33)






