from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = np.asarray(Image.open('data/imgs/dog1.jpg'))
print(img.shape)

imgplot = plt.imshow(img)
print(type(imgplot))
plt.show()