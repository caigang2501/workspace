import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg



img = (np.random.randn(15,15)+2.5)*50//1
plt.imshow(img)
plt.show()
img = Image.fromarray(img)
# img = mpimg.imread('tangwei.jpg')
plt.imshow(img)
plt.show()
img.thumbnail((7,7))
img = mpimg.pil_to_array(img)
plt.imshow(img[:,:,0])
plt.show()
def drow(t):
    x,y = t
    fig,ax = plt.subplots()
    ax.plot(x,y)
    plt.show()

