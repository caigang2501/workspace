from torchvision.io import read_image
import os,sys
sys.path.append(os.getcwd())
# C:\Users\EDY\.cache\huggingface\hub\models--timm--resnet50.a1_in1k
def test():
    print(chr(97))
    print(ord('z'))

    l = [[1,4],[2,3]]
    l.sort(key=lambda x:x[1])
    print(l)

    arr = [2,3,1]
    sorted_indices = sorted(range(len(arr)), key=lambda k: arr[k])

if __name__=='__main__':
    pass


