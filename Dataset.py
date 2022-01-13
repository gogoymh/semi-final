import torch
from torch.utils.data import Dataset
from skimage.io import imread
import os
import numpy as np
import random
from torchvision import transforms

class Numberset(Dataset):
    def __init__(self, num):
        super().__init__()
        
        self.len = num
        
    def __getitem__(self, index):
        return index
    
    def __len__(self):
        return self.len

class USDataset(Dataset):
    def __init__(self, dataset, path1, path2, input_transform=None, target_transform=None):
        super().__init__()
        
        self.dataset = dataset
        
        self.path1 = path1
        self.path2 = path2
        
        self.input_transform = input_transform
        self.target_transform = target_transform
        
        self.normalize = transforms.Normalize((0.5,), (0.5,))
        
        if dataset == 'wrist_HM70A':
            self.idx_init = 0
            self.len = 641
        elif dataset == 'forearm_HM70A':
            self.idx_init = 641
            self.len = 162
        elif dataset == 'wrist_miniSONO':
            self.idx_init = 803
            self.len = 311
        elif dataset == 'forearm_miniSONO':
            self.idx_init = 1114
            self.len = 191
        
        print("Dataset: " + self.dataset + " & Length: " + str(self.len))
        
    def __getitem__(self, index):
        
        input_img = imread(os.path.join(self.path1, "%07d.jpg" % (index + self.idx_init)), as_gray=True)
        target_img = imread(os.path.join(self.path2, "%07d.jpg" % (index + self.idx_init)), as_gray=True)
        
        input_img = np.expand_dims(input_img, axis=2).astype('float32')
        target_img = np.expand_dims(target_img, axis=2).astype('float32')
        
        seed = np.random.randint(2147483647)
        
        if self.input_transform is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            input_img = self.input_transform(input_img)

            input_img = self.normalize(input_img)
            
        if self.target_transform is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            target_img = self.target_transform(target_img)
        
        return input_img, target_img
    
    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    #path1 = "C://유민형//개인 연구//MobiusNet//US_Image//train_jpg//"
    #path2 = "C://유민형//개인 연구//MobiusNet//US_Image//target_jpg//"
    
    #path1 = "C://유민형//개인 연구//MobiusNet//US_Image////wrist_HM70A//"
    #path2 = "C://유민형//개인 연구//MobiusNet//US_Image//newset//wrist_target//wrist_HM70A//"
    
    #path1 ="/DATA/ymh_ksw/dataset/US_image/newset/wrist_train/wrist_HM70A"
    #path2 ="/DATA/ymh_ksw/dataset/US_image/newset/wrist_target/wrist_HM70A"
    
    #path1 = "/data/ymh/US_segmentation/newset/wrist_train/wrist_HM70A/"
    #path2 = "/data/ymh/US_segmentation/newset/wrist_target/wrist_HM70A/"
    
    path1 = "/home/compu/ymh/newset/wrist_train/wrist_HM70A"
    path2 = "/home/compu/ymh/newset/wrist_target/wrist_HM70A"
    
    
    from torchvision import transforms as tf
    
    transform_valid = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        tf.Resize((256,256)),
        tf.RandomAffine(0, shear=[-15, 15, -15, 15]),
        tf.ToTensor()
     ])
    
    a = USDataset('wrist_HM70A', path1, path2, transform_valid, transform_valid)
    
    import matplotlib.pyplot as plt
    
    #index = np.random.choice(641, 1)[0]
    index = 100
    b, c = a.__getitem__(index)
    
    '''
    print(b.shape, c.shape)
    print(b.dtype, c.dtype)
    
    print(b[:,:5,:5])
    
    print(c[:,:5,:5])
    
    
    '''
    b = b.squeeze().numpy() * 0.5 + 0.5
    
    plt.imshow(b, cmap="gray")
    plt.savefig("/home/compu/ymh/input.png")
    #plt.savefig("/DATA/ymh_ksw/script/input2.png")
    plt.show()
    plt.close()
    
    c = c.squeeze().numpy()
    
    plt.imshow(c, cmap="gray")
    #plt.savefig("/DATA/ymh_ksw/script/target2.png")
    plt.savefig("/home/compu/ymh/target.png")
    plt.show()
    plt.close()
    