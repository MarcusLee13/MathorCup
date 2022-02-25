import os
from PIL import Image
from torch.utils import data


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    """Judge a file is an image file or not."""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path,channel):
    """Open the image in `path` as RGB form."""
    if channel==3:
        return Image.open(path).convert('RGB')
    elif channel==1:
        return Image.open(path).convert('L')


def make_dataset(dir, max_dataset_size=float('inf')):
    """Get `max_dataset_size` image file paths at most in `dir` and its subdir."""
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    
    # test all path
    # print(images)
    images.sort()
    print(images)
    # print()

    return images[:min(max_dataset_size, len(images))]



class ImageFolder(data.Dataset):
    """A modified ImageFolder class.
    
    Modified by the official PyTorch ImageFolder class (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
    so that this class can load images from both current directory and its subdirectories.
    """
    def __init__(self, ori_root,noise_root,mask_root, transform=None, loader=default_loader):
        images = make_dataset(ori_root)
        noise = make_dataset(noise_root)
        mask = make_dataset(mask_root)
        

        if len(images) == 0:
            raise(RuntimeError("Found 0 images in: " + ori_root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        assert len(images)==len(mask)
        assert len(images)==len(noise)

        self.ori_root = ori_root
        self.noise_root = noise_root
        self.mask_root = mask_root

        self.images = images
        self.noise = noise
        self.mask=mask

        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):

        ori_path = self.images[index]
        noise_path = self.noise[index]
        mask_path = self.mask[index]
        # print(ori_path)
        # print(noise_path)
        # print(mask_path)

        image = self.loader(ori_path,3)
        noise = self.loader(noise_path,3)
        mask = self.loader(mask_path,1)

        if self.transform is not None:
            image = self.transform(image)
            noise = self.transform(noise)
            mask = self.transform(mask)
        
        return (image,noise,mask)
    
    def __len__(self):
        return len(self.images)
