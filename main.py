import os
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from utils import *
# from networks import *
from NestedUNet import *
# from networks_unet import *

from noise_filter import *
# from unet import *
def debug_f():
    torch.autograd.set_detect_anomaly(True)
def main():
    debug_f()
    # judge the tunning status
    if torch.cuda.is_available():
        print("----GPU----")
    else :
        print("----CPU----")
    
    cudnn.benchmark = True # speed up

    assert config.DATA_DIR # if data_dir


    try:
        if config.test_path == "":
            if not os.path.exists(config.checkpoint_path):
                os.makedirs(config.checkpoint_path)
            if not os.path.exists(config.train_pics_save_path):
                os.makedirs(config.train_pics_save_path)
            if not os.path.exists(config.val_pics_save_path):
                os.makedirs(config.val_pics_save_path)
        else:
            pass
    except OSError:
        print("mkdir failed")

    save_config()

    # get the cover and mask 
    # train_dir = os.path.join(config.DATA_DIR,'train')
    # val_dir = os.path.join(config.DATA_DIR,'val')

    # train_cover_dir =os.path.join(train_dir,'ori')
    # train_noise_dir =os.path.join(train_dir,'noise_20')
    # train_mask_dir = os.path.join(train_dir,'mask')

    # val_cover_dir =os.path.join(val_dir,'ori')
    # val_noise_dir =os.path.join(val_dir,'noise_20')
    # val_mask_dir = os.path.join(val_dir,'mask')

    train_cover_dir =os.path.join(config.DATA_DIR,'./0/train')
    train_noise_dir =os.path.join(config.DATA_DIR,'./90/train')
    train_mask_dir = os.path.join(config.DATA_DIR,'./label/train')

    val_cover_dir =os.path.join(config.DATA_DIR,'./0/val')
    val_noise_dir =os.path.join(config.DATA_DIR,'./90/val')
    val_mask_dir = os.path.join(config.DATA_DIR,'./label/val')

    

    # data preprocess
    transform = transforms.Compose([
        transforms.Resize([config.image_size, config.image_size]),
        transforms.ToTensor()
    ])

    # read data image
    if config.test_path=="":
        train_data=ImageFolder(train_cover_dir,train_noise_dir,train_mask_dir,transform)
        val_data=ImageFolder(val_cover_dir,val_noise_dir,val_mask_dir,transform)
    else:
        pass

    # init the Hnet model
    Hnet = UnetGenerator(
        input_nc=config.channel_cover,
        output_nc=config.channel_mask,
        num_downs=config.num_downs,
        norm_type=config.norm_type,
        output_function='sigmoid'
    )
    Fnet = Filter_net(
        input_nc=config.channel_cover,
        output_nc=config.channel_cover,
        num_downs=config.num_downs,
        norm_type=config.norm_type,
        output_function='sigmoid'
    )

    # Fnet = noise_filters(
    #     input_nc=config.channel_cover,
    #     output_nc=config.channel_cover
    # )

    Hnet.apply(weights_init)
    Fnet.apply(weights_init)

    Hnet = torch.nn.DataParallel(Hnet).cuda()
    Fnet = torch.nn.DataParallel(Fnet).cuda()

    if config.checkpoint!="":
        print("retrain")
        checkpoint=torch.load(config.checkpoint)
        Hnet.load_state_dict(checkpoint["H_state_dict"])
        Fnet.load_state_dict(checkpoint["F_state_dict"])


    if config.loss == "l1":
        criterion=nn.L1loss().cuda()
    elif config.loss =="l2":
        criterion=nn.MSELoss().cuda()
    else:
        print("config not define the loss funtion")

    if config.test_path=="":
        params = list(Hnet.parameters())+list(Fnet.parameters())
        optimizer = optim.Adam(params, lr=config.lr, betas=(0.5, 0.999))
        scheduler=ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=8,
            verbose=True
        )

        train_loader=DataLoader(
            train_data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=int(config.workers)
        )
        

        val_loader=DataLoader(
            val_data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=int(config.workers)
        )

        
        train(
            # zip loader in train
            train_loader,
            val_loader,
            Hnet,Fnet,
            optimizer,
            scheduler,
            criterion
        )


if __name__=="__main__":
    main()