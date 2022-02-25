import os,shutil
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from utils import *
from networks_unet import *
# from NestedUNet import *
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
    test_dir=config.DATA_DIR
    
    


    # get the cover and mask 

    test_cover_dir =os.path.join(test_dir,'./80/test')
    test_mask_dir = os.path.join(test_dir,'/content/mask')
    test_cover2_dir = os.path.join(test_dir,'/content/cover')

    if not os.path.exists(test_mask_dir):
        os.makedirs(test_mask_dir)
    else:
        shutil.rmtree(test_mask_dir)
        os.makedirs(test_mask_dir)


    if not os.path.exists(test_cover2_dir):
        os.makedirs(test_cover2_dir)
    else:
        shutil.rmtree(test_cover2_dir)
        os.makedirs(test_cover2_dir)
        
    

    # data preprocess
    transform = transforms.Compose([
        transforms.Resize([config.image_size, config.image_size]),
        transforms.ToTensor()
    ])

    # read data image
    if config.test_path=="":
        test_data_cover=ImageFolder(test_cover_dir,test_cover_dir,test_cover_dir,transform)
        
    else:
        pass

    # init the Hnet model
    Hnet = UnetGenerator(
        input_nc=config.channel_cover,
        output_nc=config.channel_mask,
        num_downs=config.num_downs,
        norm_type=config.norm_type,
        output_function='tanh'
    )
    Fnet = Filter_net(
        input_nc=config.channel_cover,
        output_nc=config.channel_cover,
        num_downs=config.num_downs,
        norm_type=config.norm_type,
        output_function='sigmoid'
    )
    
    # load weight
    Hnet.apply(weights_init)
    Fnet.apply(weights_init)

    Hnet = torch.nn.DataParallel(Hnet).cuda()
    Fnet = torch.nn.DataParallel(Fnet).cuda()

    check_path="/content/drive/MyDrive/agriculture_recon/segment/unet-80-filter-saved50/checkpoint/checkpoints_200.pth.tar"
    
    print(check_path)
    checkpoint=torch.load(check_path)
    Hnet.load_state_dict(checkpoint["H_state_dict"])
    Fnet.load_state_dict(checkpoint["F_state_dict"])
    
    # print the structure of Hnet
    # print_network(Hnet)

    if config.loss == "l1":
        criterion=nn.L1loss().cuda()
    elif config.loss =="l2":
        criterion=nn.MSELoss().cuda()
    else:
        print("config not define the loss funtion")

    
    

    test_loader_cover=DataLoader(
        test_data_cover,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=int(config.workers)
    )
    
    # test_loader=zip(test_loader_cover,test_loader_cover)

    
    
    Hnet.eval()
    
    for i,(cover_image,_,_) in enumerate(test_loader_cover,start=1):
        

        F_input=cover_image.cuda()

        F_output=Fnet(F_input).cuda()

        H_output=Hnet(F_output).cuda()
        # save=torch.cat((H_input,H_output))
        print(H_output.shape)
        save=H_output
        for k in range(1):
            save_image(torch.reshape(save[k],(1,config.image_size,config.image_size)), os.path.join(test_mask_dir,str(i)+"_"+str(k)+".png"), config.batch_size, padding=0, normalize=True)
            save_image(torch.reshape(F_input[k],(3,config.image_size,config.image_size)), os.path.join(test_cover2_dir,str(i)+"_"+str(k)+".png"), config.batch_size, padding=0, normalize=True)
        

def save_image(img,path, batch_size,padding=1, normalize=True):
    grid = vutils.make_grid(img, nrow=batch_size, padding=1, normalize=True)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

if __name__=="__main__":
    main()



    # if not save_csv(get_mask[0,:].clone().detach()):
    #     print("write error")
    #     raise EOFError