import os
import time
import shutil
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.utils as vutils

import config
from image_folder import ImageFolder

import pandas as pd
import numpy as np

from PIL import Image
class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def weights_init(m):
    """init the weight for a network"""
    classname=m.__class__.__name__
    # print(classname)
    if classname.find("conv2d")!=-1:
        nn.init.kaiming_normal_(
            m.weight.data,
            a=0,
            mode="fan_out"
        )
    elif classname.find("BatchNorm")!=-1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)

def print_log(log_info,log_path=config.log_path,
console=True,debug=False):
    """print log information to the console and log files."""
    if console:# print the info into console
        print(log_info)
    if not debug:#debug mode don't write the log into files
        # write the log into log file
        if not os.path.exists(log_path):
            fp=open(log_path,"w")
            fp.writelines(log_info+"\n")
            fp.close()
        else:
            with open(log_path,"a+") as f:
                f.writelines(log_info+"\n")

def print_network(net,log_path=config.log_path):
    num_params =0
    for param in net.parameters():
        num_params+=param.numel()
    print_log(str(net),log_path)
    print_log('Total number of parameters: %d\n' % num_params, log_path)


def save_config():
    """Save configuations as .txt file."""
    fp = open(config.config_path, "w")
    # fp.writelines("ngpu\t\t\t\t%d\n" % config.ngpu)
    fp.close()


def save_checkpoint(state,is_best,epoch,
prefix,only_save_best=True):
    """save checkpoint files for training"""
    filename = '%s/checkpoints_%03d.pth.tar' % (config.checkpoint_path, epoch)
    if only_save_best:
        filename = '%s/checkpoints_best.pth.tar' % (config.checkpoint_path)
    torch.save(state, filename)
    if is_best and not only_save_best:
        shutil.copyfile(filename, '%s/best_checkpoint_%03d.pth.tar' % (config.checkpoint_path, epoch))

# save the generated mask
def save_result_pic(batch_size,cover,noise,generated_ori,true_mask,get_mask,epoch,batch_ith,save_path):
    result_name = '%s/result_pic_epoch%03d_batch%04d.png' % (save_path, epoch, batch_ith)

    mask_gap=true_mask-get_mask
    # print(cover.shape,generated_ori.shape,true_mask.shape,get_mask.shape)
    show_all = torch.cat((cover.clone(), noise.clone(),generated_ori.clone(),true_mask.clone().repeat(1,3,1,1), get_mask.clone().repeat(1,3,1,1)), dim=0)
    # print(show_all.shape)
    # vutils.save_image(show_all, result_name, batch_size, padding=1, normalize=True)
    grid = vutils.make_grid(show_all, nrow=batch_size, padding=1, normalize=False)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(result_name)

def save_csv(tensor):
    print(tensor.shape)
    print(tensor.view(config.image_size,-1).shape)
    # try:
    a=pd.DataFrame(tensor.view(config.image_size,-1).cpu().numpy())
    a.to_csv(config.experiment_dir+"/saved.csv")
    print("end")
    return True
    # except:
        # return False


# plot the curve of Hloss
def save_loss_pic(h_losses_list,save_path):
    plt.title("Training Loss for H")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.plot(list(range(1, len(h_losses_list)+1)), h_losses_list, label='H loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

import numpy as np
def save_two_loss_pic(train_h_loss_all,Val_h_loss_all,train_f_loss_all,Val_f_loss_all,save_path):
    plt.title("Training Loss and val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.plot(list(range(1, len(train_h_loss_all)+1)), train_h_loss_all, label='train h loss')
    plt.plot(list(range(1, len(Val_h_loss_all)+1)), Val_h_loss_all, label='val h loss')
    plt.plot(list(range(1, len(train_f_loss_all)+1)), train_f_loss_all, label='train f loss')
    plt.plot(list(range(1, len(Val_f_loss_all)+1)), Val_f_loss_all, label='val f loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def adjust_learning_rate(optimizer, epoch):
    """Set the learning rate to the initial LR decayed by 10 every `lr_decay_freq` epochs."""
    lr = config.lr * (0.1 ** (epoch // config.lr_decay_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def forward_pass(cover,noise,true_mask,Hnet,Fnet,criterion):
    """forward propagation for H net and caculate loss and APD
    
    Parameters:
        true_mask: the origin mask
        Hnet: the Hnet
        criterion: loss function
    """
    cover=cover.cuda()
    true_mask = true_mask.cuda()
    noise=noise.cuda()

    generated_ori=Fnet(noise)
    generated_mask=Hnet(generated_ori)
    
    # print(generated_ori.shape)

    F_loss= criterion(cover,generated_ori)
    H_loss = criterion(true_mask,generated_mask)

    H_diff = (true_mask - generated_mask).abs().mean() * 255
    F_diff= (cover - generated_ori).abs().mean() * 255

    return cover,noise,true_mask,generated_ori,generated_mask,H_loss,H_diff,F_loss,F_diff


def validation(val_loader,epoch,Hnet,Fnet,criterion):
    print("### validation begin###")

    batch_size=config.batch_size

    Hlosses=AverageMeter()
    Flosses=AverageMeter()

    Hdiff=AverageMeter()
    Fdiff=AverageMeter()


    # turn on val mode
    Hnet.eval()
    Fnet.eval()

    for i,(cover,noise,true_mask) in enumerate(val_loader,start=1):
        cover,noise,true_mask,generated_ori,generated_mask,H_loss,H_diff,F_loss,F_diff=forward_pass(cover,noise,true_mask,Hnet,Fnet,criterion)

        Hlosses.update(H_loss.item(),batch_size)
        Flosses.update(F_loss.item(),batch_size)
        Hdiff.update(H_diff.item(),batch_size)
        Fdiff.update(F_diff.item(),batch_size)
        
        if i==1:
            save_result_pic(
                batch_size,
                cover.detach(), 
                noise.detach(),
                generated_ori.detach(),
                true_mask.detach(), 
                generated_mask.detach(),
                epoch, i,
                config.val_pics_save_path
            )
    
    # print log detail
    val_log = 'Validation[%02d]\tval_Hloss: %.6f\tval_Hdiff:%.6f\tF_loss:%.6f\tF_loss:%.6f' % (
        epoch,
        Hlosses.avg, Hdiff.avg, 
        Flosses.avg, Fdiff.avg, 
    )
    print_log(val_log)
    print("#### validation end ####\n")
    return Hlosses.avg,Hdiff.avg,Flosses.avg, Fdiff.avg, 
import pandas as pd
def train(train_loader,val_loader,Hnet,Fnet,optimizer,scheduler,criterion):
    """Train Hnet and Rnet and schedule learning rate by the validation results.
    
    Parameters:
        train_loader_cover     -- train_loader for cover images
        train_loader_mask      -- train_loader for mask images
        val_loader_cover       -- val_loader for cover images
        val_loader_mask        -- val_loader for mask images
        Hnet (nn.Module)        -- hiding network
        optimizer               -- optimizer for Hnet 
        scheduler               -- scheduler for optimizer to set dynamic learning rate
        criterion               -- loss function
        
    """

    

    MIN_LOSS = 0x3f3f3f3f
    h_losses_list= []
    f_losses_list=[]
    print("######## TRAIN BEGIN ########")
    learning_rate=[]

    train_h_loss_all=[]
    train_h_diff_all=[]
    train_f_loss_all=[]
    train_f_diff_all=[]

    Val_h_loss_all=[]
    Val_f_loss_all=[]
    Val_h_diff_all=[]
    Val_f_diff_all=[]
    for epoch in range(config.epochs):
        rate=adjust_learning_rate(optimizer,epoch)
        
        # training information
        batch_time = AverageMeter()  # time for processing a batch
        data_time = AverageMeter()   # time for reading data
        Hlosses = AverageMeter()     # losses for hiding network
        Flosses = AverageMeter()     # losses for hiding network
        Hdiff = AverageMeter()       # APD for hiding network (between container and cover)
        Fdiff = AverageMeter()       # APD for hiding network (between container and cover)
        SumLosses = AverageMeter()   # losses sumed by H and R with a factor beta(0.75 for default)
        
        
        # turn on training mode
        Hnet.train()
        Fnet.train()


        start_time = time.time()

        for i,(cover,noise,true_mask)in enumerate(train_loader):
            # print(cover.shape)
            data_time.update(time.time()-start_time)
            batch_size = config.batch_size

            cover,noise,true_mask,generated_ori,generated_mask,H_loss,H_diff,F_loss,F_diff=forward_pass(cover,noise,true_mask,Hnet,Fnet,criterion)
            
            Hlosses.update(H_loss.item(),batch_size)
            Flosses.update(F_loss.item(),batch_size)
            
            Hdiff.update(H_diff.item(), batch_size)
            Fdiff.update(F_diff.item(), batch_size)
            
            loss_sum = config.loss_diff*H_loss+F_loss
            
            SumLosses.update(loss_sum.item(), batch_size)

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            batch_time.update(time.time() - start_time)
            start_time = time.time()

            log = '[%02d/%d] [%03d/%d]\tH_loss: %.6f\tH_diff: %.6f\tF_loss: %.6f\tF_loss: %.6f \tdata_time: %.6f\tbatch_time: %.6f' % (
                epoch, config.epochs, i, config.iters_per_epoch,
                Hlosses.val, Hdiff.val, 
                Flosses.val, Fdiff.val, 
                data_time.val, batch_time.val
            )

            # record log
            if i % config.log_freq == 0:
                print(log)
            if epoch == 0 and i % config.result_pic_freq == 0:
                save_result_pic(
                    batch_size,
                    cover.detach(),
                    noise.detach(), 
                    generated_ori.detach(),
                    true_mask.detach(), 
                    generated_mask.detach(),
                    epoch, i,
                    config.train_pics_save_path
                )
            if i == config.iters_per_epoch:# end of loop
                # print("end of loop")
                break
        save_result_pic(
            batch_size,
            cover.detach(), 
            noise.detach(),
            generated_ori.detach(),
            true_mask.detach(), 
            generated_mask.detach(),
            epoch, i,
            config.train_pics_save_path
        )

        # epoch_log
        epoch_log = "Training Epoch[%02d]\tHloss=%.6f\tHdiff=%.6f\tFloss=%.6f\tFdiff=%.6f\tlr= %.6f\tEpoch Time=%.6f" % (
            epoch,
            Hlosses.avg, Hdiff.avg, 
            Flosses.avg, Fdiff.avg, 
            optimizer.param_groups[0]['lr'],
            batch_time.sum
        )
        print_log(epoch_log)

        h_losses_list.append(Hlosses.avg)
        f_losses_list.append(Flosses.avg)
        
        # sane the loss curve
        
        

        #### validation, schedule learning rate and make checkpoint ####
        val_hloss,  val_hdiff,val_floss,  val_fdiff = validation(val_loader, epoch, Hnet,Fnet,  criterion)
        
        
        # scheduler.step(val_hloss)

        sum_diff = val_hdiff + val_fdiff
        is_best = sum_diff < MIN_LOSS
        MIN_LOSS = min(MIN_LOSS, sum_diff)

        if is_best:
            print_log("Save best checkpoint: epoch%03d\n" % epoch)
            save_checkpoint(
                {
                    'epoch': epoch+1,
                    'H_state_dict': Hnet.state_dict(),
                    'F_state_dict': Fnet.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                is_best, epoch,
                '%s/epoch_%d_Hloss_%.6f_Hdiff%.6f_Floss_%.6f_Fdiff%.6f' % (
                    config.checkpoint_path, epoch,
                    val_hloss, 
                    val_hdiff, 
                    val_floss,  
                    val_fdiff
                )
            )
        else:
            print_log("\n")
        if (epoch+1)%50==0:
            print_log("Save checkpoint for 50 epoch: epoch%03d\n" % epoch)
            save_checkpoint(
                {
                    'epoch': epoch+1,
                    'H_state_dict': Hnet.state_dict(),
                    'F_state_dict': Fnet.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                is_best, epoch+1,
                '%s/epoch_%d_Hloss_%.6f_Hdiff%.6f_Floss_%.6f_Fdiff%.6f' % (
                    config.checkpoint_path, epoch,
                    val_hloss, 
                    val_hdiff, 
                    val_floss,  
                    val_fdiff
                ),
                only_save_best=False
            )
        train_h_loss_all.append(Hlosses.avg)
        train_h_diff_all.append(Hdiff.avg)

        train_f_loss_all.append(Flosses.avg)
        train_f_diff_all.append(Fdiff.avg)

        
        
        Val_h_loss_all.append(val_hloss)
        Val_h_diff_all.append(val_hdiff)

        Val_f_loss_all.append(val_floss)
        Val_f_diff_all.append(val_fdiff)


        # save_loss_pic(
        #     Val_loss_all, 
        #     config.val_loss_save_path
        # )
        # save_loss_pic(
        #     h_losses_list, 
        #     config.train_loss_save_path
        # )
        save_two_loss_pic(
            train_h_loss_all,
            Val_h_loss_all,
            train_f_loss_all,
            Val_f_loss_all,
            config.val_loss_save_path
        )
        # Val_diff_all.append(val_hdiff)
        learning_rate.append(rate)
        c={
            "train_hloss" : train_h_loss_all,
            "train_hdiff" : train_h_diff_all,
            "val_hloss" : Val_h_loss_all,
            "val_hdiff" : Val_h_diff_all,

            "train_floss" : train_f_loss_all,
            "train_fdiff" : train_f_diff_all,
            "val_floss" : Val_f_loss_all,
            "val_fdiff" : Val_f_diff_all,

            "learning_rate":learning_rate
        }#将列表a，b转换成字典
        data=pd.DataFrame(c)#将字典转换成为数据框
        data.to_csv(config.loss_diff_path)
        # print(data)
    print("######## TRAIN END ########")