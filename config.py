import time
import torch


ngpu = torch.cuda.device_count()
workers = 4
image_size = 256
training_dataset_size = 5

cur_time = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
ROOT = '/content'
DATA_DIR = '/content/drive/MyDrive/ARGv2/DCUdataset'
experiment_dir = ROOT + '/segment/' + cur_time
config_path = experiment_dir + "/config.txt"
loss_diff_path=experiment_dir + "/loss_diff.csv"
log_path = experiment_dir + '/train_log.txt'
checkpoint_path = experiment_dir + '/checkpoint'
train_pics_save_path = experiment_dir + '/train_pics'
train_loss_save_path = experiment_dir + '/train_loss.png'
val_loss_save_path = experiment_dir + '/val_loss.png'
val_pics_save_path = experiment_dir + '/val_pics'
test_path = ''
test_pics_save_path = ''
checkpoint = ''
checkpoint_diff = ''

epochs = 200
batch_size = 1
beta = 0.75
lr = 0.001
lr_decay_freq = 30
iters_per_epoch = int(training_dataset_size / batch_size)

log_freq = 10
result_pic_freq = 100 # frequcey to print log

cover_dependent = False
channel_mask = 1
channel_cover = 3
num_downs = 5
norm_type = 'batch'
loss = 'l2'

loss_diff=0.0001
