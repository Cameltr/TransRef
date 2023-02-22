import time
from options.train_options import TrainOptions
from data.dataprocess import DataProcess
from models.model import create_model
import torchvision
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
import torch
if __name__ == "__main__":

    opt = TrainOptions().parse()
    # define the dataset
    dataset = DataProcess(opt.de_root,  opt.input_mask_root, opt.ref_root,  opt.isTrain)
    #dataset = DataProcess(opt.de_root, opt.st_root, opt.input_mask_root, opt.ref_root, opt, opt.isTrain)
    iterator_train = (data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.num_workers, drop_last=False, pin_memory=True))
    # Create model
    model = create_model(opt)
    total_steps=0
    # Create the logs
    dir = os.path.join(opt.log_dir, opt.name).replace('\\', '/')
    if not os.path.exists(dir):
        os.mkdir(dir)
    writer = SummaryWriter(log_dir=dir, comment=opt.name)
    # Start Training
    for epoch in range (opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for detail, mask,reference in iterator_train:
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(detail,mask,reference)
            model.optimize_parameters()
            # display the training processing
            if total_steps % opt.display_freq == 0:
                input,reference,output, GT = model.get_current_visuals()
                image_out = torch.cat([reference,input,output,GT], 0)
                grid = torchvision.utils.make_grid(image_out)
                writer.add_image('Epoch_(%d)_(%d)' % (epoch, total_steps + 1), grid, total_steps + 1)
            # display the training loss
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                writer.add_scalar('loss_L1', errors['loss_L1'], total_steps + 1)
                writer.add_scalar('Perceptual_loss', errors['Perceptual_loss'], total_steps + 1)
                writer.add_scalar('Style_loss', errors['Style_loss'], total_steps + 1)
                print('iteration time: %d' % t)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
    model.save_networks(opt.niter + opt.niter_decay + 1)
    writer.close()
