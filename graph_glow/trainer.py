import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt 
from dataset.utils import plot_animation
from .utils import linear_interpolation
from dataset.locomotion import inv_standardize

class Trainer(object):
    def __init__(self, model, optim, schedule, data, logdir, device, cfg):
        
        self.log_dir = logdir
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        self.plot_dir = os.path.join(self.log_dir, "plot")
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        cfg.dump(self.log_dir)

        self.checkpoints_gap = cfg.Train.check_gap
        
        self.model = model
        self.device = device
        self.optim = optim
        self.schedule = schedule
        self.init_lr = cfg.Optim.lr
        
        self.max_grad_clip = cfg.Train.max_grad_clip
        self.max_grad_norm = cfg.Train.max_grad_norm

        self.num_epochs = cfg.Train.num_epochs
        self.batch_size = cfg.Train.batch_size
        self.global_step = 0

        self.data = data
        self.train_dataset = data.train_dataset
        self.val_dataset = data.val_dataset
        self.test_dataset = data.test_dataset

        self.data_loader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=cfg.Train.num_workers,
                                      shuffle=cfg.Data.shuffle,
                                      drop_last=True)
        self.val_data_loader = DataLoader(self.val_dataset,
                                          batch_size=self.batch_size,
                                          num_workers=cfg.Train.num_workers,
                                          drop_last=True)
        self.test_data_loader = DataLoader(self.test_dataset,
                                          batch_size=1,
                                          num_workers=1,
                                          drop_last=False)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.scalar_log_gaps = cfg.Train.scalar_log_gap
        self.validation_log_gaps = cfg.Train.validation_log_gap
        self.test_log_gaps = cfg.Train.test_log_gap
        
    def train(self):
        for epoch in tqdm(range(self.num_epochs), desc='Epochs'):
            progress = tqdm(self.data_loader, desc="Batchs")

            for i_batch, batch in enumerate(progress):
                self.model.train()

                lr = self.schedule(init_lr=self.init_lr, global_step=self.global_step)
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
                
                self.optim.zero_grad()
                if self.global_step % self.scalar_log_gaps == 0 and self.global_step > 0:
                    self.writer.add_scalar("lr/lr", lr, self.global_step)
                
                x = batch["joints"].to(self.device)
                c = batch["control"]

                if self.global_step == 0:
                    with torch.no_grad():
                        _, loss = self.model(x)
                    self.global_step+=1
                    continue
                else:
                    _, loss = self.model(x)

                if self.global_step % self.scalar_log_gaps == 0:
                    self.writer.add_scalar("loss", loss, self.global_step)
                    
                self.model.zero_grad()
                self.optim.zero_grad()
                loss.backward()

                # operate grad
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if self.global_step % self.scalar_log_gaps == 0:
                        self.writer.add_scalar("grad_norm", grad_norm, self.global_step)

                self.optim.step()

                # save checkpoints
                if self.global_step % self.checkpoints_gap == 0 and self.global_step > 0:
                    state = {
                        "global_step": self.global_step,
                        # DataParallel wrap model in attr `module`.
                        "model": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                    }
                    _file_at_step = "save_{}k{}.pkg".format(int(self.global_step // 1000), int(self.global_step % 1000))
                    save_path = os.path.join(self.checkpoints_dir, _file_at_step)
                    torch.save(state, save_path)
                
                # validation
                if self.global_step % self.validation_log_gaps == 0:  
                    self.model.eval()   
                    loss_val = 0
                    n_batches = 0
                    for ii, val_batch in enumerate(self.val_data_loader):
                        x = val_batch["joints"].to(self.device)
                        c = val_batch["control"].to(self.device)
                        with torch.no_grad():
                            z_val, loss_val_batch = self.model(x)
                            loss_val += loss_val_batch
                            n_batches += 1
                            
                    loss_val /= n_batches      
                    self.writer.add_scalar("val_loss", loss_val, self.global_step)

                # test sample generation
                if self.global_step % self.test_log_gaps == 0:  
                    self.model.eval()
                    for ii, test_batch in enumerate(self.test_data_loader):
                        x = test_batch["joints"].to(self.device)[0]
                        c = test_batch["control"].to(self.device)[0]
                        
                        with torch.no_grad():
                            z_test, _ = self.model(x)
                            z_interpo = linear_interpolation(z_test, step=10)
                            x_reverse, _ = self.model(z_interpo)

                        if self.global_step == self.test_log_gaps:
                            clip = torch.cat((x, c.unsqueeze(2)), 2).cpu().numpy()
                            clip = clip.reshape(1, clip.shape[0], clip.shape[1]*clip.shape[2])
                            clip = inv_standardize(clip, self.data.scaler)
                            _clip_name = "test_{}.mp4".format(int(ii))
                            clip_path = os.path.join(self.plot_dir, _clip_name)
                            plot_animation(clip[0], self.data.parents, clip_path, self.data.frame_rate, axis_scale=60)               

                        clip = torch.cat((x_reverse, c.unsqueeze(2)), 2).cpu().numpy()
                        clip = clip.reshape(1, clip.shape[0], clip.shape[1]*clip.shape[2])
                        clip = inv_standardize(clip, self.data.scaler)
                        _clip_name = "test_{}_{}.mp4".format(int(self.global_step), int(ii))
                        clip_path = os.path.join(self.plot_dir, _clip_name)  
                        plot_animation(clip[0], self.data.parents, clip_path, self.data.frame_rate, axis_scale=60)
                        
                self.global_step += 1
                
        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "all_scalars.json"))
        self.writer.close()