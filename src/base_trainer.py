'''
    The main idea behind using this bastract class is to keep book-keeping stuff away from core logic of training.
    This class provides all basic methods for saving, loading and logging. We can inherit from this class.
'''
import time
import json
import os
import torch
from utility import *
from abc import ABC, abstractmethod

# Using ABC for making this class somewhat Abstarct, Python3 doesnot offer concrete Abstact classes 
class BaseTrainer(ABC):

    def __init__(self):
        self.ckpt_queue = []
        self.num_ckpts = 3

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """
            Convert all models to testing/evaluation mode
        """
        for name, m in self.models.items():
            m.eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opts.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opts.save_frequency == 0:
                self.save_model()
    
    def run_epoch(self):
        raise NotImplementedError

    def process_batch(self, inputs):
        raise NotImplementedError
    
    def compute_losses(self, inputs, outputs):
        raise NotImplementedError    

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opts.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
    
    def log(self, mode, inputs, outputs, losses):
        raise NotImplementedError


    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opts.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, is_chkpt=False, meta_dict=None):
        """Save model weights to disk
        """
        if is_chkpt:
            save_folder = os.path.join(self.log_path, "models", f"chkpts")
        else:
            save_folder = os.path.join(
                self.log_path, "models", "weights_{}".format(self.epoch))

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        
        ckpt_names = []
        for model_name, model in self.models.items():
            if is_chkpt:
                ckpt_names.append(save_path) # saving model names to push on to checkpoint queue
            else:
                save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()

            # save the meta variables along with weights,  e.g. sizes
            if meta_dict is not None:
                for k in list(meta_dict):
                    to_save[k] = meta_dict[k]
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        if is_chkpt:
            ckpt_names.append(save_path)
        torch.save(self.model_optimizer.state_dict(), save_path)

        if is_chkpt:
            # Add the checkpoint paths to queue
            self.ckpt_queue.append(ckpt_names)

            # If queue is getting overfilled, pop the element at the head of queue, and delete it
            if len(self.ckpt_queue) > self.num_ckpts:
                first_ckpt_names = self.ckpt_queue.pop(0)
                for path in first_ckpt_names:
                    os.remove(path) 
    
    def load_model(self):
        self.opts.load_weights_folder = os.path.expanduser(
            self.opts.load_weights_folder)

        assert os.path.isdir(self.opts.load_weights_folder), \
            "Cannot find folder {}".format(self.opts.load_weights_folder)
        print("loading model from folder {}".format(
            self.opts.load_weights_folder))

        for n in self.opts.models_to_load:
            if n in self.models.keys():
                print("Loading {} weights...".format(n))
                path = os.path.join(
                    self.opts.load_weights_folder, "{}.pth".format(n))
                pretrained_dict = torch.load(path)
                model_dict = self.models[n].state_dict()
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict,strict=False)

        # loading adam state
        optimizer_load_path = os.path.join(
            self.opts.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
    
    def load_infer_model(self):
        self.opts.load_weights_folder = os.path.expanduser(
            self.opts.load_weights_folder)

        assert os.path.isdir(self.opts.load_weights_folder), \
            "Cannot find folder {}".format(self.opts.load_weights_folder)
        print("loading model from folder {}".format(
            self.opts.load_weights_folder))

        for n in self.opts.models_to_load:
            if n in self.models.keys():
                print("Loading {} weights...".format(n))
                path = os.path.join(
                    self.opts.load_weights_folder, "{}.pth".format(n))
                pretrained_dict = torch.load(path)
                model_dict = self.models[n].state_dict()
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict,strict=False)
