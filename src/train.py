from options import StereoPoseOptions
from trainer import Experiment

options = StereoPoseOptions()
opts = options.parse()

if __name__ == "__main__":
    exp = Experiment(opts)

    if opts.eval_mode:
        exp.infer()
    else:
        exp.train()
