from __future__ import absolute_import, division, print_function

from inference_sota import Inference as Inference_SOTA
from inference_ours import Inference as Inference_Ours
from options import StereoPoseOptions

options = StereoPoseOptions()
opts = options.parse()


if __name__ == "__main__":
    if opts.eval_sota:
        trainer = Inference_SOTA(opts)
    else:
        trainer = Inference_Ours(opts)
        
    trainer.test()