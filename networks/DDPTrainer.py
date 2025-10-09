from .Trainer import Trainer

class DDPTrainer(Trainer):
    """
    Extension of Trainer class to train Denoising Diffusion Probabilistic Models.
    """
    def __init__(self, **kwargs):
        """
        To know args and variables, please refers to '~.Trainer.__init__'
        """
        super(DDPTrainer, self).__init__(**kwargs)

    #Override train function to make it works for ddpm
    def train(self):
        """
        Refers to '~.Trainer.train'
        """
        pass