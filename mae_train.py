import torch
from torch import Tensor
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from MAE import SceneMAE
from NViST.dataLoader.ray_dataset import MVImgNetNeRF
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import LearningRateMonitor
import torchvision.transforms as transforms

class SceneMaskAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        epochs = 500,
        lr = 1.5e-4,
        lr_min = 1e-5,
        warmup_epochs = 0,
        weight_decay = 0.05,
        betas = (0.9, 0.95),
        normalize_target = True,
        model_parameters = None,
        fixed_train = None,
        fixed_val = None
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # self.logger: TensorBoardLogger
        self.model = SceneMAE(**model_parameters) if model_parameters != None else SceneMAE()

        self.epochs = epochs
        self.lr = lr
        self.lr_min = lr_min
        self.warmup_epochs = warmup_epochs
        self.betas = betas
        self.weight_decay = weight_decay

        self.normalize_target = normalize_target
        self.fixed_train = fixed_train
        self.fixed_val = fixed_val

    def forward(self, imgs: Tensor, ray_origins: Tensor, ray_directions: Tensor) -> Tensor:
        return self.model(imgs, ray_origins, ray_directions)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.lr, betas = self.betas, weight_decay = self.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs = self.warmup_epochs, max_epochs = self.epochs, eta_min = self.lr_min)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]               
    
    def training_step(self, train_batch, batch_idx) -> Tensor:
        opt = self.optimizers()

        imgs = train_batch['images'].flatten(start_dim = 0, end_dim = 1)
        ray_origins = train_batch['ray_origins'].flatten(start_dim = 0, end_dim = 1)
        ray_directions = train_batch['ray_directions'].flatten(start_dim = 0, end_dim = 1)
        
        # Assertions
        if imgs.shape == ray_origins.shape == ray_directions:
            raise UserWarning(
                f"Input image, ray origins and ray directions should have the same size Bx3xHxW."
            )

        pred, mask = self.forward(imgs, ray_origins, ray_directions)
        target = self.model.patchify(imgs)

        loss = self.compute_loss(pred, target, mask)

        self.log(f"loss_epoch", loss.detach().cpu(), on_step = False, on_epoch = True)

        return loss

    def compute_loss(self, pred: Tensor, target: Tensor, mask: Tensor):
        if self.normalize_target == True:
            mean = target.mean(dim = -1, keepdim = True)
            var = target.var(dim = -1, keepdim = True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim = -1)

        loss = (loss * mask).sum() / mask.sum()
        self.log(f"loss_step", loss.detach().cpu())

        return loss

    def validation_step(self, batch, batch_idx):
        imgs = batch['images'].flatten(start_dim = 0, end_dim = 1)
        ray_origins = batch['ray_origins'].flatten(start_dim = 0, end_dim = 1)
        ray_directions = batch['ray_directions'].flatten(start_dim = 0, end_dim = 1)

        # Assertions
        if imgs.shape == ray_origins.shape == ray_directions:
            raise UserWarning(
                f"Input image, ray origins and ray directions should have the same size Bx3xHxW."
            )

        with torch.no_grad():
            pred, mask = self.forward(imgs, ray_origins, ray_directions)

        target = self.model.patchify(imgs)
        val_loss = self.compute_loss(pred, target, mask)
        
        self.log(f"val_loss", val_loss.detach().cpu(), on_epoch=True)

    def log_reconstruct_img(self, tag):
        imgs = self.fixed_train['images'].to('cuda')       
        ray_origins = self.fixed_train['ray_origins'].to('cuda')
        ray_directions = self.fixed_train['ray_directions'].to('cuda')
        
        with torch.no_grad():
            pred, mask = self.forward(imgs, ray_origins, ray_directions)
        
        patches = self.model.patchify(imgs)
        mask = mask.unsqueeze(-1).expand(-1, -1, 480).bool()
        recon = patches * ~mask + pred * mask
        recon = self.model.unpatchify(recon)
        pred = self.model.unpatchify(pred)

        recon = self.unnormalize_imgs(recon).cpu().permute(0, 2, 3, 1).numpy().reshape(2, 2, 160, 90, 3).transpose(0, 2, 1, 3, 4).reshape(320, 180, 3)
        pred = self.unnormalize_imgs(pred).cpu().permute(0, 2, 3, 1).numpy().reshape(2, 2, 160, 90, 3).transpose(0, 2, 1, 3, 4).reshape(320, 180, 3)
        img = self.unnormalize_imgs(imgs).cpu().permute(0, 2, 3, 1).numpy().reshape(2, 2, 160, 90, 3).transpose(0, 2, 1, 3, 4).reshape(320, 180, 3)
        
        tensorboard = self.logger.experiment
        tensorboard.add_image(tag + '_img', img, self.current_epoch, dataformats='HWC')
        tensorboard.add_image(tag + '_pred', pred, self.current_epoch, dataformats='HWC')
        tensorboard.add_image(tag + '_mix', recon, self.current_epoch, dataformats='HWC')

    def unnormalize_imgs(self, imgs: Tensor) -> Tensor:
        mean = torch.tensor([0.5,0.5,0.5])
        std = torch.tensor([0.5,0.5,0.5])

        return imgs * std.reshape(1,3,1,1).type_as(imgs) + mean.reshape(1,3,1,1).type_as(imgs)
    
    def on_train_epoch_end(self):
        if self.fixed_train is not None:
            self.log_reconstruct_img('train')
    
    def on_val_epoch_end(self):
        if self.fixed_val is not None:
            self.log_reconstruct_img('val')

if __name__ == "__main__":
    import os
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description="Description of the argument parser.")

    # Training Epochs
    parser.add_argument("-e", "--epochs", dest = "epochs", type = int, help = "Epochs for trainings", default = 800)
    # Batch Size
    parser.add_argument("-b", "--batch-size", dest = "batch_size", type = int, help = "Batch size per iteration", default = 256)
    # Dataset
    parser.add_argument(
        "-d",
        "--dataset",
        dest = "dataset_path",
        type = str,
        default = "./data/",
        help = "Dataset path",
    )
    # Accelerator
    parser.add_argument("-acc", "--accelerator", dest = "accelerator", type = str, default = "gpu", help = "Pytroch Lightning accelerator")
    # GPUs
    parser.add_argument(
        "-g",
        "--gpus",
        dest = "gpus",
        type = int,
        nargs = "+",
        default = [0],
        help = "comma separated list of cuda device (e.g. GPUs) to be used",
    )
    # Seed
    parser.add_argument("-s", "--seed", dest = "seed", type = int, default = 2024, help = "Seed for dataset splitting")
    parser.add_argument("--num_workers", dest = "num_workers", type = int, default = 8, help = "Number of threads to load data")
    parser.add_argument("--name", dest = "name", type = str, help = "Name of the worker", default = 'SceneMAE')
    parser.add_argument("-cp", "--ckpt-path", dest="ckpt_path", type=str, default=None, help="Checkpoint path")
    parser.add_argument("-v", "--verbose", dest = "verbose", type = bool, help = "Show progress bar or not", default = True)
    
    #### Set Parameters / Initialize ###
    args = parser.parse_args()  # Get commandline arguments
    print(args)
    
    name = args.name
    epochs = args.epochs
    batch_size = args.batch_size
    dataset_path = args.dataset_path
    accelerator = args.accelerator
    gpus = args.gpus  # Or list of GPU-Ids; 0 is CPU
    print(f"Running on {accelerator}:")
    for gpu in gpus:
        print(f"\t[{gpu}]: {torch.cuda.get_device_name(gpu)}")
    seed = args.seed

    if args.ckpt_path == "None":
        ckpt_path = None
    else:
        ckpt_path = args.ckpt_path

    print(f"Seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    # Get dataset and split it
    if not os.path.exists(dataset_path):
        raise UserWarning(f'Dataset path "{dataset_path}" does not exist!!')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop([160, 90], scale=(0.2, 1.0), interpolation=3),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = MVImgNetNeRF(dataset_path, split = 'train', number_of_imgs_from_the_same_scene = 4, patch_hw = [16,10], transform = transform)
    validation_dataset = MVImgNetNeRF(dataset_path, split = 'test', number_of_imgs_from_the_same_scene = 4, patch_hw = [16,10], transform = transform)

    fixed_train = train_dataset.__getitem__(0)
    fixed_val = validation_dataset.__getitem__(0)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = args.num_workers, pin_memory = True)
    validation_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = False, num_workers = args.num_workers, pin_memory = True)

    f = open('model_parameters.yaml','rb')
    model_parameters = yaml.load(f, Loader = yaml.FullLoader)

    lighting_module = SceneMaskAutoEncoder(epochs = epochs, model_parameters = model_parameters, fixed_train = fixed_train, fixed_val = fixed_val)
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    # Define Last and best Checkpoints to be saved.
    checkpoint_callback = ModelCheckpoint(
        # dirpath = './saved_models',
        filename="{epoch}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        every_n_epochs = 1,
        save_top_k = 5,
        save_last = True
    )

    # Create TensorBoardLogger
    logger = TensorBoardLogger("lightning_logs", name = name, default_hp_metric = False)
    print("######################################")
    print("experiment_name:", name)
    print("######################################")

    lr_monitor = LearningRateMonitor(logging_interval = 'step')

    # Setup Trainer
    trainer = pl.Trainer(
        accelerator = accelerator,
        devices = gpus,
        num_nodes=1,
        # limit_train_batches=limit_train_batches,
        # limit_val_batches=5,
        max_epochs = epochs,  # Stopping epoch
        logger = logger,
        callbacks = [checkpoint_callback, lr_monitor],  # You may add here additional call back that saves the best model
        # limit_train_batches=10,
        # detect_anomaly=True,
        strategy = ("ddp_find_unused_parameters_true" if len(gpus) > 1 else "auto"),  # for distributed compatibility
        log_every_n_steps = 500,
        enable_progress_bar = args.verbose,
        precision = 16
    )

    # Fit/train model
    if ckpt_path != None:  # try to continue training
        # ckpt_path = Path(ckpt_path)
        if not os.path.exists(ckpt_path):
            raise UserWarning(f'Checkpoint path "{ckpt_path}" does not exist!!')
        print(f"Try to resume from {ckpt_path}")
        trainer.fit(lighting_module, train_loader, validation_loader, ckpt_path = ckpt_path)
    else:  # start training anew
        trainer.fit(lighting_module, train_loader, validation_loader)
