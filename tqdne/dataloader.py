from torch.utils.data import DataLoader

from tqdne.dataset import Dataset


def get_train_and_val_loader(config, num_workers, batchsize, cond=False):
    train_dataset = Dataset(
        config.datapath, config.representation, cut=config.t, cond=cond, split="train_validation"
    )
    val_dataset = Dataset(
        config.datapath, config.representation, cut=config.t, cond=cond, split="test"
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        #prefetch_factor=1,
        #persistent_workers=True,
        num_workers=num_workers,
        batch_size=batchsize,
    )
    val_loader = DataLoader(
        val_dataset,
        drop_last=False,
        #prefetch_factor=1,
        #persistent_workers=True,
        num_workers=num_workers,
        batch_size=batchsize,
    )
    return train_loader, val_loader
