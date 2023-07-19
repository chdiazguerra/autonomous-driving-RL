import os
from argparse import ArgumentParser
import pickle

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.dataset import AutoencoderDataset
from models.autoencoder import Autoencoder, AutoencoderSEM

def main(args):
    os.makedirs(args.dirpath, exist_ok=True)

    img_size = [int(x) for x in args.img_size.split('x')] if args.img_size != 'default' else None

    # Load split
    if args.split:
        with open(args.split, 'rb') as f:
            split = pickle.load(f)
        train = split['train']
        val = split['val']
    
    else:
        with open(args.file, 'rb') as f:
            data = pickle.load(f)

        stratify = [d[0] for d in data]

        # Split data stratify by folder (different weather conditions)
        train, val = train_test_split(data, test_size=args.val_size, random_state=42,
                                    shuffle=True, stratify=stratify)
        
        #Save split
        with open(args.dirpath + '/split.pkl', 'wb') as f:
            pickle.dump({'train': train, 'val': val}, f)
    
    # Create datasets
    train_dataset = AutoencoderDataset(train, img_size, args.no_norm, args.low_sem)
    val_dataset = AutoencoderDataset(val, img_size, args.no_norm, args.low_sem)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=False)

    # Create model
    img_size = tuple(train_dataset[0][0].shape[1:]) if img_size is None else img_size
    num_classes = 14 if args.low_sem else 29
    if args.sem:
        model = AutoencoderSEM(img_size, args.emb_size, num_classes, args.lr)
    else:
        model = Autoencoder(img_size, args.emb_size, num_classes, args.lr)

    # Train model
    trainer = pl.Trainer(callbacks=[pl.callbacks.ModelCheckpoint(dirpath=args.dirpath, monitor='val_loss', save_top_k=1),
                                    pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
                                    pl.callbacks.ModelCheckpoint(dirpath=args.dirpath, filename="{epoch}")],
                         logger=args.no_logger,
                         accelerator=args.device,
                         max_epochs=args.epochs,
                         default_root_dir=args.dirpath)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.pretrained)
    

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, default='dataset.pkl', help='dataset file')
    parser.add_argument('--val_size', type=float, default=0.2, help='validation size')
    parser.add_argument('--img_size', type=str, default='default', help='image size')
    parser.add_argument('-no_norm', action='store_false', help='Not normalize input image')
    parser.add_argument('--emb_size', type=int, default=256, help='embedding size')
    parser.add_argument('-low_sem', action='store_true', help='Use low resolution semantic segmentation (14 classes)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--device', type=str, default='auto', help='device', choices=['auto', 'gpu', 'cpu'])
    parser.add_argument('--dirpath', type=str, default='./bestModel', help='directory path to save the model')
    parser.add_argument('-no_logger', action='store_false', help='Not use logger')
    parser.add_argument('-sem', action='store_true', help='Use Autoencoder with cross entropy loss for semantic segmentation')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model')
    parser.add_argument('--split', type=str, default=None, help='split file')
    args = parser.parse_args()

    main(args)