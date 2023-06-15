from argparse import ArgumentParser
import pickle

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.dataset import AutoencoderDataset
from models.autoencoder import Autoencoder, AutoencoderSEM

def main(args):
    img_size = [int(x) for x in args.img_size.split('x')] if args.img_size != 'default' else None

    # Load data
    with open(args.file, 'rb') as f:
        data = pickle.load(f)

    # Split data stratify by folder (different weather conditions)
    train, val = train_test_split(data, test_size=args.val_size, random_state=42,
                                   shuffle=True, stratify=[d['IDX'] for d in data])
    
    # Create datasets
    train_dataset = AutoencoderDataset(train, img_size, args.no_norm)
    val_dataset = AutoencoderDataset(val, img_size, args.no_norm)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=False)

    # Create model
    img_size = tuple(train_dataset[0][0].shape[1:]) if img_size is None else img_size
    if args.sem:
        model = AutoencoderSEM(img_size, args.emb_size, args.num_classes, args.lr)
    else:
        model = Autoencoder(img_size, args.emb_size, args.num_classes, args.lr)

    # Train model
    trainer = pl.Trainer(callbacks=[pl.callbacks.ModelCheckpoint(dirpath=args.dirpath, monitor='val_loss', save_top_k=1)],
                         logger=args.no_logger,
                         accelerator=args.device,
                         max_epochs=args.epochs,
                         default_root_dir=args.dirpath)
    trainer.fit(model, train_loader, val_loader)
    

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, default='dataset.pkl', help='dataset file')
    parser.add_argument('--val_size', type=float, default=0.2, help='validation size')
    parser.add_argument('--img_size', type=str, default='default', help='image size')
    parser.add_argument('-no_norm', action='store_false', help='Not normalize input image')
    parser.add_argument('--emb_size', type=int, default=256, help='embedding size')
    parser.add_argument('--num_classes', type=int, default=29, help='number of semantic classes')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--device', type=str, default='auto', help='device', choices=['auto', 'gpu', 'cpu'])
    parser.add_argument('--dirpath', type=str, default='./bestModel', help='directory path to save the model')
    parser.add_argument('-no_logger', action='store_false', help='Not use logger')
    parser.add_argument('-sem', action='store_true', help='Use Autoencoder with cross entropy loss for semantic segmentation')
    args = parser.parse_args()

    main(args)