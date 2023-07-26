import os
from argparse import ArgumentParser
import pickle

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data.dataset import AutoencoderDataset
from models.autoencoder import Autoencoder, AutoencoderSEM
from models.vae import VAE
from configuration.config import *

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

    
    normalize_output = False if args.model == 'AutoencoderSEM' else True
    # Create datasets
    train_dataset = AutoencoderDataset(train[:100], img_size, args.norm_input, args.low_sem, args.use_img_out, normalize_output)
    val_dataset = AutoencoderDataset(val[:100], img_size, args.norm_input, args.low_sem, args.use_img_out, normalize_output)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=False)

    # Create model
    img_size = tuple(train_dataset[0][0].shape[1:]) if img_size is None else img_size
    num_classes = 14 if args.low_sem else 29
    
    if args.model == 'Autoencoder':
        model = Autoencoder(input_size=img_size, emb_size=args.emb_size, lr=args.lr, weights=(0.8, 0.1, 0.1),
                            use_additional_data=args.additional_data, out_ch=4 if args.use_img_out else 1)
    elif args.model == 'AutoencoderSEM':
        model = AutoencoderSEM(input_size=img_size, emb_size=args.emb_size, lr=args.lr, weights=(0.8, 0.1, 0.1),
                            use_additional_data=args.additional_data, num_classes=num_classes)
    elif args.model == 'VAE':
        model = VAE(input_size=img_size, emb_size=args.emb_size, lr=args.lr,
                            out_ch=4 if args.use_img_out else 1)
    else:
        raise ValueError('Model not found')

    # Train model
    trainer = pl.Trainer(callbacks=[pl.callbacks.ModelCheckpoint(dirpath=args.dirpath, monitor='val_loss', save_top_k=1),
                                    pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
                                    pl.callbacks.ModelCheckpoint(dirpath=args.dirpath, filename="{epoch}")],
                         accelerator=args.device,
                         max_epochs=args.epochs,
                         default_root_dir=args.dirpath)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.pretrained)
    

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, default=AE_DATASET_FILE, help='dataset file')
    parser.add_argument('--val_size', type=float, default=AE_VAL_SIZE, help='validation size')
    parser.add_argument('--img_size', type=str, default=AE_IMG_SIZE, help='image size')

    parser.add_argument('--norm_input', type=bool, default=AE_NORM_INPUT, help='Normalize input image')
    parser.add_argument('--emb_size', type=int, default=AE_EMB_SIZE, help='embedding size')
    parser.add_argument('--batch_size', type=int, default=AE_BATCH_SIZE, help='batch size')
    parser.add_argument('--epochs', type=int, default=AE_EPOCHS, help='number of epochs')
    parser.add_argument('--lr', type=float, default=AE_LR, help='learning rate')
    parser.add_argument('--device', type=str, default='auto', help='device', choices=['auto', 'gpu', 'cpu'])
    parser.add_argument('--dirpath', type=str, default=AE_DIRPATH, help='directory path to save the model')

    parser.add_argument('--low_sem', type=bool, default=AE_LOW_SEM, help='Use low resolution semantic segmentation (14 classes)')
    parser.add_argument('--use_img_out', type=bool, default=AE_USE_IMG_AS_OUTPUT, help='Use image as output (not used if model is VAE)')
    parser.add_argument('--model', type=str, default=AE_MODEL, help='model', choices=['Autoencoder', 'AutoencoderSEM', 'VAE'])
    parser.add_argument('--additional_data', type=bool, default=AE_ADDITIONAL_DATA, help='Use additional data (not used if model is VAE)')

    parser.add_argument('--pretrained', type=str, default=AE_PRETRAINED, help='pretrained model')
    parser.add_argument('--split', type=str, default=AE_SPLIT, help='split file')
    args = parser.parse_args()

    main(args)