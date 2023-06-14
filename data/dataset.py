import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms


class AutoencoderDataset(Dataset):
    def __init__(self, data, resize=None, normalize=True):
        self.data = data
        self.resize = resize
        self.normalize = normalize

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        camera = read_image(sample['CAMERA'])
        depth = read_image(sample['DEPTH'])
        semantic = read_image(sample['SEMANTIC'])

        if self.resize:
            camera = transforms.Resize(self.resize)(camera)
            depth = transforms.Resize(self.resize)(depth)
            semantic = transforms.Resize(self.resize, transforms.InterpolationMode.NEAREST)(semantic)

        if self.normalize:
            camera = camera / 255.0
            depth = depth / 255.0

        image = torch.cat((camera, depth), dim=0).to(torch.float32)

        semantic = semantic.to(torch.long).squeeze()

        data = torch.FloatTensor(sample['DATA'])

        junction = torch.FloatTensor([sample['JUNCTION']])

        return image, semantic, data, junction