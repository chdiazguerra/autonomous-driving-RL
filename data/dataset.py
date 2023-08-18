import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from .utils import low_resolution_semantics
   
class AutoencoderDataset(Dataset):
    def __init__(self, data, resize=None, normalize=True, low_sem=True,
                 use_img_as_output=False, normalize_output=False):
        self.images = []
        self.semantics = []
        self.data = []
        self.junctions = []

        i = 0
        for _, rgb, depth_image, semantic_image, additional, junction in data:
            if i%200==0:
                print("Loading data: ", i)
            image = torch.cat((read_image(rgb), read_image(depth_image)), dim=0)
            semantic = read_image(semantic_image)
            if resize:
                image = transforms.Resize(resize)(image)
                semantic = transforms.Resize(resize, transforms.InterpolationMode.NEAREST)(semantic)
            if low_sem:
                low_resolution_semantics(semantic)
            semantic = semantic.squeeze()
            data = torch.FloatTensor(additional)
            self.images.append(image)
            self.semantics.append(semantic)
            self.data.append(data)
            self.junctions.append(junction)
            i += 1

        self.normalize = normalize
        self.use_img_as_output = use_img_as_output
        self.normalize_output = normalize_output
        self.low_sem = low_sem

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].to(torch.float32)
        if self.use_img_as_output:
            output = torch.clone(image)
            if self.normalize_output:
                output /= 255.0
        else:
            output = self.semantics[idx].to(torch.long)
            if self.normalize_output:
                output = output.to(torch.float32)
                output = output / 13. if self.low_sem else output / 28.
        data = self.data[idx]
        junction = torch.FloatTensor([self.junctions[idx]])

        if self.normalize:
            image /= 255.0

        return image, output, data, junction