import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection

class SymbolDataset(Dataset):
    """Custom Dataset for loading symbol images with their labels."""
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (str): Root directory of the dataset
            transform (callable, optional): Optional transform to be applied on images
            split (str): Which dataset split to use ('train', 'valid', or 'test')
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_dir = os.path.join(self.root_dir, 'images')
        self.labels_dir = os.path.join(self.root_dir, 'labels')
        
        # Get all image files
        self.image_files = [f for f in os.listdir(self.image_dir) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Load all labels
        self.labels = []
        for img_file in self.image_files:
            label_file = os.path.join(self.labels_dir, 
                                    os.path.splitext(img_file)[0] + '.txt')
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    # Get only the class ID (first number in the file)
                    class_id = int(f.readline().split()[0])
                    self.labels.append(class_id)
            else:
                print(f"Warning: No label file found for {img_file}")
                self.labels.append(-1)  # Use -1 for missing labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Returns the image and its label at the given index."""
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor()
        ])
        return data_transforms

    def get_dataset(self, name, n_views, split):
        """
        Returns the dataset with the given name and number of views.
        
        Args:
            name (str): Name of the dataset ('cifar10', 'stl10', or 'symbols')
            n_views (int): Number of augmented views to generate for each image
            split (str): Train, Test, or Validate
        """
        valid_datasets = {
            # 'cifar10': lambda: datasets.CIFAR10(
            #     self.root_folder, 
            #     train=True,
            #     transform=ContrastiveLearningViewGenerator(
            #         self.get_simclr_pipeline_transform(32),
            #         n_views
            #     ),
            #     download=True
            # ),
            # 'stl10': lambda: datasets.STL10(
            #     self.root_folder, 
            #     split='unlabeled',
            #     transform=ContrastiveLearningViewGenerator(
            #         self.get_simclr_pipeline_transform(96),
            #         n_views
            #     ),
            #     download=True
            # ),
            'symbols': lambda: SymbolDataset(
                self.root_folder,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(64),  # Adjust size as needed
                    n_views
                ),
                split=split  # Default to training split
            )
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()