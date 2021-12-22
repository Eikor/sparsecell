from torch.utils.data import Dataset
import torch

class genericDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.images = None
        self.labels = None
        self.annotations = None

    def __len__(self):
        return len(self.images) 

    def annotation_to_label(self, annotation):
        label = annotation
        return label

    def label_to_annotation(self, label):
        annotation = label
        return annotation

    def update_annotation(self, prediction):
        new_annotation = self.label_to_annotation(prediction)
        self.annotations = new_annotation

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return {
            'image': image, # C*H*W
            'label': label # C*H*W
        }     