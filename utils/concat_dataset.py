from tqdm import tqdm
from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    def __init__(self, dataset, wrap_size=4096):
        self.dataset = dataset
        self.wrap_size = wrap_size
        self.dataset_package = list()
        tem_buffer = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for item in tqdm(dataset, desc='Processing dataset', dynamic_ncols=True):
            tem_buffer['input_ids'] += item['input_ids']
            tem_buffer['attention_mask'] += item['attention_mask']
            tem_buffer['labels'] += item['labels']
            while len(tem_buffer['input_ids']) > self.wrap_size:
                package = {key: value[:self.wrap_size] for key, value in tem_buffer.items()}
                tem_buffer = {key: value[self.wrap_size:] for key, value in tem_buffer.items()}
                self.dataset_package.append(package)

    def __getitem__(self, item):
        return self.dataset_package[item]

    def __len__(self):
        return len(self.dataset_package)
