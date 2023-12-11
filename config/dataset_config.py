from dataclasses import dataclass


@dataclass()
class default_dataset:
    path: str = 'user defined'
    train_split: str = "train"
    test_split: str = "test"
