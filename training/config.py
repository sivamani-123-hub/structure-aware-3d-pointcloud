from dataclasses import dataclass

@dataclass
class TrainConfig:
    dataset_root: str = r"c:\DLproj\dataset\ModelNet40"
    num_classes: int = 40
    num_points: int = 1024
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    val_split_ratio: float = 0.1
    use_tnet: bool = False
    reg_weight: float = 0.001
    dropout: float = 0.3
    seed: int = 42
    checkpoint_dir: str = r"c:\DLproj\checkpoints"
    vis_dir: str = r"c:\DLproj\visualizations"
