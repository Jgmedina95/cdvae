from __future__ import annotations

try:
    import lightning.pytorch as pl
    from lightning.pytorch import Callback, seed_everything
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import WandbLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning import Callback, seed_everything
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from pytorch_lightning.loggers import WandbLogger

try:
    from torch_geometric.loader import DataLoader as PYGDataLoader
except ImportError:
    from torch_geometric.data import DataLoader as PYGDataLoader
