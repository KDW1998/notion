# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .kcqr_deterio import KCQRDeterioDataset


@DATASETS.register_module()
class KCQRCrackDataset(KCQRDeterioDataset):
    """Crack dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    
    Note that palette is RGB.
    """
    METAINFO = dict(classes=('background', 'crack'),
                    palette=[[0, 0, 0], [255, 0, 0]])
