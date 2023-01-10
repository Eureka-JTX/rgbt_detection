# Copyright (c) OpenMMLab. All rights reserved.
import imp
from .bbox_head import BBoxHead
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
from .dii_head import DIIHead
from .double_bbox_head import DoubleConvFCBBoxHead
from .sabl_head import SABLHead
from .scnet_bbox_head import SCNetBBoxHead
from .convfc_res_bbox_head import Shared2FCResBBoxHead
from .convfc_bbox_head_search import ConvFCBBoxHeadSearch
from .convfc_bbox_head_subnet import ConvFCBBoxHeadSubnet
from .convfc_bbox_head_subnet_v2 import ConvFCBBoxHeadSubnetV2
from .linear_head import LinearHead, LinearResHead

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'SABLHead', 'DIIHead',
    'SCNetBBoxHead', 'Shared2FCResBBoxHead', 'ConvFCBBoxHeadSearch', 'ConvFCBBoxHeadSubnet',
    'LinearHead', 'LinearResHead', 'ConvFCBBoxHeadSubnetV2'
]
