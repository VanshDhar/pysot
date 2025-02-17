from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.models.temporal.TempNET import TemporalNet

TEMPORAL = {
    'TemporalNET' : TemporalNet
}

def get_temporal_head(name):
    return TEMPORAL[name]()