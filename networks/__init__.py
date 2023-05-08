from .deeplabv3_mb import *
from .deeplabv3_resnet import *

def get_network(backbone = 'mbv2_deeplab',
                rep_size = 0,
                num_classes=11,
                text_features = None,
                ):
    
    if backbone == 'mbv2_deeplab':
        model = DeepLabv3Plus_MB(backbone = backbone,
                                 rep_size = rep_size,
                                 num_classes = num_classes
                                 )
    elif backbone == 'r50_deeplab' or backbone == 'r101_deeplab' or backbone == 'r50d_deeplab' or backbone =='r101d_deeplab':
        model = DeepLabv3Plus_RN(backbone = backbone,
                                 rep_size = rep_size,
                                 num_classes = num_classes
                                 )
    else:
        raise NotImplementedError("Backbone not supported at this time")
        
    return model

