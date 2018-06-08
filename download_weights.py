"""Download the pretrained weights of the networks"""

import os
import sys

import config

def download_weights():
    if sys.version_info >= (3, 0):
        import urllib.request as urltool
    else:
        import urllib as urltool

    code_dir = config.code_root

    # ROI-net
    print("Downloading pretrained ROI-net")
    roi_net_source = 'http://www-sop.inria.fr/members/Qiao.Zheng/CardiacSegmentationPropagation/ROI/model_roi_net_epoch050.h5'
    roi_net_destination = os.path.join(code_dir, 'ROI', 'model_roi_net_epoch050.h5')
    urltool.urlretrieve(roi_net_source, roi_net_destination)

    # LVRV-net
    print("Downloading pretrained LVRV-net")
    lvrv_net_source = 'http://www-sop.inria.fr/members/Qiao.Zheng/CardiacSegmentationPropagation/LVRV_Segmentation/model_lvrv_net_epoch080.h5'
    lvrv_net_destination = os.path.join(code_dir, 'LVRV_Segmentation', 'model_lvrv_net_epoch080.h5')
    urltool.urlretrieve(lvrv_net_source, lvrv_net_destination)

    # LV-net
    print("Downloading pretrained LV-net")
    lv_net_source = 'http://www-sop.inria.fr/members/Qiao.Zheng/CardiacSegmentationPropagation/LV_Segmentation/model_lv_net_epoch080.h5'
    lv_net_destination = os.path.join(code_dir, 'LV_Segmentation', 'model_lv_net_epoch080.h5')
    urltool.urlretrieve(lv_net_source, lv_net_destination)




if __name__ == '__main__':
    download_weights()
