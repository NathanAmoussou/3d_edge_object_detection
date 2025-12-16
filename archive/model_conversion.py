import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

import torchvision
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteHead

class RawSSDLite(nn.Module):
    """
    Wrap pour exporter des sorties brutes (bbox_regression, cls_logits),
    sans post-process / NMS.
    Input: uint8 NCHW [0..255] pour être facile à alimenter côté OAK.
    """
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, x_u8):
        x = x_u8.to(torch.float32) / 255.0  # simple scaling
        feats = self.backbone(x)
        if isinstance(feats, torch.Tensor):
            feats = OrderedDict([("0", feats)])
        feats = list(feats.values())
        out = self.head(feats)
        # Selon versions, out peut être dict ou tuple
        if isinstance(out, dict):
            return out["bbox_regression"], out["cls_logits"]
        return out

def build_model(num_classes=2):
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)

    size = (320, 320)
    out_channels = det_utils.retrieve_out_channels(model.backbone, size)
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    model.head = SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer)
    return model

def main():
    ckpt_path = "cube_ssdlite.pth"
    onnx_path = "cube_raw.onnx"

    model = build_model(num_classes=2)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapped = RawSSDLite(model).eval()

    dummy = torch.zeros(1, 3, 320, 320, dtype=torch.uint8)  # input uint8
    torch.onnx.export(
        wrapped,
        dummy,
        onnx_path,
        opset_version=11,
        input_names=["input_u8"],
        output_names=["bbox_regression", "cls_logits"],
        do_constant_folding=True,
    )
    print("Wrote:", onnx_path)

if __name__ == "__main__":
    main()
