from enum import Enum
from typing import Optional, List

from pydantic import BaseModel


class DeviceEnum(str, Enum):
    cuda = "cuda"
    cpu = "cpu"


class ImageRestoreFace(BaseModel):
    input_images: List[str]
    fidelity_weight: Optional[float] = 0.5
    upscale: Optional[int] = 2
    has_aligned: Optional[bool] = False
    only_center_face: Optional[bool] = False
    draw_box: Optional[bool] = False
    detection_model: Optional[str] = "retinaface_resnet50"
    bg_upsampler: Optional[str] = None
    face_upsampler: Optional[bool] = False
    bg_tile: Optional[int] = 400
    device: Optional[DeviceEnum] = DeviceEnum.cuda
