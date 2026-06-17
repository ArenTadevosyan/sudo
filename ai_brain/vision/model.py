from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


# Anchors are expressed in grid-cell units (YOLOv2 style), tuned for a 13x13 grid.
DEFAULT_ANCHORS: list[tuple[float, float]] = [
    (1.3221, 1.73145),
    (3.19275, 4.00944),
    (5.05587, 8.09892),
    (9.47112, 4.84053),
    (11.2364, 10.0071),
]


@dataclass
class VisionConfig:
    num_classes: int
    img_size: int = 416
    anchors: list[tuple[float, float]] | None = None
    class_names: list[str] | None = None

    def __post_init__(self) -> None:
        if self.img_size % 32 != 0:
            raise ValueError("img_size must be a multiple of 32")
        if self.anchors is None:
            self.anchors = [tuple(a) for a in DEFAULT_ANCHORS]
        else:
            self.anchors = [tuple(a) for a in self.anchors]
        if self.class_names is None:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]

    @property
    def grid_size(self) -> int:
        return self.img_size // 32

    @property
    def num_anchors(self) -> int:
        return len(self.anchors)


def save_config(path: Path, config: VisionConfig) -> None:
    Path(path).write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_config(path: Path) -> VisionConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return VisionConfig(**data)


def build_model(config: VisionConfig):
    import torch.nn as nn

    def conv(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        padding = kernel // 2
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    class Detector(nn.Module):
        """Compact Darknet-style backbone with a single detection head.

        Downsamples the input by 32x, so a 416px image yields a 13x13 grid.
        Every grid cell predicts ``num_anchors`` boxes, each carrying
        ``5 + num_classes`` numbers: tx, ty, tw, th, objectness, class scores.
        """

        def __init__(self) -> None:
            super().__init__()
            num_outputs = config.num_anchors * (5 + config.num_classes)
            self.features = nn.Sequential(
                conv(3, 32),                 # 416
                conv(32, 64, stride=2),      # 208
                conv(64, 128, stride=2),     # 104
                conv(128, 64, kernel=1),
                conv(64, 128),
                conv(128, 256, stride=2),    # 52
                conv(256, 128, kernel=1),
                conv(128, 256),
                conv(256, 512, stride=2),    # 26
                conv(512, 256, kernel=1),
                conv(256, 512),
                conv(512, 1024, stride=2),   # 13
                conv(1024, 512, kernel=1),
                conv(512, 1024),
                conv(1024, 1024),
            )
            self.head = nn.Conv2d(1024, num_outputs, 1)

        def forward(self, x):
            batch = x.size(0)
            x = self.features(x)
            x = self.head(x)
            grid = x.size(-1)
            # (B, A*(5+C), G, G) -> (B, A, G, G, 5+C)
            x = x.view(batch, config.num_anchors, 5 + config.num_classes, grid, grid)
            return x.permute(0, 1, 3, 4, 2).contiguous()

    return Detector()
