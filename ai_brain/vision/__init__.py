"""Vision layer: object detection trained from scratch in pure PyTorch.

Pipeline:
- dataset.py     : YOLO-format dataset loader with letterbox + augmentation
- model.py       : Darknet-style backbone + single-scale detection head
- loss.py        : YOLO-style objectness + box + class loss with target builder
- postprocess.py : decode raw grid output to boxes, confidence filter, NMS
- describe.py    : turn boxes into spatial language (location, size, relations)
- memory_bridge.py : write what the model saw into the Rust-backed brain memory
"""
