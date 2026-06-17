from __future__ import annotations

from pathlib import Path

from ..brain import Brain
from .describe import describe_detection, describe_relations, describe_scene
from .postprocess import Detection


class VisionMemory:
    """Write what the detector saw into the Rust-backed brain memory.

    Observations are stored with kind="vision" so the existing recall,
    reinforce and reflect machinery can reason over them alongside dialogue.
    """

    def __init__(self, brain: Brain | None = None) -> None:
        self.brain = brain or Brain()

    def record_observation(
        self,
        detections: list[Detection],
        image_w: int,
        image_h: int,
        source: str = "",
        store_relations: bool = True,
    ) -> int:
        tag_source = Path(source).name if source else "stream"

        scene = describe_scene(detections, image_w, image_h)
        self.brain.remember(
            f"Наблюдение [{tag_source}]: {scene}",
            tags=f"vision,scene,{tag_source}",
            importance=0.6,
            kind="vision",
        )

        for det in detections:
            line = describe_detection(det, image_w, image_h)
            importance = min(0.55 + det.confidence * 0.4, 0.95)
            self.brain.remember(
                f"Объект [{tag_source}]: {line}",
                tags=f"vision,object,{det.class_name},{tag_source}",
                importance=importance,
                kind="vision",
            )

        if store_relations and len(detections) > 1:
            for relation in describe_relations(detections):
                self.brain.remember(
                    f"Связь [{tag_source}]: {relation}",
                    tags=f"vision,relation,{tag_source}",
                    importance=0.5,
                    kind="vision",
                )

        return len(detections)
