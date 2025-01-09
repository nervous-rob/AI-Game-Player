"""Frame capture type definitions."""
from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class FrameMetadata:
    """Metadata for a captured frame."""
    timestamp: int  # Microseconds since epoch
    width: int
    height: int
    channels: int
    sequence_num: int

@dataclass
class Frame:
    """A captured frame with metadata."""
    metadata: FrameMetadata
    data: np.ndarray

@dataclass
class CaptureConfig:
    """Configuration for frame capture."""
    target_fps: float = 60.0
    max_frame_queue: int = 30
    enable_metrics: bool = True
    enable_frame_pacing: bool = True
    region: Optional[tuple[int, int, int, int]] = None  # (x, y, width, height)

class CaptureError(Exception):
    """Error raised by frame capture operations."""
    pass 