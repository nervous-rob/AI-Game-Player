"""Frame capture implementation."""
import time
import threading
from queue import Queue
import cv2
import numpy as np
from typing import Optional, Tuple
from .frame_types import Frame, FrameMetadata, CaptureConfig, CaptureError
from .win32_capture import Win32ScreenCapturer

class FrameCapture:
    """Captures frames from the screen."""
    def __init__(self, capture_region: Optional[Tuple[int, int, int, int]] = None):
        self.config = CaptureConfig()
        self._impl = Win32ScreenCapturer()
        self._frame_queue = Queue(maxsize=30)  # Buffer 0.5s at 60fps
        self._processing_thread = None
        self._should_process = False
        self._is_initialized = False
        self._is_paused = False
        self._frame_count = 0
        self._dropped_frames = 0
        self._current_fps = 60.0  # Start with target FPS
        self._last_frame_time = time.time()
        self._frame_times = []
        self._fps_update_time = time.time()
        
        # Validate and set capture region
        if capture_region is not None:
            x, y, w, h = capture_region
            if w <= 0 or h <= 0:
                raise ValueError("Capture region width and height must be positive")
            if x < 0 or y < 0:
                raise ValueError("Capture region coordinates must be non-negative")
        self.capture_region = capture_region
    
    def initialize(self, window_title: str, config: Optional[CaptureConfig] = None):
        """Initialize the frame capturer."""
        if self._is_initialized:
            raise CaptureError("Capturer already initialized")
        
        if config:
            self.config = config
            if config.region is not None:
                x, y, w, h = config.region
                if w <= 0 or h <= 0:
                    raise ValueError("Capture region width and height must be positive")
                if x < 0 or y < 0:
                    raise ValueError("Capture region coordinates must be non-negative")
                self.capture_region = config.region
        
        self._impl.initialize(window_title)
        self._is_initialized = True
    
    def capture_frame(self) -> Frame:
        """Capture a single frame."""
        if not self._is_initialized:
            raise CaptureError("Capturer not initialized")
        
        if self._is_paused:
            raise CaptureError("Capturer is paused")
        
        # Frame pacing
        if self.config.enable_frame_pacing and self.config.target_fps > 0:
            current_time = time.time()
            frame_delta = current_time - self._last_frame_time
            target_frame_time = 1.0 / self.config.target_fps
            
            if frame_delta < target_frame_time:
                time.sleep(target_frame_time - frame_delta)
                current_time = time.time()
                frame_delta = current_time - self._last_frame_time
        
        # Capture frame
        frame_data = self._impl.capture()
        if self.capture_region:
            x, y, w, h = self.capture_region
            frame_data = frame_data[y:y+h, x:x+w]
        
        # Convert to RGB (drop alpha channel)
        frame_data = frame_data[:, :, :3]
        
        # Update metrics
        current_time = time.time()
        if self.config.enable_metrics:
            self._frame_times.append(current_time)
            
            # Keep only last second of frame times
            while self._frame_times and self._frame_times[0] < current_time - 1.0:
                self._frame_times.pop(0)
            
            # Update FPS every 100ms
            if current_time - self._fps_update_time >= 0.1:
                if len(self._frame_times) > 1:
                    self._current_fps = (len(self._frame_times) - 1) / (
                        self._frame_times[-1] - self._frame_times[0]
                    )
                self._fps_update_time = current_time
        
        self._last_frame_time = current_time
        
        metadata = FrameMetadata(
            timestamp=int(current_time * 1_000_000),  # microseconds
            width=frame_data.shape[1],
            height=frame_data.shape[0],
            channels=frame_data.shape[2],
            sequence_num=self._frame_count
        )
        
        self._frame_count += 1
        return Frame(metadata=metadata, data=frame_data)
    
    def start_capture(self):
        """Start continuous frame capture."""
        if not self._is_initialized:
            raise CaptureError("Capturer not initialized")
        
        if self._should_process:
            return
        
        self._should_process = True
        self._processing_thread = threading.Thread(target=self._capture_loop)
        self._processing_thread.start()
    
    def stop_capture(self):
        """Stop continuous frame capture."""
        self._should_process = False
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join()
            self._processing_thread = None
    
    def pause(self):
        """Pause frame capture."""
        self._is_paused = True
    
    def resume(self):
        """Resume frame capture."""
        self._is_paused = False
    
    def is_capturing(self) -> bool:
        """Check if capture is active."""
        return self._is_initialized and not self._is_paused
    
    @property
    def frame_count(self) -> int:
        """Get total number of frames captured."""
        return self._frame_count
    
    @property
    def dropped_frames(self) -> int:
        """Get number of dropped frames."""
        return self._dropped_frames
    
    @property
    def current_fps(self) -> float:
        """Get current frames per second."""
        return self._current_fps
    
    def shutdown(self):
        """Clean up resources."""
        self.stop_capture()
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Queue.Empty:
                break
        self._impl.shutdown()
        self._is_initialized = False
    
    def _capture_loop(self):
        """Main capture loop."""
        while self._should_process:
            if self._is_paused:
                time.sleep(0.001)  # Small sleep to prevent CPU spin
                continue
            
            try:
                # Capture frame
                frame = self.capture_frame()
                
                # Queue frame
                try:
                    self._frame_queue.put_nowait(frame)
                except Queue.Full:
                    self._dropped_frames += 1
                
            except Exception as e:
                print(f"Error in capture loop: {e}")
                continue 