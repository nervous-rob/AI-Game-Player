"""
Tests for the frame capture module.
"""
import pytest
import numpy as np
from src.game_interface.frame_capture import FrameCapture

def test_frame_capture_initialization():
    """Test that FrameCapture initializes correctly."""
    # Test with default parameters
    capture = FrameCapture()
    assert capture.capture_region is None
    
    # Test with custom capture region
    region = (0, 0, 1920, 1080)
    capture = FrameCapture(capture_region=region)
    assert capture.capture_region == region

def test_frame_capture_not_implemented():
    """Test that frame capture raises NotImplementedError."""
    capture = FrameCapture()
    with pytest.raises(NotImplementedError):
        capture.capture_frame() 