"""Tests for frame capture functionality."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.game_interface.frame_capture import FrameCapture
from src.game_interface.frame_types import Frame, FrameMetadata, CaptureConfig, CaptureError

@pytest.fixture
def mock_win32():
    """Mock Win32 API calls."""
    with patch('src.game_interface.win32_capture.win32gui.FindWindow') as mock_find, \
         patch('src.game_interface.win32_capture.win32gui.GetClientRect') as mock_rect, \
         patch('src.game_interface.win32_capture.win32gui.GetDC') as mock_dc, \
         patch('src.game_interface.win32_capture.win32ui.CreateDCFromHandle') as mock_create_dc, \
         patch('src.game_interface.win32_capture.win32ui.CreateBitmap') as mock_create_bitmap:
        
        # Set up mock return values
        mock_find.return_value = 12345  # Dummy window handle
        mock_rect.return_value = (0, 0, 1920, 1080)  # x1, y1, x2, y2
        mock_dc.return_value = 67890  # Dummy DC handle
        mock_create_dc.return_value = MagicMock()  # Mock DC object
        mock_create_bitmap.return_value = MagicMock()  # Mock bitmap object
        
        yield {
            'find_window': mock_find,
            'get_client_rect': mock_rect,
            'get_dc': mock_dc,
            'create_dc': mock_create_dc,
            'create_bitmap': mock_create_bitmap
        }

def test_frame_capture_initialization(mock_win32):
    """Test frame capture initialization."""
    capture = FrameCapture()
    
    # Test initialization
    capture.initialize("Test Window")
    assert capture.is_capturing()
    
    # Test double initialization
    with pytest.raises(CaptureError, match="already initialized"):
        capture.initialize("Test Window")
    
    # Test shutdown
    capture.shutdown()
    assert not capture.is_capturing()

def test_frame_capture_configuration(mock_win32):
    """Test frame capture configuration."""
    capture = FrameCapture()
    
    # Test custom configuration
    config = CaptureConfig(
        target_fps=30,
        max_frame_queue=10,
        enable_metrics=True,
        enable_frame_pacing=True,
        region=(0, 0, 1920, 1080)
    )
    
    capture.initialize("Test Window", config)
    assert capture.config.target_fps == 30
    assert capture.config.max_frame_queue == 10
    assert capture.config.enable_metrics
    assert capture.config.enable_frame_pacing
    assert capture.config.region == (0, 0, 1920, 1080)
    
    # Test default configuration
    capture = FrameCapture()
    capture.initialize("Test Window")
    assert capture.config.target_fps == 60.0
    assert capture.config.max_frame_queue == 30
    assert capture.config.enable_metrics
    assert capture.config.enable_frame_pacing
    assert capture.config.region is None

@patch('src.game_interface.win32_capture.Win32ScreenCapturer.capture')
def test_frame_capture_basic(mock_capture, mock_win32):
    """Test basic frame capture functionality."""
    # Mock capture to return a frame
    mock_frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
    mock_capture.return_value = mock_frame
    
    capture = FrameCapture()
    capture.initialize("Test Window")
    
    # Test single frame capture
    frame = capture.capture_frame()
    assert isinstance(frame, Frame)
    assert isinstance(frame.metadata, FrameMetadata)
    assert frame.metadata.width == 1920
    assert frame.metadata.height == 1080
    assert frame.metadata.channels == 3
    assert frame.metadata.sequence_num == 0
    
    # Test continuous capture
    capture.start_capture()
    assert capture._should_process
    capture.stop_capture()
    assert not capture._should_process

def test_frame_capture_state_control(mock_win32):
    """Test frame capture state control."""
    capture = FrameCapture()
    capture.initialize("Test Window")
    
    # Test pause/resume
    assert capture.is_capturing()
    capture.pause()
    assert not capture.is_capturing()
    capture.resume()
    assert capture.is_capturing()
    
    # Test error on capture while paused
    capture.pause()
    with pytest.raises(CaptureError, match="paused"):
        capture.capture_frame()

@patch('src.game_interface.win32_capture.Win32ScreenCapturer.capture')
def test_frame_capture_metrics(mock_capture, mock_win32):
    """Test frame capture performance metrics."""
    # Mock capture to return a frame
    mock_frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
    mock_capture.return_value = mock_frame
    
    capture = FrameCapture()
    capture.initialize("Test Window")
    
    # Capture a few frames
    for _ in range(3):
        frame = capture.capture_frame()
        assert isinstance(frame, Frame)
        assert isinstance(frame.metadata, FrameMetadata)
        assert frame.metadata.sequence_num == capture.frame_count - 1
    
    # Check metrics
    assert capture.frame_count == 3
    assert capture.current_fps > 0
    assert capture.dropped_frames == 0

def test_frame_capture_error_handling(mock_win32):
    """Test frame capture error handling."""
    capture = FrameCapture()
    
    # Test capture before initialization
    with pytest.raises(CaptureError, match="not initialized"):
        capture.capture_frame()
    
    # Test start/stop before initialization
    with pytest.raises(CaptureError, match="not initialized"):
        capture.start_capture()
    
    # Test window not found error
    mock_win32['find_window'].return_value = 0
    with pytest.raises(CaptureError, match="not found"):
        capture.initialize("Nonexistent Window")

def test_frame_capture_region(mock_win32):
    """Test frame capture with region."""
    capture = FrameCapture(capture_region=(100, 100, 500, 500))
    capture.initialize("Test Window")
    
    # Test region is set
    assert capture.capture_region == (100, 100, 500, 500)
    
    # Test invalid region
    with pytest.raises(ValueError):
        FrameCapture(capture_region=(100, 100, -1, 500))

def test_frame_capture_cleanup(mock_win32):
    """Test frame capture cleanup."""
    capture = FrameCapture()
    capture.initialize("Test Window")
    
    # Start capture and verify thread
    capture.start_capture()
    assert capture._processing_thread is not None
    assert capture._processing_thread.is_alive()
    
    # Shutdown and verify cleanup
    capture.shutdown()
    assert not capture._is_initialized
    assert not capture._should_process
    assert capture._processing_thread is None 