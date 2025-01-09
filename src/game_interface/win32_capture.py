"""Windows screen capture implementation."""
import win32gui
import win32ui
import win32con
import numpy as np
from typing import Optional
from .frame_types import CaptureError

class Win32ScreenCapturer:
    """Windows-specific screen capture implementation."""
    def __init__(self):
        self._hwnd = None
        self._dc = None
        self._memdc = None
        self._bitmap = None
        self._bmp_info = None
        self._width = 0
        self._height = 0
    
    def initialize(self, window_title: str):
        """Initialize the capturer for a specific window."""
        # Find window handle
        self._hwnd = win32gui.FindWindow(None, window_title)
        if not self._hwnd:
            raise CaptureError(f"Window '{window_title}' not found")
        
        try:
            # Get window dimensions
            rect = win32gui.GetClientRect(self._hwnd)
            self._width = rect[2] - rect[0]
            self._height = rect[3] - rect[1]
            
            # Create device contexts
            self._dc = win32gui.GetDC(self._hwnd)
            self._memdc = win32ui.CreateDCFromHandle(self._dc)
            self._bitmap = win32ui.CreateBitmap()
            self._bitmap.CreateCompatibleBitmap(self._memdc, self._width, self._height)
            
            # Set up bitmap info
            self._bmp_info = {
                'bmWidth': self._width,
                'bmHeight': self._height,
                'bmPlanes': 1,
                'bmBitsPixel': 32,
                'bmType': 0
            }
            
        except Exception as e:
            self.shutdown()
            raise CaptureError(f"Failed to initialize capture: {e}")
    
    def capture(self) -> np.ndarray:
        """Capture the current window contents."""
        if not self._hwnd:
            raise CaptureError("Capturer not initialized")
        
        try:
            # Select bitmap into memory DC
            old_bitmap = self._memdc.SelectObject(self._bitmap)
            
            # Copy screen contents
            self._memdc.BitBlt(
                (0, 0), (self._width, self._height),
                win32ui.CreateDCFromHandle(self._dc),
                (0, 0),
                win32con.SRCCOPY
            )
            
            # Get bitmap bits
            self._bitmap.GetBitmapBits(True)
            
            # Convert to numpy array
            img_array = np.frombuffer(
                self._bitmap.GetBitmapBits(True),
                dtype=np.uint8
            )
            img_array = img_array.reshape(self._height, self._width, 4)
            
            # Restore bitmap
            self._memdc.SelectObject(old_bitmap)
            
            return img_array[:, :, :3]  # Drop alpha channel
            
        except Exception as e:
            raise CaptureError(f"Failed to capture frame: {e}")
    
    def shutdown(self):
        """Clean up resources."""
        try:
            if self._memdc:
                self._memdc.DeleteDC()
            if self._bitmap:
                self._bitmap.DeleteObject()
            if self._dc:
                win32gui.ReleaseDC(self._hwnd, self._dc)
        except Exception as e:
            print(f"Error during shutdown: {e}")
        finally:
            self._hwnd = None
            self._dc = None
            self._memdc = None
            self._bitmap = None
            self._bmp_info = None 