"""Windows-specific input simulation implementation."""
import time
import ctypes
from ctypes import wintypes
import win32api
import win32con
import win32gui
from .input_types import *

class Win32InputError(Exception):
    """Windows-specific input simulation errors."""
    pass

class Win32InputSimulator:
    """Windows implementation of input simulation."""
    def __init__(self):
        self.target_window = None
        self._mouse_position = (0, 0)
    
    def initialize(self, window_title: str):
        """Initialize the Windows input simulator."""
        self.target_window = win32gui.FindWindow(None, window_title)
        if not self.target_window:
            raise Win32InputError(f"Window not found: {window_title}")
    
    def process_input(self, event: InputEvent):
        """Process an input event using Windows API."""
        if not self.target_window:
            raise Win32InputError("Not initialized")
        
        if event.type == InputType.KEYBOARD:
            self._send_keyboard_input(event.data)
        elif event.type == InputType.MOUSE_MOVE:
            self._send_mouse_move(event.data)
        elif event.type == InputType.MOUSE_BUTTON:
            self._send_mouse_button(event.data)
        elif event.type == InputType.GAMEPAD:
            self._send_gamepad_input(event.data)
    
    def _send_keyboard_input(self, event: KeyboardEvent):
        """Send keyboard input using Windows API."""
        flags = 0
        if not event.is_pressed:
            flags |= win32con.KEYEVENTF_KEYUP
        if event.is_extended:
            flags |= win32con.KEYEVENTF_EXTENDEDKEY
        
        win32api.SendMessage(
            self.target_window,
            win32con.WM_KEYDOWN if event.is_pressed else win32con.WM_KEYUP,
            event.virtual_key,
            flags
        )
    
    def _send_mouse_move(self, event: MouseMoveEvent):
        """Send mouse movement using Windows API."""
        # Convert to screen coordinates
        x, y = event.x, event.y
        if self.target_window:
            client_rect = win32gui.GetClientRect(self.target_window)
            screen_x, screen_y = win32gui.ClientToScreen(self.target_window, (x, y))
            x, y = screen_x, screen_y
        
        # Calculate absolute coordinates
        normalized_x = int(x * 65535 / win32api.GetSystemMetrics(win32con.SM_CXSCREEN))
        normalized_y = int(y * 65535 / win32api.GetSystemMetrics(win32con.SM_CYSCREEN))
        
        # Send mouse input
        win32api.mouse_event(
            win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE,
            normalized_x,
            normalized_y,
            0,
            0
        )
        
        self._mouse_position = (x, y)
    
    def _send_mouse_button(self, event: MouseButtonEvent):
        """Send mouse button input using Windows API."""
        button_flags = {
            0: (win32con.MOUSEEVENTF_LEFTDOWN, win32con.MOUSEEVENTF_LEFTUP),
            1: (win32con.MOUSEEVENTF_RIGHTDOWN, win32con.MOUSEEVENTF_RIGHTUP),
            2: (win32con.MOUSEEVENTF_MIDDLEDOWN, win32con.MOUSEEVENTF_MIDDLEUP)
        }
        
        if event.button not in button_flags:
            raise Win32InputError(f"Invalid mouse button: {event.button}")
        
        down_flag, up_flag = button_flags[event.button]
        flag = down_flag if event.is_pressed else up_flag
        
        # If coordinates are provided, move mouse first
        if event.x != 0 or event.y != 0:
            self._send_mouse_move(MouseMoveEvent(x=event.x, y=event.y))
        
        win32api.mouse_event(flag, 0, 0, 0, 0)
    
    def _send_gamepad_input(self, event: GamepadEvent):
        """Send gamepad input using Windows API."""
        # Implementation depends on the specific gamepad API being used
        # (e.g., XInput, DirectInput)
        pass
    
    def shutdown(self):
        """Clean up Windows input simulator."""
        self.target_window = None 