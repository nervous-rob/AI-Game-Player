"""Input types for the game interface."""
from dataclasses import dataclass
from typing import Union, Dict, Any
import enum

class InputType(enum.Enum):
    KEYBOARD = "keyboard"
    MOUSE_MOVE = "mouse_move"
    MOUSE_BUTTON = "mouse_button"
    GAMEPAD = "gamepad"

@dataclass
class KeyboardEvent:
    virtual_key: int
    is_pressed: bool
    is_extended: bool = False

@dataclass
class MouseMoveEvent:
    x: int
    y: int
    delta_x: int = 0
    delta_y: int = 0

@dataclass
class MouseButtonEvent:
    button: int  # 0 = left, 1 = right, 2 = middle
    is_pressed: bool
    x: int = 0
    y: int = 0

@dataclass
class GamepadEvent:
    button: int
    value: float = 1.0  # For analog inputs

@dataclass
class InputEvent:
    type: InputType
    data: Union[KeyboardEvent, MouseMoveEvent, MouseButtonEvent, GamepadEvent]
    timestamp: int = 0  # Microseconds since epoch 