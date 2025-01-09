"""
Tests for gamepad input functionality.
"""
import pytest
from unittest.mock import patch
from src.game_interface.input_simulation import (
    InputSimulator,
    InputSimulationConfig,
    InputSimulationError
)
from src.game_interface.input_types import (
    InputType,
    InputEvent,
    GamepadEvent
)

@pytest.fixture
def mock_window():
    """Mock window for testing."""
    with patch('win32gui.FindWindow', return_value=42):
        yield

def test_gamepad_input_queueing(mock_window):
    """Test queueing of gamepad inputs."""
    simulator = InputSimulator()
    simulator.initialize("Test Window")
    
    # Queue a gamepad button press
    event = InputEvent(
        type=InputType.GAMEPAD,
        data=GamepadEvent(button=0)  # A button
    )
    simulator.queue_input(event)
    assert simulator._input_queue.qsize() == 1
    
    # Queue multiple gamepad events
    events = [
        InputEvent(
            type=InputType.GAMEPAD,
            data=GamepadEvent(button=1)  # B button
        ),
        InputEvent(
            type=InputType.GAMEPAD,
            data=GamepadEvent(button=2, value=0.5)  # Trigger half-pressed
        )
    ]
    simulator.queue_inputs(events)
    assert simulator._input_queue.qsize() == 3

def test_gamepad_input_processing(mock_window):
    """Test gamepad input processing."""
    simulator = InputSimulator()
    simulator.initialize("Test Window")
    
    # Start processing
    simulator.start_processing()
    assert simulator._should_process
    
    # Queue some gamepad inputs
    events = [
        InputEvent(
            type=InputType.GAMEPAD,
            data=GamepadEvent(button=0)
        ),
        InputEvent(
            type=InputType.GAMEPAD,
            data=GamepadEvent(button=1, value=0.75)
        )
    ]
    simulator.queue_inputs(events)
    
    # Let some processing happen
    import time
    time.sleep(0.1)
    
    # Stop processing
    simulator.stop_processing()
    assert simulator._input_queue.empty()

def test_gamepad_error_handling(mock_window):
    """Test error handling for gamepad inputs."""
    simulator = InputSimulator()
    
    # Test queueing before initialization
    with pytest.raises(InputSimulationError, match="Not initialized"):
        simulator.queue_input(InputEvent(
            type=InputType.GAMEPAD,
            data=GamepadEvent(button=0)
        ))
    
    # Initialize with small queue
    simulator.initialize("Test Window")
    simulator.config = InputSimulationConfig(input_queue_size=1)
    
    # Fill queue
    simulator.queue_input(InputEvent(
        type=InputType.GAMEPAD,
        data=GamepadEvent(button=0)
    ))
    
    # Test queue overflow
    with pytest.raises(InputSimulationError, match="Input queue full"):
        simulator.queue_input(InputEvent(
            type=InputType.GAMEPAD,
            data=GamepadEvent(button=1)
        )) 