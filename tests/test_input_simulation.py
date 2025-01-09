"""Tests for input simulation module."""
import pytest
import time
from unittest.mock import patch
from src.game_interface.input_simulation import (
    InputSimulator,
    InputSimulationConfig,
    InputSimulationError
)
from src.game_interface.input_types import (
    InputType,
    InputEvent,
    KeyboardEvent,
    MouseMoveEvent,
    MouseButtonEvent
)

@pytest.fixture
def mock_window():
    """Mock window for testing."""
    with patch('win32gui.FindWindow', return_value=42):
        yield

def test_input_simulator_initialization(mock_window):
    """Test input simulator initialization."""
    simulator = InputSimulator()
    assert not simulator._is_initialized
    
    # Test initialization
    simulator.initialize("Test Window")
    assert simulator._is_initialized
    
    # Test double initialization
    with pytest.raises(InputSimulationError, match="Already initialized"):
        simulator.initialize("Test Window")

def test_input_queue_management(mock_window):
    """Test input queue operations."""
    simulator = InputSimulator()
    simulator.initialize("Test Window")
    
    # Test queueing single input
    event = InputEvent(
        type=InputType.KEYBOARD,
        data=KeyboardEvent(virtual_key=65, is_pressed=True)
    )
    simulator.queue_input(event)
    assert simulator._input_queue.qsize() == 1
    
    # Test queueing multiple inputs
    events = [
        InputEvent(
            type=InputType.KEYBOARD,
            data=KeyboardEvent(virtual_key=66, is_pressed=True)
        ),
        InputEvent(
            type=InputType.MOUSE_MOVE,
            data=MouseMoveEvent(x=100, y=100)
        )
    ]
    simulator.queue_inputs(events)
    assert simulator._input_queue.qsize() == 3
    
    # Test queue clearing
    simulator.clear_input_queue()
    assert simulator._input_queue.empty()

def test_input_processing_control(mock_window):
    """Test input processing control."""
    simulator = InputSimulator()
    simulator.initialize("Test Window")
    
    # Test starting processing
    simulator.start_processing()
    assert simulator._should_process
    assert simulator._processing_thread is not None
    assert simulator._processing_thread.is_alive()
    
    # Test stopping processing
    thread = simulator._processing_thread
    simulator.stop_processing()
    assert not simulator._should_process
    assert not thread.is_alive()
    assert simulator._processing_thread is None

def test_error_handling(mock_window):
    """Test error handling in input simulation."""
    simulator = InputSimulator()
    
    # Test operations before initialization
    with pytest.raises(InputSimulationError, match="Not initialized"):
        simulator.queue_input(InputEvent(
            type=InputType.KEYBOARD,
            data=KeyboardEvent(virtual_key=65, is_pressed=True)
        ))
    
    with pytest.raises(InputSimulationError, match="Not initialized"):
        simulator.start_processing()
    
    # Test queue overflow
    simulator.initialize("Test Window")
    simulator.config = InputSimulationConfig(input_queue_size=1)
    
    simulator.queue_input(InputEvent(
        type=InputType.KEYBOARD,
        data=KeyboardEvent(virtual_key=65, is_pressed=True)
    ))
    
    with pytest.raises(InputSimulationError, match="Input queue full"):
        simulator.queue_input(InputEvent(
            type=InputType.KEYBOARD,
            data=KeyboardEvent(virtual_key=66, is_pressed=True)
        ))

def test_shutdown(mock_window):
    """Test simulator shutdown."""
    simulator = InputSimulator()
    simulator.initialize("Test Window")
    
    # Start processing and queue some inputs
    simulator.start_processing()
    simulator.queue_input(InputEvent(
        type=InputType.KEYBOARD,
        data=KeyboardEvent(virtual_key=65, is_pressed=True)
    ))
    
    # Shutdown
    simulator.shutdown()
    assert not simulator._is_initialized
    assert not simulator._should_process
    assert simulator._input_queue.empty()
    
    # Test operations after shutdown
    with pytest.raises(InputSimulationError, match="Not initialized"):
        simulator.queue_input(InputEvent(
            type=InputType.KEYBOARD,
            data=KeyboardEvent(virtual_key=65, is_pressed=True)
        )) 