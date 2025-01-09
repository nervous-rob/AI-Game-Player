"""Input simulation implementation."""
import time
import threading
from queue import Queue, Full, Empty
from typing import Optional, List
from .input_types import (
    InputType, InputEvent, KeyboardEvent,
    MouseMoveEvent, MouseButtonEvent, GamepadEvent
)
from .win32_input import Win32InputSimulator

class InputSimulationError(Exception):
    """Error raised by input simulation operations."""
    pass

class InputSimulationConfig:
    """Configuration for input simulation."""
    def __init__(self, input_queue_size: int = 100):
        if input_queue_size <= 0:
            raise ValueError("Input queue size must be positive")
        self.input_queue_size = input_queue_size

class InputSimulator:
    """Simulates keyboard, mouse and gamepad inputs."""
    def __init__(self):
        self._config = InputSimulationConfig()
        self._impl = Win32InputSimulator()
        self._input_queue = None
        self._processing_thread = None
        self._should_process = False
        self._is_initialized = False
    
    @property
    def config(self) -> InputSimulationConfig:
        """Get the current configuration."""
        return self._config
    
    @config.setter
    def config(self, value: InputSimulationConfig):
        """Set the configuration."""
        if not isinstance(value, InputSimulationConfig):
            raise TypeError("Config must be an InputSimulationConfig instance")
        
        self._config = value
        if self._input_queue is not None:
            # Create new queue with updated size
            old_queue = self._input_queue
            self._input_queue = Queue(maxsize=value.input_queue_size)
            
            # Transfer items from old queue to new queue
            try:
                while not old_queue.empty():
                    item = old_queue.get_nowait()
                    if not self._input_queue.full():
                        self._input_queue.put_nowait(item)
                    old_queue.task_done()
            except Empty:
                pass
    
    def initialize(self, window_title: str, config: Optional[InputSimulationConfig] = None):
        """Initialize the input simulator."""
        if self._is_initialized:
            raise InputSimulationError("Already initialized")
        
        if config:
            self.config = config
        
        self._impl.initialize(window_title)
        self._input_queue = Queue(maxsize=self.config.input_queue_size)
        self._is_initialized = True
    
    def queue_input(self, event: InputEvent):
        """Queue an input event for processing."""
        if not self._is_initialized:
            raise InputSimulationError("Not initialized")
        
        if self._input_queue.full():
            raise InputSimulationError("Input queue full")
        
        try:
            self._input_queue.put_nowait(event)
        except Full:
            raise InputSimulationError("Input queue full")
    
    def queue_inputs(self, events: List[InputEvent]):
        """Queue multiple input events for processing."""
        if not self._is_initialized:
            raise InputSimulationError("Not initialized")
        
        # Check if there's enough space in the queue
        available_space = self.config.input_queue_size - self._input_queue.qsize()
        if len(events) > available_space:
            raise InputSimulationError("Input queue full")
        
        for event in events:
            self.queue_input(event)
    
    def clear_input_queue(self):
        """Clear all pending input events."""
        if not self._is_initialized:
            raise InputSimulationError("Not initialized")
        
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
                self._input_queue.task_done()
            except Empty:
                break
    
    def start_processing(self):
        """Start processing queued inputs."""
        if not self._is_initialized:
            raise InputSimulationError("Not initialized")
        
        if self._should_process:
            return
        
        self._should_process = True
        self._processing_thread = threading.Thread(target=self._process_loop)
        self._processing_thread.start()
    
    def stop_processing(self):
        """Stop processing queued inputs."""
        if self._processing_thread is None:
            return
        
        # Signal thread to stop and get reference
        self._should_process = False
        thread = self._processing_thread
        
        # Wait for thread to finish current task
        if thread.is_alive():
            try:
                thread.join(timeout=1.0)  # Wait up to 1 second
            except Exception as e:
                print(f"Warning: Error while stopping input thread: {e}")
            
            if thread.is_alive():
                print("Warning: Input processing thread did not stop in time")
        
        # Clear thread reference
        self._processing_thread = None
    
    def is_processing(self) -> bool:
        """Check if input processing is active."""
        return self._is_initialized and self._should_process and \
               self._processing_thread is not None and \
               self._processing_thread.is_alive()
    
    def shutdown(self):
        """Clean up resources."""
        self.stop_processing()
        self.clear_input_queue()
        self._impl.shutdown()
        self._is_initialized = False
    
    def _process_loop(self):
        """Main input processing loop."""
        while self._should_process:
            try:
                # Get next input event
                event = self._input_queue.get(timeout=0.001)
                
                try:
                    # Process event based on type
                    if event.type == InputType.KEYBOARD:
                        self._impl.send_keyboard_input(event.data)
                    elif event.type == InputType.MOUSE_MOVE:
                        self._impl.send_mouse_move(event.data)
                    elif event.type == InputType.MOUSE_BUTTON:
                        self._impl.send_mouse_button(event.data)
                    elif event.type == InputType.GAMEPAD:
                        self._impl.send_gamepad_input(event.data)
                finally:
                    # Always mark task as done, even if processing failed
                    self._input_queue.task_done()
                
            except Empty:
                continue  # No input available
            except Exception as e:
                print(f"Error processing input: {e}")
                continue 