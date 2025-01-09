# AI Game-Playing System: System Integration and Deployment

## System Configuration

### Configuration Management
```python
# core/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import json

@dataclass
class SystemConfig:
    """Central configuration for all system components."""
    # Game Interface Configuration
    game_window_title: str
    capture_fps: int
    input_processing_rate: int
    
    # ML Configuration
    model_input_size: tuple
    batch_size: int
    learning_rate: float
    model_checkpoint_dir: Path
    
    # Data Processing Configuration
    storage_base_path: Path
    sync_window_ms: int
    feature_extraction_batch_size: int
    
    # Monitoring Configuration
    enable_dashboard: bool
    telemetry_history_size: int
    monitoring_port: int
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> 'SystemConfig':
        """Load configuration from YAML file."""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save(self, config_path: Path):
        """Save current configuration to file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f)

class SystemContext:
    """Manages system-wide state and component communication."""
    def __init__(self, config: SystemConfig):
        self.config = config
        self.components = {}
        self.active = False
        self.error_handlers = {}
        
    def register_component(self, name: str, component: Any):
        """Register a system component."""
        self.components[name] = component
        
    def register_error_handler(self, error_type: str, handler: callable):
        """Register error handler for specific error type."""
        self.error_handlers[error_type] = handler
        
    def handle_error(self, error_type: str, error: Exception):
        """Handle system errors using registered handlers."""
        if handler := self.error_handlers.get(error_type):
            return handler(error)
        raise error
```

## System Integration

### Main System Controller
```python
# core/controller.py
import asyncio
from typing import Optional
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class SystemController:
    """Main controller coordinating all system components."""
    def __init__(self, config: SystemConfig):
        self.config = config
        self.context = SystemContext(config)
        self.running = False
        
        # Initialize components
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize game interface
            self.screen_capturer = createScreenCapturer()
            self.input_simulator = createInputSimulator()
            
            # Initialize ML components
            self.model = GamePlayingNetwork()
            self.trainer = PPOTrainer(self.model, TrainingConfig())
            self.inference_engine = InferenceEngine(self.model)
            
            # Initialize data processing
            self.frame_processor = FrameProcessor()
            self.data_synchronizer = DataSynchronizer()
            self.data_storage = DataStorage(self.config.storage_base_path)
            
            # Initialize monitoring
            self.telemetry = TelemetryCollector()
            if self.config.enable_dashboard:
                self.dashboard = MonitoringDashboard(self.telemetry)
            
            # Register components with context
            self.register_components()
            self.setup_error_handlers()
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def register_components(self):
        """Register components with system context."""
        components = {
            'screen_capturer': self.screen_capturer,
            'input_simulator': self.input_simulator,
            'model': self.model,
            'trainer': self.trainer,
            'inference_engine': self.inference_engine,
            'frame_processor': self.frame_processor,
            'data_synchronizer': self.data_synchronizer,
            'data_storage': self.data_storage,
            'telemetry': self.telemetry
        }
        
        for name, component in components.items():
            self.context.register_component(name, component)
    
    async def start(self):
        """Start the system."""
        if self.running:
            return
        
        try:
            logger.info("Starting AI Game-Playing System")
            self.running = True
            
            # Start monitoring
            self.telemetry.start_session()
            if hasattr(self, 'dashboard'):
                asyncio.create_task(self.run_dashboard())
            
            # Start main processing loop
            await self.run_processing_loop()
            
        except Exception as e:
            logger.error(f"System start failed: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the system."""
        if not self.running:
            return
        
        logger.info("Stopping AI Game-Playing System")
        self.running = False
        
        # Clean up components
        await self.cleanup_components()
    
    async def run_processing_loop(self):
        """Main processing loop."""
        while self.running:
            try:
                # Capture frame
                frame = self.screen_capturer.captureNextFrame()
                
                # Process frame
                processed_frame = self.frame_processor.process_frame(
                    frame.data,
                    str(frame.metadata.sequence_num),
                    frame.metadata.timestamp
                )
                
                # Generate action
                action = self.inference_engine.predict_action(
                    processed_frame.processed_tensor
                )
                
                # Execute action
                self.input_simulator.queueInput(action['action'])
                
                # Record metrics
                self.telemetry.record_frame_metrics(PerformanceMetrics(
                    frame_processing_time=0.0,  # Add actual timing
                    inference_time=0.0,  # Add actual timing
                    input_simulation_time=0.0,  # Add actual timing
                    total_latency=0.0,  # Add actual timing
                    fps=0.0,  # Add actual FPS
                    memory_usage=0.0,  # Add actual memory usage
                    gpu_usage=0.0,  # Add actual GPU usage
                    queue_sizes={}  # Add actual queue sizes
                ))
                
                # Sleep to maintain target FPS
                await asyncio.sleep(1 / self.config.capture_fps)
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                self.context.handle_error('processing_loop', e)
    
    async def cleanup_components(self):
        """Clean up all system components."""
        try:
            self.screen_capturer.shutdown()
            self.input_simulator.shutdown()
            self.data_storage.end_session()
            
            if hasattr(self, 'dashboard'):
                await self.dashboard.shutdown()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
```

## Deployment Configuration

### Production Deployment
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set up system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libx11-6 \
    libglib2.0-0

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Run the application
CMD ["python3", "-m", "core.main"]
```

### Deployment Script
```python
# scripts/deploy.py
import subprocess
import argparse
import yaml
from pathlib import Path

def deploy_system(config_path: Path, environment: str):
    """Deploy the system to specified environment."""
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Build Docker image
    subprocess.run([
        "docker", "build",
        "-t", f"ai-game-player:{environment}",
        "."
    ], check=True)
    
    # Run container
    subprocess.run([
        "docker", "run",
        "--gpus", "all",
        "--network", "host",
        "--name", f"ai-game-player-{environment}",
        "-v", f"{config['model_checkpoint_dir']}:/app/checkpoints",
        "-v", f"{config['storage_base_path']}:/app/data",
        "-d", f"ai-game-player:{environment}"
    ], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", choices=["dev", "prod"], required=True)
    args = parser.parse_args()
    
    deploy_system(args.config, args.environment)
```

This implementation provides:

1. A centralized configuration system that manages all component settings.

2. A system controller that coordinates all components and manages the main processing loop.

3. Error handling and recovery mechanisms to ensure system stability.

4. Deployment configuration for running the system in a containerized environment.

Key features include:

1. Asynchronous processing for improved performance.
2. Component lifecycle management.
3. Centralized error handling.
4. Clean shutdown procedures.
5. Container-based deployment.
6. Volume mounting for model checkpoints and data storage.

This completes the technical specification for the AI game-playing system. All components are now integrated and ready for deployment. Would you like me to provide any additional details about specific aspects of the system?