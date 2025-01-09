# AI Game-Playing System: Monitoring and Analytics System

## Performance Monitoring

### Telemetry System
```python
# monitoring/telemetry.py
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import numpy as np
from collections import deque

@dataclass
class PerformanceMetrics:
    """Core performance metrics for system monitoring."""
    frame_processing_time: float
    inference_time: float
    input_simulation_time: float
    total_latency: float
    fps: float
    memory_usage: float
    gpu_usage: float
    queue_sizes: Dict[str, int]

class TelemetryCollector:
    def __init__(self, history_size: int = 1000):
        self.metrics_history = deque(maxlen=history_size)
        self.current_session_start = None
        self.current_metrics = None
        
        # Performance tracking
        self.frame_times = deque(maxlen=100)
        self.latency_buffer = deque(maxlen=100)
    
    def start_session(self):
        """Initialize a new monitoring session."""
        self.current_session_start = time.time()
        self.current_metrics = {
            'start_time': self.current_session_start,
            'frame_count': 0,
            'action_count': 0,
            'error_count': 0
        }
    
    def record_frame_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics for a single frame."""
        self.metrics_history.append(metrics)
        self.frame_times.append(metrics.frame_processing_time)
        self.latency_buffer.append(metrics.total_latency)
        
        if self.current_metrics:
            self.current_metrics['frame_count'] += 1
    
    def get_performance_summary(self) -> Dict:
        """Generate summary of current performance metrics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-100:]
        
        return {
            'average_fps': np.mean([m.fps for m in recent_metrics]),
            'average_latency': np.mean([m.total_latency for m in recent_metrics]),
            'frame_processing_time': np.mean([m.frame_processing_time for m in recent_metrics]),
            'inference_time': np.mean([m.inference_time for m in recent_metrics]),
            'memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'gpu_usage': np.mean([m.gpu_usage for m in recent_metrics])
        }
```

### Real-time Monitoring Dashboard
```python
# monitoring/dashboard.py
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List
import pandas as pd

class MonitoringDashboard:
    def __init__(self, telemetry_collector: TelemetryCollector):
        self.telemetry = telemetry_collector
        self.initialize_dashboard()
    
    def initialize_dashboard(self):
        """Set up the Streamlit dashboard structure."""
        st.set_page_config(layout="wide")
        st.title("AI Game-Playing System Monitor")
        
        # Create dashboard layout
        col1, col2 = st.columns(2)
        
        with col1:
            self.performance_metrics = st.empty()
            self.fps_chart = st.empty()
        
        with col2:
            self.latency_chart = st.empty()
            self.resource_usage = st.empty()
    
    def update_dashboard(self):
        """Update dashboard with latest metrics."""
        metrics = self.telemetry.get_performance_summary()
        
        # Update performance metrics
        self.performance_metrics.metric(
            "System Performance",
            f"{metrics['average_fps']:.1f} FPS",
            f"{metrics['average_latency']:.1f} ms latency"
        )
        
        # Update FPS chart
        self._update_fps_chart()
        
        # Update latency chart
        self._update_latency_chart()
        
        # Update resource usage
        self._update_resource_usage(metrics)
    
    def _update_fps_chart(self):
        """Update FPS time series chart."""
        recent_metrics = list(self.telemetry.metrics_history)[-100:]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=[m.fps for m in recent_metrics],
            mode='lines',
            name='FPS'
        ))
        
        fig.update_layout(
            title="Frames Per Second",
            xaxis_title="Time",
            yaxis_title="FPS"
        )
        
        self.fps_chart.plotly_chart(fig)
    
    def _update_latency_chart(self):
        """Update system latency chart."""
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=list(self.telemetry.latency_buffer),
            nbinsx=30,
            name='Latency Distribution'
        ))
        
        fig.update_layout(
            title="System Latency Distribution",
            xaxis_title="Latency (ms)",
            yaxis_title="Count"
        )
        
        self.latency_chart.plotly_chart(fig)
```

### Analytics Engine
```python
# monitoring/analytics.py
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats

class PerformanceAnalyzer:
    def __init__(self, telemetry_collector: TelemetryCollector):
        self.telemetry = telemetry_collector
    
    def analyze_performance(self) -> Dict:
        """Generate comprehensive performance analysis."""
        metrics = pd.DataFrame([
            vars(m) for m in self.telemetry.metrics_history
        ])
        
        analysis = {
            'frame_processing': self._analyze_frame_processing(metrics),
            'latency': self._analyze_latency(metrics),
            'resource_usage': self._analyze_resource_usage(metrics),
            'stability': self._analyze_stability(metrics)
        }
        
        return analysis
    
    def _analyze_frame_processing(self, metrics: pd.DataFrame) -> Dict:
        """Analyze frame processing performance."""
        return {
            'average_fps': metrics['fps'].mean(),
            'fps_stability': metrics['fps'].std(),
            'frame_time_percentiles': {
                '50th': np.percentile(metrics['frame_processing_time'], 50),
                '95th': np.percentile(metrics['frame_processing_time'], 95),
                '99th': np.percentile(metrics['frame_processing_time'], 99)
            }
        }
    
    def _analyze_latency(self, metrics: pd.DataFrame) -> Dict:
        """Analyze system latency patterns."""
        return {
            'average_latency': metrics['total_latency'].mean(),
            'latency_std': metrics['total_latency'].std(),
            'latency_breakdown': {
                'frame_processing': metrics['frame_processing_time'].mean(),
                'inference': metrics['inference_time'].mean(),
                'input_simulation': metrics['input_simulation_time'].mean()
            }
        }
    
    def _analyze_resource_usage(self, metrics: pd.DataFrame) -> Dict:
        """Analyze system resource utilization."""
        return {
            'average_memory': metrics['memory_usage'].mean(),
            'average_gpu': metrics['gpu_usage'].mean(),
            'memory_trend': stats.linregress(
                range(len(metrics)),
                metrics['memory_usage']
            ).slope
        }
    
    def _analyze_stability(self, metrics: pd.DataFrame) -> Dict:
        """Analyze system stability metrics."""
        fps_changes = metrics['fps'].diff().abs()
        latency_changes = metrics['total_latency'].diff().abs()
        
        return {
            'fps_volatility': fps_changes.mean(),
            'latency_volatility': latency_changes.mean(),
            'stable_fps_percentage': (fps_changes < 5).mean() * 100,
            'stable_latency_percentage': (latency_changes < 10).mean() * 100
        }
```

This implementation provides:

1. A comprehensive telemetry system that collects and tracks:
   - Frame processing performance
   - System latency measurements
   - Resource utilization
   - Queue states and processing times

2. A real-time monitoring dashboard featuring:
   - Live performance metrics
   - Interactive visualizations
   - Resource utilization tracking
   - System health indicators

3. An analytics engine that offers:
   - Detailed performance analysis
   - Latency breakdown and bottleneck identification
   - Resource usage patterns
   - System stability metrics

The monitoring system enables:
- Real-time performance tracking
- Early detection of performance issues
- Historical trend analysis
- Performance optimization insights

This system provides comprehensive monitoring and analytics capabilities while maintaining minimal overhead on the main processing pipeline. Would you like me to proceed with the final part, which would cover system integration and deployment?