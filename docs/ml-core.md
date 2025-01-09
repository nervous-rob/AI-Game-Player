# AI Game-Playing System: Machine Learning Core

## Model Architecture

### Policy Network Implementation
```python
# ml_core/models/policy_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

class GamePlayingNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_actions: int = 15,
        hidden_size: int = 512,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Vision backbone using ResNet-style architecture
        self.vision_backbone = nn.Sequential(
            self._make_conv_block(input_channels, 32),
            self._make_conv_block(32, 64),
            self._make_conv_block(64, 128),
            self._make_conv_block(128, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # State processing
        self.state_processor = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Action head (policy)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_actions)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.vision_backbone(state)
        state_features = self.state_processor(features)
        
        action_logits = self.policy_head(state_features)
        value = self.value_head(state_features)
        
        return action_logits, value

class ActionEncoder:
    def __init__(self, num_keys: int = 10, num_mouse_positions: int = 5):
        self.num_keys = num_keys
        self.num_mouse_positions = num_mouse_positions
        self.total_actions = num_keys + (num_mouse_positions * num_mouse_positions)
    
    def encode_action(self, action_dict: Dict) -> int:
        """Convert action dictionary to integer index."""
        if action_dict['type'] == 'keyboard':
            return action_dict['key_id']
        else:  # mouse action
            x_pos = action_dict['x_grid']
            y_pos = action_dict['y_grid']
            return self.num_keys + (y_pos * self.num_mouse_positions + x_pos)
    
    def decode_action(self, action_idx: int) -> Dict:
        """Convert integer index back to action dictionary."""
        if action_idx < self.num_keys:
            return {'type': 'keyboard', 'key_id': action_idx}
        else:
            mouse_idx = action_idx - self.num_keys
            y_pos = mouse_idx // self.num_mouse_positions
            x_pos = mouse_idx % self.num_mouse_positions
            return {
                'type': 'mouse',
                'x_grid': x_pos,
                'y_grid': y_pos
            }
```

### Training Implementation
```python
# ml_core/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    batch_size: int = 64

class PPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        self.action_encoder = ActionEncoder()
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * \
                self.config.gae_lambda * (1 - dones[t]) * last_gae
        
        return advantages
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Execute single training step using PPO."""
        
        # Move data to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        # Get old action probabilities and values
        with torch.no_grad():
            old_action_logits, old_values = self.model(states)
            old_action_probs = F.softmax(old_action_logits, dim=-1)
            old_action_log_probs = F.log_softmax(old_action_logits, dim=-1)
            old_action_log_probs = old_action_log_probs.gather(1, actions)
        
        # Calculate advantages
        advantages = self.compute_gae(rewards, old_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO training loop
        total_loss = 0
        for _ in range(self.config.update_epochs):
            action_logits, values = self.model(states)
            action_probs = F.softmax(action_logits, dim=-1)
            action_log_probs = F.log_softmax(action_logits, dim=-1)
            action_log_probs = action_log_probs.gather(1, actions)
            
            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon
            ) * advantages
            
            # Calculate losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, rewards)
            entropy_loss = -(action_probs * action_log_probs).sum(dim=-1).mean()
            
            loss = (
                policy_loss +
                self.config.value_loss_coef * value_loss -
                self.config.entropy_coef * entropy_loss
            )
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return {
            'total_loss': total_loss / self.config.update_epochs,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item()
        }
```

### Inference Engine
```python
# ml_core/inference/engine.py
import torch
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np

class InferenceEngine:
    def __init__(
        self,
        model: nn.Module,
        action_encoder: ActionEncoder,
        device: str = "cuda",
        temperature: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.action_encoder = action_encoder
        self.temperature = temperature
        
        self.model.eval()
    
    def predict_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Dict:
        """Generate next action based on current game state."""
        with torch.no_grad():
            action_logits, value = self.model(
                state.unsqueeze(0).to(self.device)
            )
            
            if deterministic:
                action_idx = torch.argmax(action_logits).item()
            else:
                # Apply temperature scaling
                scaled_logits = action_logits / self.temperature
                probs = F.softmax(scaled_logits, dim=-1)
                action_idx = torch.multinomial(probs, 1).item()
            
            return {
                'action': self.action_encoder.decode_action(action_idx),
                'value': value.item(),
                'logits': action_logits.cpu().numpy()
            }
```

This implementation provides:

1. A deep neural network architecture designed for game playing, featuring:
   - A vision backbone for processing game frames
   - Separate policy and value heads for actor-critic learning
   - Dropout for regularization
   - Efficient feature extraction

2. A PPO-based training system that includes:
   - Generalized Advantage Estimation
   - Policy clipping for stable updates
   - Entropy regularization
   - Gradient clipping
   - Configurable hyperparameters

3. An inference engine supporting:
   - Both deterministic and stochastic action selection
   - Temperature-based exploration
   - Efficient batch processing
   - Action space encoding/decoding

The system is designed for efficient training and inference while maintaining stability and performance. Would you like me to proceed with Part 6, which would cover the Monitoring and Analytics system?