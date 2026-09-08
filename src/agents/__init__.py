from .tabular_q import TabularQAgent

try:
    from .dqn import DQNAgent
except ImportError:
    DQNAgent = None  # torch not installed; DQN unavailable
