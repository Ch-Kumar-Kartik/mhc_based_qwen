"""
Stream expansion and collapse operations for mHC.

StreamExpand: Converts single-stream hidden states to multi-stream
StreamCollapse: Converts multi-stream hidden states back to single-stream
"""

import torch
import torch.nn as nn


class StreamExpand(nn.Module):
    """Expands single-stream hidden states to multi-stream.
    
    At initialization, this simply copies the hidden state n times to create
    n identical streams. This is the correct initialization for equivalence
    with the original model.
    
    Later during training, the streams will diverge as the model learns
    inter-stream mixing through the H_res matrices.
    
    Input shape: (Batch, Sequence, Hidden)
    Output shape: (Batch, Sequence, n_streams, Hidden)
    """
    
    def __init__(self, n_streams: int = 4):
        """Initialize stream expansion.
        
        Args:
            n_streams: Number of parallel streams to create.
        """
        super().__init__()
        self.n_streams = n_streams
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand single stream to n streams by copying.
        
        Args:
            x: Hidden state of shape (B, S, C).
            
        Returns:
            Expanded hidden state of shape (B, S, n, C).
        """
        # Add stream dimension and expand
        # (B, S, C) -> (B, S, 1, C) -> (B, S, n, C)
        return x.unsqueeze(-2).expand(-1, -1, self.n_streams, -1).contiguous()
    
    def extra_repr(self) -> str:
        return f"n_streams={self.n_streams}"


class StreamCollapse(nn.Module):
    """Collapses multi-stream hidden states to single-stream.
    
    At initialization with equivalence parameters, all streams are identical,
    so averaging produces the same result as taking any single stream.
    
    During/after training, averaging combines the learned multi-stream 
    representations back into a single stream for the output projection.
    
    Input shape: (Batch, Sequence, n_streams, Hidden)
    Output shape: (Batch, Sequence, Hidden)
    """
    
    def __init__(self, n_streams: int = 4):
        """Initialize stream collapse.
        
        Args:
            n_streams: Number of streams (for validation).
        """
        super().__init__()
        self.n_streams = n_streams
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse n streams to single stream by averaging.
        
        Args:
            x: Expanded hidden state of shape (B, S, n, C).
            
        Returns:
            Collapsed hidden state of shape (B, S, C).
        """
        # Average across stream dimension
        return x.mean(dim=-2)
    
    def extra_repr(self) -> str:
        return f"n_streams={self.n_streams}"


class StreamExpandLearnable(nn.Module):
    """Learnable stream expansion (alternative to simple copy).
    
    This version uses learned projections to expand the hidden state.
    Not used for equivalence initialization but could be useful for 
    experiments with different expansion strategies.
    
    Input shape: (Batch, Sequence, Hidden)
    Output shape: (Batch, Sequence, n_streams, Hidden)
    """
    
    def __init__(self, hidden_size: int, n_streams: int = 4):
        """Initialize learnable stream expansion.
        
        Args:
            hidden_size: Hidden dimension.
            n_streams: Number of streams.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_streams = n_streams
        
        # Projection to expand each position
        self.expand_proj = nn.Linear(hidden_size, hidden_size * n_streams, bias=False)
        
        # Initialize to approximate copy behavior
        self._init_weights()
    
    def _init_weights(self):
        """Initialize to approximate identity expansion."""
        with torch.no_grad():
            # Set up to copy input to each stream
            weight = torch.zeros(self.hidden_size * self.n_streams, self.hidden_size)
            for i in range(self.n_streams):
                start = i * self.hidden_size
                end = start + self.hidden_size
                weight[start:end, :] = torch.eye(self.hidden_size)
            self.expand_proj.weight.copy_(weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand with learned projection.
        
        Args:
            x: Hidden state (B, S, C).
            
        Returns:
            Expanded hidden state (B, S, n, C).
        """
        B, S, C = x.shape
        # Project and reshape
        x = self.expand_proj(x)  # (B, S, n*C)
        return x.view(B, S, self.n_streams, C)


class StreamCollapseLearnable(nn.Module):
    """Learnable stream collapse (alternative to simple average).
    
    Uses learned weights to combine streams, initialized to uniform 
    averaging for equivalence.
    
    Input shape: (Batch, Sequence, n_streams, Hidden)
    Output shape: (Batch, Sequence, Hidden)
    """
    
    def __init__(self, hidden_size: int, n_streams: int = 4):
        """Initialize learnable stream collapse.
        
        Args:
            hidden_size: Hidden dimension.
            n_streams: Number of streams.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_streams = n_streams
        
        # Learned stream weights (applied before averaging)
        self.stream_weights = nn.Parameter(torch.ones(n_streams) / n_streams)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse with learned weights.
        
        Args:
            x: Expanded hidden state (B, S, n, C).
            
        Returns:
            Collapsed hidden state (B, S, C).
        """
        # Normalize weights to sum to 1
        weights = torch.softmax(self.stream_weights, dim=0)
        
        # Weighted sum across streams
        # weights: (n,) -> (1, 1, n, 1) for broadcasting
        weights = weights.view(1, 1, -1, 1)
        return (x * weights).sum(dim=-2)
