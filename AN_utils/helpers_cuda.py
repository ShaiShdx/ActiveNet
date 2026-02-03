"""
CUDA utilities for GPU acceleration
Handles device management and tensor conversions
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try to import torch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - running on CPU only")


class DeviceManager:
    """Manage CPU/CUDA device selection and tensor operations"""

    def __init__(self, use_cuda=True):
        """
        Initialize device manager

        Args:
            use_cuda: Try to use CUDA if available
        """
        self.torch_available = TORCH_AVAILABLE
        self.device = self._setup_device(use_cuda)

        self.using_cuda = self.device.type == "cuda" if self.torch_available else False
        self.using_mps = self.device.type == "mps" if self.torch_available else False

        logger.info(f"************** USING Device: {self.device} **************")
        logger.info(f"************** MPS_USE ??? {self.using_mps} **************")

        if self.using_cuda:
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        elif self.using_mps:
            logger.info("Using MPS device")
        elif self.torch_available:
            logger.info("Using CPU (CUDA not available)")
        else:
            logger.info("Using NumPy on CPU (PyTorch not installed)")

    def _setup_device(self, use_cuda):
        """Setup compute device"""
        if not self.torch_available:
            return None

        if use_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Running on CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Running on MPS")
        else:
            device = torch.device("cpu")

        return device

    def to_tensor(self, array, dtype=None):
        """
        Convert numpy array to tensor on appropriate device

        Args:
            array: numpy array or existing tensor
            dtype: torch dtype (default: float32 for cuda, float64 for cpu)

        Returns:
            tensor on device (or numpy array if torch not available)
        """
        if not self.torch_available:
            return array

        if isinstance(array, torch.Tensor):
            return array.to(self.device)

        # # Use float32 for GPU (faster), float64 for CPU (precision)
        # if dtype is None:
        #     dtype = (
        #         torch.float32 if self.using_cuda or self.using_mps else torch.float64
        #     )

        if dtype is None:
            dtype = (
                torch.float32 if (self.using_cuda or self.using_mps) else torch.float64
            )
            # if self.using_cuda or self.using_mps:
            #     dtype = torch.float32
            # else:
            #     dtype = torch.float64

        tensor = torch.from_numpy(array).to(dtype)
        return tensor.to(self.device)

    def to_numpy(self, tensor):
        """
        Convert tensor to numpy array

        Args:
            tensor: torch tensor or numpy array

        Returns:
            numpy array
        """
        if not self.torch_available or isinstance(tensor, np.ndarray):
            return tensor

        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()

        return tensor

    def zeros(self, shape, dtype=None):
        """Create zeros array/tensor"""
        if not self.torch_available:
            return np.zeros(shape)

        if dtype is None:
            dtype = (
                torch.float32 if (self.using_cuda or self.using_mps) else torch.float64
            )

        return torch.zeros(shape, dtype=dtype, device=self.device)

    def ones(self, shape, dtype=None):
        """Create ones array/tensor"""
        if not self.torch_available:
            return np.ones(shape)

        if dtype is None:
            dtype = (
                torch.float32 if (self.using_cuda or self.using_mps) else torch.float64
            )
        return torch.ones(shape, dtype=dtype, device=self.device)

    def randn(self, shape, dtype=None):
        """Create random normal array/tensor"""
        if not self.torch_available:
            return np.random.randn(*shape)

        if dtype is None:
            dtype = (
                torch.float32 if (self.using_cuda or self.using_mps) else torch.float64
            )
        return torch.randn(shape, dtype=dtype, device=self.device)

    def sqrt(self, x):
        """Square root operation"""
        if not self.torch_available:
            return np.sqrt(x)
        return torch.sqrt(x)

    def exp(self, x):
        """Exponential operation"""
        if not self.torch_available:
            return np.exp(x)
        return torch.exp(x)

    def sum(self, x, axis=None):
        """Sum operation"""
        if not self.torch_available:
            return np.sum(x, axis=axis)
        if axis is None:
            return torch.sum(x)
        return torch.sum(x, dim=axis)


# Global device manager instance
_device_manager = None


def get_device_manager(use_cuda=True):
    """Get or create global device manager"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(use_cuda=use_cuda)
    return _device_manager


def reset_device_manager():
    """Reset global device manager"""
    global _device_manager
    _device_manager = None
