"""
Logging utilities
"""

import logging
import os
import sys
from typing import Optional
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (Optional[str]): Path to log file
        log_format (Optional[str]): Custom log format
        include_timestamp (bool): Whether to include timestamp in logs
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger("base-research")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default log format
    if log_format is None:
        if include_timestamp:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            log_format = "%(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "base-research") -> logging.Logger:
    """
    Get logger instance
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    Logger for training metrics and progress
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup loggers
        self.logger = setup_logging(
            log_level="INFO",
            log_file=os.path.join(self.log_dir, f"{self.experiment_name}.log")
        )
        
        # Metrics file
        self.metrics_file = os.path.join(self.log_dir, f"{self.experiment_name}_metrics.log")
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.metrics = {}
    
    def log_info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """
        Log training metrics
        
        Args:
            metrics (dict): Metrics dictionary
            step (Optional[int]): Current step
        """
        if step is not None:
            self.step = step
        
        self.metrics.update(metrics)
        
        # Log to console
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {self.step}: {metrics_str}")
        
        # Log to metrics file
        with open(self.metrics_file, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp} - Step {self.step}: {metrics_str}\n")
    
    def log_epoch_start(self, epoch: int) -> None:
        """Log epoch start"""
        self.epoch = epoch
        self.logger.info(f"Starting epoch {epoch}")
    
    def log_epoch_end(self, epoch: int, epoch_metrics: dict) -> None:
        """Log epoch end"""
        self.epoch = epoch
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()])
        self.logger.info(f"Epoch {epoch} completed: {metrics_str}")
        
        # Log to metrics file
        with open(self.metrics_file, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp} - Epoch {epoch}: {metrics_str}\n")
    
    def log_training_start(self, config: dict) -> None:
        """Log training start"""
        self.logger.info("=" * 50)
        self.logger.info(f"Starting training experiment: {self.experiment_name}")
        self.logger.info("=" * 50)
        
        # Log configuration
        self.logger.info("Configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"    {sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_training_end(self, final_metrics: dict) -> None:
        """Log training end"""
        self.logger.info("=" * 50)
        self.logger.info("Training completed!")
        self.logger.info("Final metrics:")
        for key, value in final_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        self.logger.info("=" * 50)
    
    def log_model_info(self, model_info: dict) -> None:
        """Log model information"""
        self.logger.info("Model Information:")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_checkpoint_saved(self, checkpoint_path: str, metrics: dict) -> None:
        """Log checkpoint saved"""
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Checkpoint saved: {checkpoint_path} ({metrics_str})")
    
    def log_early_stopping(self, patience: int, best_metric: float) -> None:
        """Log early stopping"""
        self.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
        self.logger.info(f"Best metric: {best_metric:.4f}")


def create_experiment_logger(
    experiment_name: str,
    log_dir: str = "logs",
    log_level: str = "INFO"
) -> TrainingLogger:
    """
    Create experiment logger
    
    Args:
        experiment_name (str): Name of the experiment
        log_dir (str): Directory for log files
        log_level (str): Logging level
        
    Returns:
        TrainingLogger: Configured training logger
    """
    return TrainingLogger(log_dir=log_dir, experiment_name=experiment_name)


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information
    
    Args:
        logger (logging.Logger): Logger instance
    """
    import platform
    import torch
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python version: {platform.python_version()}")
    logger.info(f"  PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA available: Yes")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info(f"  CUDA available: No")


def setup_file_logging(log_file: str, log_level: str = "INFO") -> None:
    """
    Setup file logging only
    
    Args:
        log_file (str): Path to log file
        log_level (str): Logging level
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
