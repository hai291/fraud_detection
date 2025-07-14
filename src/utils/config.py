"""
Configuration utilities
"""

import yaml
import os
from typing import Dict, Any, Optional
import json


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to file
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        config_path (str): Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.safe_dump(config, f, default_flow_style=False, indent=2)
        elif config_path.endswith('.json'):
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries
    
    Args:
        base_config (Dict[str, Any]): Base configuration
        override_config (Dict[str, Any]): Override configuration
        
    Returns:
        Dict[str, Any]: Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any], required_keys: list) -> None:
    """
    Validate configuration contains required keys
    
    Args:
        config (Dict[str, Any]): Configuration to validate
        required_keys (list): List of required keys
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = []
    
    for key in required_keys:
        if '.' in key:
            # Handle nested keys
            keys = key.split('.')
            current = config
            
            for k in keys:
                if not isinstance(current, dict) or k not in current:
                    missing_keys.append(key)
                    break
                current = current[k]
        else:
            if key not in config:
                missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get value from nested configuration dictionary
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        key (str): Key to retrieve (can be nested with dots)
        default (Any): Default value if key not found
        
    Returns:
        Any: Configuration value
    """
    if '.' in key:
        keys = key.split('.')
        current = config
        
        for k in keys:
            if not isinstance(current, dict) or k not in current:
                return default
            current = current[k]
        
        return current
    else:
        return config.get(key, default)


def set_config_value(config: Dict[str, Any], key: str, value: Any) -> None:
    """
    Set value in nested configuration dictionary
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        key (str): Key to set (can be nested with dots)
        value (Any): Value to set
    """
    if '.' in key:
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    else:
        config[key] = value


def update_config_from_args(config: Dict[str, Any], args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Update configuration from command line arguments
    
    Args:
        config (Dict[str, Any]): Base configuration
        args (Optional[Dict[str, Any]]): Command line arguments
        
    Returns:
        Dict[str, Any]: Updated configuration
    """
    if args is None:
        return config
    
    updated_config = config.copy()
    
    # Map common argument names to config keys
    arg_mappings = {
        'learning_rate': 'training.learning_rate',
        'batch_size': 'training.batch_size',
        'num_epochs': 'training.num_epochs',
        'model_name': 'model.name',
        'output_dir': 'output_dir',
        'seed': 'seed'
    }
    
    for arg_name, config_key in arg_mappings.items():
        if arg_name in args and args[arg_name] is not None:
            set_config_value(updated_config, config_key, args[arg_name])
    
    return updated_config


class ConfigManager:
    """
    Configuration manager for handling multiple config files
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.configs = {}
    
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all configuration files from config directory"""
        if not os.path.exists(self.config_dir):
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
        
        for filename in os.listdir(self.config_dir):
            if filename.endswith(('.yaml', '.yml', '.json')):
                config_name = os.path.splitext(filename)[0]
                config_path = os.path.join(self.config_dir, filename)
                
                try:
                    self.configs[config_name] = load_config(config_path)
                except Exception as e:
                    print(f"Error loading config {filename}: {e}")
        
        return self.configs
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get specific configuration"""
        if config_name not in self.configs:
            config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
            if os.path.exists(config_path):
                self.configs[config_name] = load_config(config_path)
            else:
                raise KeyError(f"Configuration not found: {config_name}")
        
        return self.configs[config_name]
    
    def merge_configs(self, config_names: list) -> Dict[str, Any]:
        """Merge multiple configurations"""
        merged = {}
        
        for config_name in config_names:
            config = self.get_config(config_name)
            merged = merge_configs(merged, config)
        
        return merged
    
    def save_merged_config(self, config_names: list, output_path: str) -> None:
        """Save merged configuration to file"""
        merged = self.merge_configs(config_names)
        save_config(merged, output_path)
