"""Configuration loader for the recommender system."""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the recommender system."""

    def __init__(self, config_path: str = None):
        """Initialize configuration.

        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., 'data.num_items')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.

        Args:
            section: Section name (e.g., 'data', 'embeddings')

        Returns:
            Dictionary containing section configuration
        """
        return self._config.get(section, {})

    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config


# Global config instance
_config = None


def get_config(config_path: str = None) -> Config:
    """Get global configuration instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
