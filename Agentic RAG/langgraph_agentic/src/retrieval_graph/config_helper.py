"""Helper functions for setting up configurations for the retrieval graph."""

import os
from typing import Optional

from langchain_core.runnables import RunnableConfig

def get_default_config() -> RunnableConfig:
    """
    Get the default configuration with user_id from environment variables.
    
    Returns:
        RunnableConfig: A configuration object with user_id set from the environment.
    """
    user_id = os.environ.get("USER_ID", "default_user")
    retriever_provider = os.environ.get("RETRIEVER_PROVIDER", "elastic-local")
    
    return RunnableConfig(
        configurable={
            "user_id": user_id,
            "retriever_provider": retriever_provider
        }
    )

def ensure_user_id(config: Optional[RunnableConfig] = None) -> RunnableConfig:
    """
    Ensure that the configuration has a user_id set.
    
    Args:
        config (Optional[RunnableConfig]): The configuration to check.
        
    Returns:
        RunnableConfig: A configuration with user_id set.
    """
    if config is None:
        return get_default_config()
    
    configurable = config.get("configurable", {})
    if "user_id" not in configurable:
        user_id = os.environ.get("USER_ID", "default_user")
        configurable["user_id"] = user_id
        return RunnableConfig(configurable=configurable)
    
    return config 