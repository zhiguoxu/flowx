import os

from auto_flow.core.logging import get_logger

logger = get_logger(__name__)


def get_user_agent() -> str:
    """Get user agent from environment variable."""
    env_user_agent = os.environ.get("USER_AGENT")
    if not env_user_agent:
        logger.warning(
            "USER_AGENT environment variable not set, "
            "consider setting it to identify your requests."
        )
        return "DefaultFlowXUserAgent"
    return env_user_agent
