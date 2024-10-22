from typing import Literal
from pydantic.fields import Field
from auto_flow.core.tool import tool
import json


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
@tool
def get_current_weather(location: str = Field(description="The city and state, e.g. San Francisco, CA"),
                        unit: Literal["celsius", "fahrenheit"] = "fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})
