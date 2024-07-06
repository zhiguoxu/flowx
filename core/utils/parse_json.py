import json
from typing import Dict, Any


# Adapted from https://github.com/KillianLucas/open-interpreter/blob/5b6080fae1f8c68938a1e4fa8667e3744084ee21/interpreter/utils/parse_partial_json.py
# and https://github.com/langchain-ai/langchain/blob/ee579c77c1691bdf6b39aef649e1570516917e28/libs/core/langchain_core/utils/json.py
# MIT License

def parse_partial_json(s: str, strict: bool = False) -> Dict[str, Any]:
    """Parse a JSON string that may be missing closing braces."""

    # Attempt to parse the string as-is.
    try:
        return json.loads(s, strict=strict)
    except json.JSONDecodeError:
        pass

    # Initialize variables.
    new_s = ""
    stack = []
    is_inside_string = False
    escaped = False

    # Process each character in the string one at a time.
    for char in s:
        if is_inside_string:
            if char == '"' and not escaped:
                is_inside_string = False
            elif char == "\n" and not escaped:
                char = "\\n"  # Replace the newline character with the escape sequence.
            elif char == "\\":
                escaped = not escaped
            else:
                escaped = False
        else:
            if char == '"':
                is_inside_string = True
                escaped = False
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char == "}" or char == "]":
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    raise json.JSONDecodeError("Mismatched closing character", s, 0)

        # Append the processed character to the new string.
        new_s += char

    # If we're still inside a string at the end of processing,
    # we need to close the string.
    if is_inside_string:
        new_s += '"'

    # Try to parse mods of string until we succeed or run out of characters.
    while new_s:
        final_s = new_s

        # Close any remaining open structures in the reverse
        # order that they were opened.
        for closing_char in reversed(stack):
            final_s += closing_char

        # Attempt to parse the modified string as JSON.
        try:
            return json.loads(final_s, strict=strict)
        except json.JSONDecodeError:
            # If we still can't parse the string as JSON, try removing the last character
            for index in range(len(new_s) - 1, -1, -1):
                c = new_s[index]
                if c == ",":
                    new_s = new_s[:index]
                    break
                if c in "[{" and index != len(new_s) - 1:
                    new_s = new_s[:index + 1]
                    break
            else:
                new_s = new_s[:-1]

    # If we got here, we ran out of characters to remove
    # and still couldn't parse the string as JSON, so return the parse error
    # for the original string.
    return json.loads(s, strict=strict)
