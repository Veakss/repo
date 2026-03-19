import re

def extract_marker_content(text: str, start_marker: str = "[START_MARKER]", end_marker: str = "[END_MARKER]") -> str | None:
    """
    Extracts content between two specified markers in a string.

    Args:
        text: The input string to search within.
        start_marker: The starting delimiter.
        end_marker: The ending delimiter.

    Returns:
        The extracted content as a string, or None if markers are not found.
    """
    # Escape special regex characters in markers
    start = re.escape(start_marker)
    end = re.escape(end_marker)
    
    # Pattern to capture content non-greedily between the markers
    pattern = rf"{start}(.*?){end}"
    
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        content = match.group(1)
        # Explicitly remove leading/trailing newlines from the captured content
        content = re.sub(r'^\s+|\s+$', '', content)
        return content
    return None

if __name__ == '__main__':
    # Example usage for demonstration
    sample_text = "Some preamble text. [START_MARKER]\\nThis is the important data to extract.\\n[END_MARKER] Some trailing text."
    content = extract_marker_content(sample_text)
    print(f"Extracted Content: {content}")
