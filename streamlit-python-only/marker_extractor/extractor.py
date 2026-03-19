import re

def extract_markers(text: str) -> list[str]:
    """
    Extracts all substrings enclosed in double curly braces {{...}} from the text.
    """
    # Regex to find content between {{ and }} non-greedily
    markers = re.findall(r"\{\{(.*?)\}\}", text)
    return markers

if __name__ == '__main__':
    sample_text = "This is a test with {{marker_one}} and another {{marker_two}}."
    extracted = extract_markers(sample_text)
    print(f"Sample Text: {sample_text}")
    print(f"Extracted Markers: {extracted}")