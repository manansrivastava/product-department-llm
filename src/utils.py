
def extract_department(text: str) -> str:
    """
    Extract department from model output.
    Example:
    'Product ->: Department' → 'Department'
    """
    if "->:" not in text:
        return ""
    return text.split("->:")[1].strip()
