def build_repair_prompt() -> str:
    return (
        "Repair the answer using verified evidence only. "
        "Do not add unsupported claims. Keep the required output contract."
    )
