from streamlit_python_only.runtime import build_runtime_system_prompt


def build_system_prompt(*args, **kwargs) -> str:
    return build_runtime_system_prompt(*args, **kwargs)
