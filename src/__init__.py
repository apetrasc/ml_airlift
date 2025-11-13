"""
Source code for airlift ML project.
"""

import importlib.util
import os
from pathlib import Path

# Import functions from src/utils.py (file) instead of src/utils/ (package)
# This is necessary because both src/utils.py and src/utils/ exist
_src_dir = Path(__file__).parent
_utils_py_path = _src_dir / "utils.py"

if _utils_py_path.exists():
    spec = importlib.util.spec_from_file_location("src.utils_module", _utils_py_path)
    utils_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_module)
    
    preprocess_and_predict = utils_module.preprocess_and_predict
    preprocess = utils_module.preprocess
    hilbert_cuda = utils_module.hilbert_cuda
    debug_pipeline = utils_module.debug_pipeline
    get_valid_data = utils_module.get_valid_data
else:
    raise ImportError(f"Could not find {_utils_py_path}")

__version__ = "0.1.0"

__all__ = [
    'preprocess_and_predict',
    'preprocess',
    'debug_pipeline',
    'get_valid_data',
]
