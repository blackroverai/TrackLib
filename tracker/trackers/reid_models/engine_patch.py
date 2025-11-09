"""
Patch for engine.py to make TensorRT optional
Run this to patch the engine.py file to work without TensorRT
"""

import os

def patch_engine_file():
    engine_file = "/home/br/git/TrackLib/tracker/trackers/reid_models/engine.py"
    
    # Read the file
    with open(engine_file, 'r') as f:
        content = f.read()
    
    # Replace the import
    old_import = """# tensor rt converter and inferencer
from accelerations.tensorrt_tools import TensorRTConverter, TensorRTInference"""
    
    new_import = """# tensor rt converter and inferencer - optional
try:
    from tracklib.tracker.accelerations.tensorrt_tools import TensorRTConverter, TensorRTInference
    TENSORRT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # TensorRT is optional - not required for basic DeepSORT functionality
    import warnings
    warnings.filterwarnings("ignore", message="TensorRT not available")
    TensorRTConverter = None
    TensorRTInference = None
    TENSORRT_AVAILABLE = False"""
    
    # Replace in content
    content = content.replace(old_import, new_import)
    
    # Write back
    with open(engine_file, 'w') as f:
        f.write(content)
    
    print(f"✓ Patched {engine_file}")
    print("  TensorRT is now optional - DeepSORT will work without it")

if __name__ == "__main__":
    patch_engine_file()