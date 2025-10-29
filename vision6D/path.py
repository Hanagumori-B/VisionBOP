import os
import pathlib

# Constants
SAVE_ROOT = pathlib.Path(os.path.abspath(__file__)).parent / "Output"
ICON_PATH = pathlib.Path(os.path.abspath(__file__)).parent / "data" / "icons"
MODEL_PATH = pathlib.Path(os.path.abspath(__file__)).parent / "data" / "model"