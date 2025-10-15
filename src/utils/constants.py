import re
from pathlib import Path

SHAPE_RE = re.compile(r'[A-Z]|[a-z]|[0-9]')
SPLIT = {'train', 'validation', 'test'}
ROOT_DIR = Path(__file__).parent.parent.parent