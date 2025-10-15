import re
from pathlib import Path

_SHAPE_RE = re.compile(r'[A-Z]|[a-z]|[0-9]')
SPLIT = {'train', 'validation', 'test'}
ROOT_DIR = Path(__file__).parent.parent.parent

LOC_LIST = ["Cairo", "Alexandria", "London", "Paris", "New York"]
ORG_LIST = ["Google", "Microsoft", "OpenAI", "BBC", "UNICEF"]
PER_TITLES = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof."]

PATTERNS = [
    # Money
    {"label": "MONEY", "pattern": [{"TEXT": {"REGEX": r"^\$|€|£"}} , {"LIKE_NUM": True}]},
    {"label": "MONEY", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["usd","eur","egp","sar"]}}]},
    # Percent
    {"label": "PERCENT", "pattern": [{"LIKE_NUM": True}, {"TEXT": {"REGEX": r"^%$"}}]},
    # Dates (simple forms; extend as needed)
    {"label": "DATE", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]}}, {"LIKE_NUM": True}]},
    {"label": "DATE", "pattern": [{"LIKE_NUM": True}, {"TEXT": {"REGEX": r"^[/\-]$"}}, {"LIKE_NUM": True}, {"TEXT": {"REGEX": r"^[/\-]$"}}, {"LIKE_NUM": True}]},
    # Person title + Name (Title + Capitalized token)
    {"label": "PER", "pattern": [{"TEXT": {"IN": PER_TITLES}}, {"IS_TITLE": True}]},
    # ORG with suffixes
    {"label": "ORG", "pattern": [{"IS_TITLE": True, "OP": "+"}, {"LOWER": {"IN": ["inc","ltd","corp","co.","co","plc","llc"]}}]},
    # Emails
    {"label": "EMAIL", "pattern": [{"TEXT": {"REGEX": r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"}}]},
    # URLs
    {"label": "URL", "pattern": [{"TEXT": {"REGEX": r"^https?://\S+$"}}]},
]