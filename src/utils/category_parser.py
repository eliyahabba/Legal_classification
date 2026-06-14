import re
from typing import Dict, Optional, Set, Tuple

NOT_RELEVANT = "לא רלוונטי"
REASONING_PREFIX = "נימוק:"
CATEGORY_PREFIX = "קטגוריה:"


def build_category_lookup(categories: list) -> tuple[Dict[int, str], Set[str]]:
    id_to_name = {cat["id"]: cat["name"] for cat in categories}
    return id_to_name, set(id_to_name.values())


def _strip_category_number(text: str) -> str:
    return re.sub(r"^\d+\s*:\s*", "", text.strip()).strip()


def _resolve_category_text(
    category_text: str,
    id_to_name: Dict[int, str],
    valid_names: Set[str],
) -> str:
    category_text = category_text.strip().strip('"').strip("'")
    if not category_text:
        return NOT_RELEVANT

    first_line = category_text.splitlines()[0].strip()
    if first_line == NOT_RELEVANT or first_line.startswith(NOT_RELEVANT):
        return NOT_RELEVANT

    match = re.match(r"^(\d+)", first_line)
    if match:
        cat_id = int(match.group(1))
        if cat_id in id_to_name:
            return id_to_name[cat_id]

    name_text = _strip_category_number(first_line)
    if name_text in valid_names:
        return name_text

    for name in valid_names:
        if name in name_text or name_text in name:
            return name

    return name_text


def parse_classification_response(
    raw: str,
    id_to_name: Dict[int, str],
    valid_names: Optional[Set[str]] = None,
) -> Tuple[str, Optional[str]]:
    """Parse model output into canonical category name and optional reasoning."""
    text = str(raw).strip()
    if not text:
        return NOT_RELEVANT, None

    if valid_names is None:
        valid_names = set(id_to_name.values())

    if CATEGORY_PREFIX not in text and REASONING_PREFIX not in text:
        return _resolve_category_text(text, id_to_name, valid_names), None

    reasoning = None
    category_text = text

    if CATEGORY_PREFIX in text:
        reasoning_block, category_text = text.split(CATEGORY_PREFIX, 1)
        category_text = category_text.strip()
        reasoning_block = reasoning_block.strip()
        if REASONING_PREFIX in reasoning_block:
            reasoning = reasoning_block.split(REASONING_PREFIX, 1)[1].strip()
        elif reasoning_block:
            reasoning = reasoning_block
    elif REASONING_PREFIX in text:
        reasoning = text.split(REASONING_PREFIX, 1)[1].strip()
        if CATEGORY_PREFIX in reasoning:
            reasoning, category_text = reasoning.split(CATEGORY_PREFIX, 1)
            reasoning = reasoning.strip()
            category_text = category_text.strip()
        else:
            category_text = NOT_RELEVANT

    reasoning = reasoning.splitlines()[0].strip() if reasoning else None
    category = _resolve_category_text(category_text, id_to_name, valid_names)
    return category, reasoning or None


def parse_category_response(
    raw: str,
    id_to_name: Dict[int, str],
    valid_names: Optional[Set[str]] = None,
) -> str:
    """Resolve a model answer to a canonical category name."""
    category, _ = parse_classification_response(raw, id_to_name, valid_names)
    return category
