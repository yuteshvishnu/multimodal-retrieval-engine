"""
Scout Version-12: reuse a recent answer when the same question comes back in other words.

After an answer, its question, confirmed meaning, steps and SQL are kept (the 10 most recent,
least recently used dropped first). When a new question is close to a saved one, the saved
meaning is offered as the first option; if the user picks it, the saved SQL runs again (fresh
rows, no LLM calls) instead of writing steps and SQL.

"Close" is decided by code, not the LLM, and needs all three:
    - the questions' embeddings are similar (cosine >= MIN_SIMILARITY)
    - their key words overlap (>= MIN_WORD_OVERLAP of the shorter question's words)
    - no clash in numbers, months or scope: "March 2026" never matches "April 2026" (their
      embeddings are 0.99 alike), and "for each store" never matches "overall"

Every answer is saved unless it's a follow-up or about a time that moves ("last week",
"today"): an answer nobody reports counts as right. A reported answer is removed, and a
change to any table (pipeline.reload) clears everything.
"""
import re
import threading
import uuid
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

import numpy as np

MIN_SIMILARITY = 0.80
MIN_WORD_OVERLAP = 0.5
# a new option this close to the saved meaning, with the same limiting words, is the same meaning
REPEAT_SIMILARITY = 0.92

MONTHS = {
    "jan": "january", "feb": "february", "mar": "march", "apr": "april", "jun": "june", "jul": "july",
    "aug": "august", "sep": "september", "sept": "september", "oct": "october", "nov": "november", "dec": "december",
}
MONTH_NAMES = set(MONTHS.values()) | {"may"}

# answers that depend on when they're asked: the saved SQL would keep yesterday's dates
MOVING_TIME = re.compile(
    r"\b(today|tonight|yesterday|tomorrow|now|current(ly)?|recent(ly)?|ago|so far|to date|"
    r"(this|last|past|next|previous|coming)\s+(\d+\s+)?(day|week|month|quarter|year)s?)\b",
    re.IGNORECASE,
)

STOPWORDS = set(
    "a an the of in on at to for by from with and or is are was were be been do does did done has have had "
    "how many much what which who whom whose when where why there their them they it its this that these those "
    "each every all any some me my our we us you your i show list give tell find get got please can could would "
    "number count total amount".split()
)


def words(text: str) -> set:
    """The question's key words, lower case and singular: 'Orders placed' -> {'order', 'placed'}."""
    found = set()
    for word in re.findall(r"[a-z]+", text.lower()):
        word = MONTHS.get(word, word)
        if word in STOPWORDS or len(word) < 2:
            continue
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        found.add(word)
    return found


PER_GROUP = re.compile(r"\b(each|every|per)\b", re.IGNORECASE)
OVERALL = re.compile(r"\b(overall|in total|altogether|across all)\b", re.IGNORECASE)


def details(text: str) -> Dict[str, set]:
    """The numbers, months and scope a question names; they must not clash for a match."""
    lowered = text.lower()
    months = {MONTHS.get(w, w) for w in re.findall(r"[a-z]+", lowered)} & MONTH_NAMES
    scope = {"per group"} if PER_GROUP.search(text) else {"overall"} if OVERALL.search(text) else set()
    return {"numbers": set(re.findall(r"\d+(?:\.\d+)?", lowered)), "months": months, "scope": scope}


def clash(a: Dict[str, set], b: Dict[str, set]) -> bool:
    """Both name numbers (or months, or a scope) and they differ."""
    return any(a[k] and b[k] and a[k] != b[k] for k in ("numbers", "months", "scope"))


# words that narrow what's counted: "only orders not cancelled" vs "every order"
LIMITING = re.compile(r"\b(only|not|except|exclud\w*|without|never|unless)\b|n't\b", re.IGNORECASE)


def limits(text: str) -> set:
    """The limiting words in a meaning, folded: 'excluded' -> 'exclud', "aren't" -> 'not'."""
    found = set()
    for m in LIMITING.finditer(text):
        word = m.group(0).lower()
        found.add("not" if word == "n't" else "exclud" if word.startswith("exclud") else word)
    return found


def moves_with_time(*texts: str) -> bool:
    return any(MOVING_TIME.search(t or "") for t in texts)


class AnswerCache:
    def __init__(self, capacity: int, encode: Callable[[str], np.ndarray]):
        self.capacity = capacity
        self.encode = encode
        self._entries: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()

    def _vector(self, text: str) -> np.ndarray:
        v = np.asarray(self.encode(text), dtype="float32")
        return v / (np.linalg.norm(v) + 1e-8)

    def add(
        self, question: str, meaning: str, steps: List[str], sql: str, question_id: Optional[str] = None,
        confidence: Optional[str] = None,
    ) -> Optional[str]:
        """Saves an answer; returns its id, or None when it shouldn't be reused."""
        if not (question.strip() and meaning.strip() and sql.strip()) or moves_with_time(question, meaning):
            return None
        entry = {
            "id": uuid.uuid4().hex[:12], "question": question, "meaning": meaning, "steps": list(steps), "sql": sql, "confidence": confidence,
            "vector": self._vector(question), "words": words(question), "details": details(question),
            "question_ids": {question_id} if question_id else set(),
        }
        with self._lock:
            # the same meaning saved again replaces the older copy
            for key in [k for k, e in self._entries.items() if e["meaning"] == meaning]:
                del self._entries[key]
            self._entries[entry["id"]] = entry
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)  # drop the least recently used
        return entry["id"]

    def match(self, question: str) -> Optional[dict]:
        """The closest saved answer that passes all three checks, or None."""
        with self._lock:
            entries = list(self._entries.values())
        if not entries:
            return None
        vector, key_words, named = self._vector(question), words(question), details(question)
        best, best_score = None, 0.0
        for entry in entries:
            similarity = float(entry["vector"] @ vector)
            shorter = min(len(key_words), len(entry["words"])) or 1
            overlap = len(key_words & entry["words"]) / shorter
            if similarity < MIN_SIMILARITY or overlap < MIN_WORD_OVERLAP or clash(named, entry["details"]):
                continue
            if similarity > best_score:
                best, best_score = entry, similarity
        return best

    def repeats(self, saved: str, option: str) -> bool:
        """
        An option the LLM wrote that only rewords the saved meaning (it's told not to, but may).
        Embeddings alone can't tell: "every order, whatever its status" vs "only orders not
        cancelled" are 0.94 alike, a rewording 0.95. So the limiting words must be the same too.
        """
        if clash(details(saved), details(option)) or limits(saved) != limits(option):
            return False
        return float(self._vector(saved) @ self._vector(option)) >= REPEAT_SIMILARITY

    def get(self, entry_id: str) -> Optional[dict]:
        """A saved answer, now the most recently used."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is not None:
                self._entries.move_to_end(entry_id)
            return entry

    def link(self, entry_id: str, question_id: Optional[str]) -> None:
        """Ties a question answered from this entry to it, so a report on it removes the entry."""
        with self._lock:
            if entry_id in self._entries and question_id:
                self._entries[entry_id]["question_ids"].add(question_id)

    def forget_question(self, question_id: str) -> bool:
        """A reported answer: drop the saved answer it came from."""
        with self._lock:
            keys = [k for k, e in self._entries.items() if question_id in e["question_ids"]]
            for key in keys:
                del self._entries[key]
            return bool(keys)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
