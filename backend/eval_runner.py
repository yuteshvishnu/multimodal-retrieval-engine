"""
Version-14: run an evaluation from the Evaluation tab and watch it live.

One run at a time, in a background thread, one question at a time, the way the app answers
when nobody steps in ("no user"): the first meaning is taken, the steps are accepted as
written, then the SQL is written, checked and run. Each answer is compared with the correct
SQL's rows (any order), as in BIRD.

    start(suite, first, last)  begins a run on questions first..last (1-based, in the suite's
                         order); its file is saved after every question, so a stopped run keeps
                         what it finished and can be resumed
    progress()           the run in memory: question n of N, its stage, accuracy so far, calls,
                         tokens, cost, any rate-limit wait (the UI polls it)
    stop() / resume()    stop after the current question / carry on where it stopped

Every LLM call is tagged "eval:<run>:<question>", so the run can price each question while
the Dashboard counts it as an outside call, not as a user's question. Eval runs neither use
nor fill the saved-answer cache (Version-12). Wrong answers become Eval issues (Version-11).

Version-18: BIRD Mini-Dev, read from BIRD's own files, left exactly as downloaded
(data/benchmarks/bird/, outside data/text_to_sql/ so normal Scout never sees it). Each question
is answered by a pipeline that has only its own database (made when a run first needs it),
with BIRD's evidence as the hint. The gold SQL is read only to score an answer. Each BIRD run
also writes its SQL in BIRD's official prediction format (runs/bird_predictions/<run>.json), so
BIRD's own evaluation scripts can score the same answers later.
"""
import json
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from backend import usage
from backend.text_to_sql.flows import FlowRunner, adaptive_vote, fit_accuracies, majority_vote, weighted_vote
from backend.config import (
    BIRD_CORRECTED_DIR, BIRD_MINI_DEV_DIR, EVAL_MODELS, MIXED_MODELS, SQL_REFRESHER,
    EVAL_CALLS_PER_QUESTION, EVAL_RATE_LIMIT_TRIES, EVAL_RATE_LIMIT_WAIT_SECONDS, EVAL_SECONDS_PER_QUESTION,
    EVAL_TOKENS_PER_QUESTION, LLM_MODEL_SQL, LLM_PRICES, LLM_PROVIDER, MAX_MEANING_TRIES, PROVIDER_TOKENS_PER_DAY,
    PROVIDERS,
)

RUNS_DIR = Path("data/text_to_sql/eval/runs")
CHECKS_FOLDER = "checks"  # Version-21: one-question checks, under runs/ but never read as runs
# BIRD's official prediction format: {"<question_id>": "<sql>\t----- bird -----\t<db_id>"}
BIRD_SEPARATOR = "\t----- bird -----\t"

# the test suites a run can use; a suite that isn't on disk yet says why
SUITES = {
    "practice": {
        "name": "Practice (retail store)", "short": "Practice",
        "path": Path("data/text_to_sql/retail_store/questions.json"),
        "about": "Our 20 questions with traps (dates, cents, city spellings, ties) on the 10-table practice database.",
    },
    "bird_minidev": {
        "name": "BIRD Mini-Dev", "short": "BIRD",
        "path": BIRD_MINI_DEV_DIR / "mini_dev_sqlite.json",          # BIRD's file, unchanged
        "databases": BIRD_MINI_DEV_DIR / "dev_databases",            # one folder per db_id
        "about": "500 questions from BIRD's dev set across 11 real databases, with BIRD's evidence hints.",
        "missing": "Not downloaded. Unzip BIRD's minidev.zip so data/benchmarks/bird/minidev/MINIDEV/ exists (see README).",
    },
    # Version-21: the same questions with BIRD's annotation errors fixed (questions, hints, correct SQL and
    # 10 column descriptions), by the UIUC Kang lab; the databases are BIRD's own, linked (see README)
    "bird_corrected": {
        "name": "BIRD Mini-Dev, corrected", "short": "BIRD fixed",
        "path": BIRD_CORRECTED_DIR / "arcwise_plat_full_with_diff.json",  # their file, unchanged
        "databases": BIRD_CORRECTED_DIR / "dev_databases",                  # BIRD's databases + fixed descriptions
        "difficulty_from": BIRD_MINI_DEV_DIR / "mini_dev_sqlite.json",      # their file has no difficulty
        "about": "BIRD Mini-Dev's 498 questions with annotation errors fixed (Arcwise-Plat, UIUC Kang lab, PVLDB 19): "
                 "wrong correct-SQL, unclear questions and hints, column descriptions.",
        "missing": "Not downloaded. See README, Version-21 (BIRD Mini-Dev, corrected).",
    },
}

# the practice questions use easy / medium / hard; BIRD uses simple / moderate / challenging
DIFFICULTY = {"easy": "simple", "medium": "moderate", "hard": "challenging"}


def as_set(rows) -> set:
    """Rows in any order, repeats ignored, values compared exactly: BIRD's own scorer (evaluation_ex.py,
    set(predicted) == set(gold)). Exact also means 66.62300 isn't 66.62301."""
    return {tuple(row) for row in rows}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_questions(suite: str) -> List[Dict]:
    """The suite's questions in its own order, each with "n" (1-based position) and its database."""
    info = SUITES[suite]
    raw = json.loads(info["path"].read_text(encoding="utf-8"))
    if "databases" not in info:
        return [{**q, "n": i, "db_id": q.get("db_id", "retail_store")} for i, q in enumerate(raw, 1)]
    difficulty = {}
    if info.get("difficulty_from") and info["difficulty_from"].exists():
        difficulty = {str(q["question_id"]): q["difficulty"]
                      for q in json.loads(info["difficulty_from"].read_text(encoding="utf-8"))}
    # BIRD's fields -> ours; the gold SQL stays out of everything Scout is given
    return [{
        "n": i, "id": int(q["question_id"]), "db_id": q["db_id"], "question": q["question"],
        "evidence": q.get("evidence") or "", "gold_sql": q["SQL"],
        "difficulty": q.get("difficulty") or difficulty.get(str(q["question_id"]), ""),
    } for i, q in enumerate(raw, 1)]


def is_benchmark(suite: str) -> bool:
    """A public benchmark (BIRD): its issues may be investigated, but fixes from them are never applied."""
    return bool(SUITES.get(suite, {}).get("databases"))


def pick(questions: List[Dict], first: int = 1, last: int = 0) -> List[Dict]:
    """Questions first..last (1-based, both included); last = 0 means to the end."""
    if first < 1 or (last and last < first):
        raise ValueError("Pick a range like 1 to 50: the first question can't come after the last.")
    if first > len(questions):
        raise ValueError(f"This suite has only {len(questions)} questions.")
    return questions[first - 1:last or None]


def suites() -> List[Dict]:
    out = []
    for key, info in SUITES.items():
        ready = info["path"].exists()
        out.append({
            "id": key, "name": info["name"], "about": info["about"], "ready": ready,
            "questions": len(load_questions(key)) if ready else 0,
            "missing": None if ready else info.get("missing", "Not available."),
        })
    return out


def models() -> List[Dict]:
    """The models an evaluation can use with this provider, smallest first, each with its price."""
    out = []
    for model in EVAL_MODELS:
        if LLM_PROVIDER != "openrouter" and model["openrouter_only"]:
            continue
        price_in, price_out = LLM_PRICES.get(model["id"], (0, 0))
        out.append({"id": model["id"], "name": model["name"], "host": model["host"],
                    "price_in": price_in, "price_out": price_out})
    return out


def check_model(model: str) -> None:
    if model not in {m["id"] for m in models()}:
        raise ValueError(f"Unknown model: {model}.")


def model_of(run: Dict) -> str:
    return run.get("model") or LLM_MODEL_SQL


def estimate(questions: int, run_flows: Optional[List[str]] = None, model: str = LLM_MODEL_SQL) -> Dict:
    """What a run of this many questions should take, from measured averages (config.py); with
    several flows (a comparison), each flow's run, scaled by its LLM calls per question."""
    run_flows = run_flows or [DEFAULT_FLOW]
    scale = sum(FLOWS[f]["calls"] for f in run_flows) / EVAL_CALLS_PER_QUESTION
    tokens = round(questions * EVAL_TOKENS_PER_QUESTION * scale)
    cost = 0.0
    for flow in run_flows:
        share = tokens * FLOWS[flow]["calls"] / sum(FLOWS[f]["calls"] for f in run_flows)
        priced = [m["id"] for m in MIXED_MODELS] if flow == "mixed" else [model]  # each model does its share
        for name in priced:
            price_in, price_out = LLM_PRICES.get(name, (0, 0))
            cost += share / len(priced) * (0.75 * price_in + 0.25 * price_out) / 1e6  # about 3/4 of tokens are input
    return {
        "questions": questions, "llm_calls": round(questions * EVAL_CALLS_PER_QUESTION * scale), "tokens": tokens,
        "cost": round(cost, 4), "minutes": round(questions * EVAL_SECONDS_PER_QUESTION * scale / 60, 1),
        "over_daily_limit": bool(PROVIDER_TOKENS_PER_DAY) and tokens > PROVIDER_TOKENS_PER_DAY,
        "daily_limit": PROVIDER_TOKENS_PER_DAY, "provider": PROVIDERS[LLM_PROVIDER]["name"],
    }


# Version-22: how Scout answers a question. Each run records its flow, so flows can be compared on
# the same questions (Evaluation -> Comparison). Full and Adaptive follow "Ask Me Anything" (Arora, ...,
# Sala, Re): different routes to the SQL and a weighted vote instead of three tries from one prompt.
FLOWS = {
    "original": {
        "name": "Original", "ready": True, "calls": EVAL_CALLS_PER_QUESTION,
        "about": "One meaning, steps, 3 SQL tries from the same prompt, a retry round if they disagree, majority vote.",
    },
    "full": {
        "name": "Full (paper)", "ready": True, "calls": 8,
        "about": "4 prompt chains (each rewrites the question its own way, then writes SQL, with its own worked "
                 "examples), majority vote, no retry. 8 calls.",
    },
    "adaptive": {
        "name": "Adaptive", "ready": True, "calls": 4.5,  # 4 when the first 2 chains agree, 8 when not
        "about": "Our idea, not the paper's: the first 2 chains agree on an answer with data, stop at 4 calls; if not, "
                 "all 4 and a vote (8). Rejected SQL gets one retry. A Full run also gives a result, without the retries.",
    },
}
FLOWS["mixed"] = {
    "name": "Mixed + weak supervision", "ready": LLM_PROVIDER == "openrouter", "calls": 6,
    "note": "needs OpenRouter",
    "about": "Our extension: 3 different models answer each question, with rewrite styles and SQL methods shuffled "
             "among them; learned trust picks the answer. Uses its own models, whichever is chosen above. 6 calls.",
}
FLOWS["adaptive_english"] = {  # Version-24
    "name": "Adaptive + English program", "ready": True, "calls": 4.5,
    "about": "Adaptive, but the model never writes SQL: it writes a sentence in the strict English format, and the "
             "english2sqlite program turns it into SQL, with no AI. A rejected sentence gets one retry. 4 to 12 calls.",
}
DEFAULT_FLOW = "original"
FLOW_MODES = {  # what the run did, shown on its scorecard
    "original": "No user: the first meaning and the steps are accepted as written",
    "full": "No user: 4 prompt chains and a majority vote, no retry",
    "adaptive": "No user: 2 prompt chains, the other 2 and a majority vote only if needed, one retry for rejected SQL",
    "adaptive_english": "No user: Adaptive, with the model writing format sentences and english2sqlite writing the SQL",
    "mixed": "No user: 3 different models, shuffled styles and SQL methods, no retry. This score is the live majority "
             "vote; the Comparison ring adds weak supervision",
}
COMPARISONS_FOLDER = "comparisons"  # Version-22: run queues, under runs/ but never read as runs


def flows() -> List[Dict]:
    return [{"id": key, **info} for key, info in FLOWS.items()]


def flow_of(run: Dict) -> str:
    """Runs from before Version-22 have no flow: they all used the original one."""
    return run.get("flow") or DEFAULT_FLOW


def check_flow(flow: str) -> None:
    if flow not in FLOWS:
        raise ValueError(f"Unknown flow: {flow}.")
    if not FLOWS[flow]["ready"]:
        raise ValueError(f"The {FLOWS[flow]['name']} flow has no run of its own: {FLOWS[flow].get('note', 'not built yet')}.")


def is_rate_limited(error: str) -> bool:
    text = (error or "").lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


class EvalRunner:
    def __init__(self, pipeline, runs_dir: Path = RUNS_DIR, sleep: Callable[[float], None] = time.sleep,
                 pipeline_for: Optional[Callable[[Path], object]] = None,
                 practice_for: Optional[Callable[[], object]] = None):
        """
        pipeline: the app's Scout, for the practice suite.
        pipeline_for(folder): makes a Scout over just that folder's database (BIRD: one per db_id).
        practice_for(): makes another Scout like the app's, for a practice run on another model
        (the app's own Scout never changes model, so a run can't affect questions asked meanwhile).
        """
        self.pipeline = pipeline
        self.pipeline_for = pipeline_for
        self.practice_for = practice_for
        self._pipelines: Dict[str, object] = {}  # db_id -> its pipeline, made when first needed
        self.runs_dir = runs_dir
        self.sleep = sleep  # tests pass a no-op
        self.run: Optional[Dict] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.queue: Optional[Dict] = None  # Version-22: the comparison being run (flows one after another)
        self._queue_thread: Optional[threading.Thread] = None
        self._queue_stop = False

    # ---------- control ----------

    def busy(self) -> bool:
        return (self._thread is not None and self._thread.is_alive()) or self._queue_busy()

    def _queue_busy(self) -> bool:
        return self._queue_thread is not None and self._queue_thread.is_alive()

    def _run_busy(self, queued: bool) -> bool:
        """Busy for a new run: the queue starts its own runs, so it only waits for the run thread."""
        return (self._thread is not None and self._thread.is_alive()) or (not queued and self._queue_busy())

    def start(self, suite: str, first: int = 1, last: int = 0, wait: bool = False,
              flow: str = DEFAULT_FLOW, _queued: bool = False, model: str = LLM_MODEL_SQL) -> Dict:
        if self._run_busy(_queued):
            raise RuntimeError("An evaluation is already running.")
        check_flow(flow)
        check_model(model)
        if flow == "mixed":  # it picks its own models; the run is filed under the default
            model = LLM_MODEL_SQL
        if suite not in SUITES or not SUITES[suite]["path"].exists():
            raise ValueError("That test suite isn't available.")
        questions = pick(load_questions(suite), first, last)
        info = SUITES[suite]
        self.run = {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
            "suite": suite, "benchmark": info["name"], "short": info["short"], "sample": False, "flow": flow,
            "status": "running", "message": None, "run_at": now(), "finished_at": None,
            "model": model, "provider": PROVIDERS[LLM_PROVIDER]["name"], "mode": FLOW_MODES[flow],
            "sql_refresher": SQL_REFRESHER,  # Version-23: so runs with and without it can be told apart
            **({"models": [m["id"] for m in MIXED_MODELS]} if flow == "mixed" else {}),
            "planned": [q["id"] for q in questions], "range": [questions[0]["n"], questions[-1]["n"]],
            "hints": "BIRD's evidence" if "databases" in info else "none", "current": None, "wait_until": None,
            "elapsed_seconds": 0, "results": [],
        }
        self._save()
        return self._launch(questions, wait)

    def check(self, suite: str, question_id: int, wait: bool = False) -> Dict:
        """
        Version-21: one question again, to see if a fix worked (started from its issue). Saved in
        runs/checks/, apart from the runs: it never changes the Evaluation tab's scores, the run
        list, BIRD's prediction files, or the Issues list (a wrong answer isn't logged again).
        """
        if self.busy():
            raise RuntimeError("An evaluation is already running. Try again when it finishes.")
        if suite not in SUITES or not SUITES[suite]["path"].exists():
            raise ValueError("That test suite isn't available.")
        q = next((q for q in load_questions(suite) if str(q["id"]) == str(question_id)), None)
        if q is None:
            raise ValueError(f"Question {question_id} isn't in that suite.")
        info = SUITES[suite]
        self.run = {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
            "check": True, "question_id": q["id"], "flow": DEFAULT_FLOW,
            "suite": suite, "benchmark": info["name"], "short": info["short"], "sample": False,
            "status": "running", "message": None, "run_at": now(), "finished_at": None,
            "model": LLM_MODEL_SQL, "provider": PROVIDERS[LLM_PROVIDER]["name"],
            "mode": "No user: the first meaning and the steps are accepted as written",
            "planned": [q["id"]], "range": [q["n"], q["n"]],
            "hints": "BIRD's evidence" if "databases" in info else "none", "current": None, "wait_until": None,
            "elapsed_seconds": 0, "results": [],
        }
        self._save()
        return self._launch([q], wait)

    def checks(self, question_id) -> List[Dict]:
        """Every check of one question, newest first."""
        folder = self.runs_dir / CHECKS_FOLDER
        out = []
        for path in sorted(folder.glob("*.json"), reverse=True) if folder.exists() else []:
            run = json.loads(path.read_text(encoding="utf-8"))
            if str(run.get("question_id")) == str(question_id):
                out.append(run)
        return out

    def resume(self, run_id: str, wait: bool = False, _queued: bool = False) -> Dict:
        if self._run_busy(_queued):
            raise RuntimeError("An evaluation is already running.")
        run = self.load(run_id)
        if run is None or run["status"] == "finished":
            raise ValueError("Only a stopped run can be resumed.")
        if flow_of(run) == "mixed" and run.get("models") != [m["id"] for m in MIXED_MODELS]:
            raise ValueError("This Mixed run used other models, so it can't be resumed: start a new one.")
        done = {r["id"] for r in run["results"]}
        questions = [q for q in load_questions(run["suite"]) if q["id"] in run["planned"] and q["id"] not in done]
        self.run = {**run, "status": "running", "message": None, "finished_at": None}
        self._save()
        return self._launch(questions, wait)

    def stop(self) -> None:
        """Stops after the question being answered (a question is never cut in half); a comparison
        being run stops there too, and can be resumed."""
        self._stop.set()
        self._queue_stop = True

    def _launch(self, questions: List[Dict], wait: bool) -> Dict:
        self._stop.clear()
        self._thread = threading.Thread(target=self._work, args=(questions,), daemon=True, name="eval-run")
        self._thread.start()
        if wait:  # tests
            self._thread.join()
        return self.progress()

    # ---------- Version-22: compare flows on the same questions ----------

    def _saved_runs(self) -> List[Dict]:
        out = []
        for path in sorted(self.runs_dir.glob("*.json")) if self.runs_dir.exists() else []:
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def compare(self, suite: str, first: int = 1, last: int = 0, model: str = LLM_MODEL_SQL) -> Dict:
        """
        Each flow's score on the suite's questions first..last (last = 0: to the end), from its newest
        run that answered every one of them; a longer run counts too. Questions are always the same
        set, so the rings can be compared. Full's saved chains also give, with no LLM calls, a
        weak-supervision vote, 3 chains instead of 4, and Adaptive. Only runs on this model count.
        """
        if suite not in SUITES or not SUITES[suite]["path"].exists():
            raise ValueError("That test suite isn't available.")
        questions = load_questions(suite)
        chosen = pick(questions, first, last)
        ids = [q["id"] for q in chosen]
        best: Dict[str, Dict] = {}
        for run in self._saved_runs():
            if run.get("suite") != suite or run.get("sample"):
                continue
            if model_of(run) != model and flow_of(run) != "mixed":  # Mixed has its own models: always shown
                continue
            answered = {r["id"]: r for r in run.get("results", [])}
            if not set(ids) <= set(answered):
                continue
            flow = flow_of(run)
            if flow not in best or run["run_at"] > best[flow]["run_at"]:
                best[flow] = {**run, "_answers": answered}

        def results_of(run):
            return [run["_answers"][i] for i in ids]

        out, marks = [], {}

        def add(key, name, about, ready, run, answers, confidence, calls, cost, **extra):
            """answers: id -> right or wrong, in the order of the questions."""
            entry = {"flow": key, "name": name, "about": about, "ready": ready, "run": None, **extra}
            if run:
                right = sum(answers.values())
                entry["run"] = {
                    "id": run["id"], "run_at": run["run_at"], "model": run.get("model"), "status": run["status"],
                    "correct": right, "total": len(ids), "accuracy": round(100 * right / len(ids), 1),
                    "llm_calls": calls, "calls_per_question": round(calls / len(ids), 1), "cost": round(cost, 5),
                    # the problem the paper targets: every chain agreed, and they were wrong
                    "confident_wrong": sum(1 for i in ids if not answers[i] and confidence[i] == "high"),
                    "confident": sum(1 for i in ids if confidence[i] == "high"),
                    "confident_right": sum(1 for i in ids if answers[i] and confidence[i] == "high"),
                    "older": bool(extra.pop("older", False)),
                }
                for i, ok in answers.items():
                    marks.setdefault(i, {})[key] = ok
            out.append(entry)

        def plain(key, info):
            run = best.get(key)
            if not run:
                return add(key, info["name"], info["about"], info["ready"], None, {}, {}, 0, 0)
            results = results_of(run)
            has_chains = all(r.get("chains") for r in results)
            extra = {"older": key != DEFAULT_FLOW and not has_chains}
            if has_chains:
                wrong = [r for r in results if not r["correct"]]
                # was a right answer among the tries, even though another was picked?
                extra["misses"] = len(wrong)
                extra["right_was_there"] = sum(1 for r in wrong if any(c.get("correct") for c in r["chains"]))
                if key in ("adaptive", "adaptive_english"):
                    extra["stopped_early"] = sum(1 for r in results if len(r["chains"]) == 2)
            add(key, info["name"], info["about"], info["ready"], run,
                {r["id"]: r["correct"] for r in results}, {r["id"]: r.get("confidence") for r in results},
                sum(r.get("llm_calls", 0) for r in results), sum(r.get("cost", 0) for r in results), **extra)

        plain(DEFAULT_FLOW, FLOWS[DEFAULT_FLOW])
        plain("full", FLOWS["full"])
        full = best.get("full")
        results = results_of(full) if full else []
        replayable = bool(results) and all(r.get("chains") for r in results)
        if replayable:
            entry = out[-1]
            per_chain = {}
            for r in results:
                for c in r["chains"]:
                    per_chain[c["chain"]] = per_chain.get(c["chain"], 0) + bool(c.get("correct"))
            entry["chains"] = [{"chain": c, "correct": n} for c, n in per_chain.items()]
            entry["average_chain"] = round(100 * sum(per_chain.values()) / (len(per_chain) * len(ids)), 1) if per_chain else None
            three = [majority_vote(r["chains"], 3) for r in results]
            entry["three_chains"] = sum(1 for d in three if d and d["winner"].get("correct"))

            accuracies = fit_accuracies([r["chains"] for r in results])
            votes = {r["id"]: weighted_vote(r["chains"], accuracies) for r in results}
            add("full_ws", "Full + weak supervision",
                "The same answers as Full, with each chain's vote weighted by the accuracy learned from how often "
                "the chains agree, never from correct answers.", True, full,
                {i: bool(d and d["winner"].get("correct")) for i, d in votes.items()},
                {i: (d["confidence"] if d else None) for i, d in votes.items()},
                sum(r.get("llm_calls", 0) for r in results), sum(r.get("cost", 0) for r in results),
                replayed=True, accuracies={c: round(a, 3) for c, a in accuracies.items()} if accuracies else None)

        # Adaptive: its own live run if that's newer, else replayed from Full's saved chains
        live_adaptive = best.get("adaptive")
        live_is_new = live_adaptive and all(live_adaptive["_answers"][i].get("chains") for i in ids)
        if replayable and not (live_is_new and live_adaptive["run_at"] > full["run_at"]):
            votes = {r["id"]: adaptive_vote(r["chains"]) for r in results}
            calls = {r["id"]: (votes[r["id"]] or {}).get("calls", 8) for r in results}
            add("adaptive", FLOWS["adaptive"]["name"], FLOWS["adaptive"]["about"], True, full,
                {i: bool(d and d["winner"].get("correct")) for i, d in votes.items()},
                {i: (d["confidence"] if d else None) for i, d in votes.items()},
                sum(calls.values()),
                sum(r.get("cost", 0) * calls[r["id"]] / max(r.get("llm_calls", 0), 1) for r in results),
                replayed=True, stopped_early=sum(1 for n in calls.values() if n == 4))
        else:
            plain("adaptive", FLOWS["adaptive"])

        plain("adaptive_english", FLOWS["adaptive_english"])

        # Mixed: weak supervision over its 4 models, worked out from its saved answers, with the majority beside it
        mixed = best.get("mixed")
        results = results_of(mixed) if mixed else []
        info = FLOWS["mixed"]
        if results and all(r.get("chains") for r in results):
            accuracies = fit_accuracies([r["chains"] for r in results])
            votes = {r["id"]: weighted_vote(r["chains"], accuracies) for r in results}
            answers = {i: bool(d and d["winner"].get("correct")) for i, d in votes.items()}
            per_chain = {}
            for r in results:
                for c in r["chains"]:
                    per_chain[c["chain"]] = per_chain.get(c["chain"], 0) + bool(c.get("correct"))
            add("mixed", info["name"], info["about"], True, {**mixed, "model": f"{len(per_chain)} models"}, answers,
                {i: (d["confidence"] if d else None) for i, d in votes.items()},
                sum(r.get("llm_calls", 0) for r in results), sum(r.get("cost", 0) for r in results),
                accuracies={c: round(a, 3) for c, a in accuracies.items()} if accuracies else None,
                majority=sum(1 for r in results if r["correct"]),
                chains=[{"chain": c, "correct": n} for c, n in per_chain.items()],
                average_chain=round(100 * sum(per_chain.values()) / (len(per_chain) * len(ids)), 1) if per_chain else None,
                misses=sum(1 for i in ids if not answers[i]),
                right_was_there=sum(1 for r in results if not answers[r["id"]] and any(c.get("correct") for c in r["chains"])))
        else:
            add("mixed", info["name"], info["about"], info["ready"], None, {}, {}, 0, 0)

        compared = [e["flow"] for e in out if e["run"]]
        differ = []
        for q in chosen:
            answers = marks.get(q["id"], {})
            if len(set(answers.values())) > 1:
                differ.append({"id": q["id"], "n": q["n"], "question": q["question"], "db_id": q["db_id"],
                               "difficulty": q.get("difficulty"), "right": answers})
        return {"suite": suite, "model": model, "first": chosen[0]["n"], "last": chosen[-1]["n"], "questions": len(ids),
                "of": len(questions), "flows": out, "compared": compared, "differ": differ}

    def start_comparison(self, suite: str, first: int = 1, last: int = 0, chosen_flows: Optional[List[str]] = None,
                         reuse: bool = True, wait: bool = False, model: str = LLM_MODEL_SQL) -> Dict:
        """
        Runs the chosen flows on the same questions, one after another (in parallel they would hit
        the provider's rate limits). With reuse, a flow that already has a run covering those
        questions isn't run again.
        """
        if self.busy():
            raise RuntimeError("An evaluation is already running.")
        if not chosen_flows:
            raise ValueError("Pick at least one flow.")
        for flow in chosen_flows:
            check_flow(flow)
        check_model(model)
        compared = self.compare(suite, first, last, model)
        current = {e["flow"]: e["run"] for e in compared["flows"]}
        items = []
        for flow in chosen_flows:
            # an older Full (before the 4 chains) doesn't count as a run of today's Full
            if reuse and current.get(flow) and not current[flow].get("older"):
                items.append({"flow": flow, "status": "reused", "run_id": current[flow]["id"]})
            else:
                items.append({"flow": flow, "status": "waiting", "run_id": None})
        self.queue = {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
            "suite": suite, "model": model, "first": compared["first"], "last": compared["last"], "status": "running",
            "message": None,
            "started_at": now(), "finished_at": None, "items": items,
        }
        self._save_queue()
        return self._launch_queue(wait)

    def resume_comparison(self, wait: bool = False) -> Dict:
        if self.busy():
            raise RuntimeError("An evaluation is already running.")
        if not self.queue:
            self.queue = self._newest_queue()
        if not self.queue or self.queue["status"] not in ("stopped", "failed"):
            raise ValueError("Only a stopped comparison can be resumed.")
        self.queue.update(status="running", message=None, finished_at=None)
        self._save_queue()
        return self._launch_queue(wait)

    def comparison(self) -> Optional[Dict]:
        """The comparison being run, or the newest one, with each run's progress."""
        queue = self.queue or self._newest_queue()
        if queue is None:
            return None
        queue = json.loads(json.dumps(queue))
        live = self.progress() if self.busy() else None
        for item in queue["items"]:
            if item["status"] == "running" and live and live.get("id") == item["run_id"]:
                item["done"], item["total"], item["correct"] = live["done"], live["total"], live["correct"]
            elif item["run_id"]:
                run = self.load(item["run_id"])
                if run:
                    results = run.get("results", [])
                    item["done"], item["total"] = len(results), len(run.get("planned", results))
                    item["correct"] = sum(1 for r in results if r["correct"])
        return queue

    def _launch_queue(self, wait: bool) -> Dict:
        self._queue_stop = False
        self._queue_thread = threading.Thread(target=self._queue_work, daemon=True, name="eval-comparison")
        self._queue_thread.start()
        if wait:  # tests
            self._queue_thread.join()
        return self.comparison()

    def _queue_work(self) -> None:
        queue = self.queue
        try:
            for item in queue["items"]:
                if item["status"] in ("done", "reused"):
                    continue
                if self._queue_stop:
                    break
                with self._lock:
                    item["status"] = "running"
                self._save_queue()
                if item["run_id"]:  # stopped earlier: carry on from where it stopped
                    self.resume(item["run_id"], _queued=True)
                else:
                    # the run's id is known as soon as it starts, so the tab can show it live
                    started = self.start(queue["suite"], queue["first"], queue["last"], flow=item["flow"], _queued=True,
                                         model=queue.get("model", LLM_MODEL_SQL))
                    with self._lock:
                        item["run_id"] = started["id"]
                self._save_queue()
                self._thread.join()  # wait for this flow's run to end
                status = (self.load(item["run_id"]) or {}).get("status")
                with self._lock:
                    item["status"] = "done" if status == "finished" else (status or "failed")
                self._save_queue()
                if status != "finished":
                    break
            pending = [i for i in queue["items"] if i["status"] not in ("done", "reused")]
            with self._lock:
                queue.update(status="stopped" if pending else "finished", finished_at=now(),
                             message="Stopped. Resume to carry on." if pending else None)
        except Exception as e:  # never leave a comparison looking alive
            with self._lock:
                queue.update(status="failed", finished_at=now(), message=f"The comparison stopped on an error: {e}")
        self._save_queue()

    def _save_queue(self) -> None:
        with self._lock:
            text = json.dumps(self.queue, indent=1)
            queue_id = self.queue["id"]
        folder = self.runs_dir / COMPARISONS_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{queue_id}.json").write_text(text + "\n", encoding="utf-8")

    def _newest_queue(self) -> Optional[Dict]:
        folder = self.runs_dir / COMPARISONS_FOLDER
        files = sorted(folder.glob("*.json")) if folder.exists() else []
        queue = json.loads(files[-1].read_text(encoding="utf-8")) if files else None
        if queue and queue["status"] == "running":  # the server restarted while it ran
            queue.update(status="stopped", message="Stopped when the server restarted. Resume to carry on.")
            for item in queue["items"]:
                if item["status"] == "running":
                    item["status"] = "stopped"
        return queue

    # ---------- reading ----------

    def progress(self) -> Optional[Dict]:
        """The run in memory, or the newest saved one, with its totals so far."""
        with self._lock:
            run = json.loads(json.dumps(self.run)) if self.run else None
        if run is None:
            run = self.newest()
        if run is None:
            return None
        results = run["results"]
        right = sum(1 for r in results if r["correct"])
        calls = sum(r.get("llm_calls", 0) for r in results)
        return {
            **{k: v for k, v in run.items() if k != "results"},
            "total": len(run["planned"]), "done": len(results), "correct": right,
            "accuracy": round(100 * right / len(results), 1) if results else None,
            "llm_calls": calls, "tokens": sum(r.get("tokens", 0) for r in results),
            "cost": round(sum(r.get("cost", 0) for r in results), 5),
            "seconds_per_question": round(run["elapsed_seconds"] / len(results), 1) if results else None,
            "recent": [
                {k: r.get(k) for k in ("id", "question", "difficulty", "correct", "outcome", "llm_calls", "cost", "seconds")}
                for r in results[-8:][::-1]
            ],
        }

    def finished_runs(self) -> List[Dict]:
        """Runs that ended with answers, newest first: what the agent can pick from (Evaluation -> Issues)."""
        runs = []
        for path in sorted(self.runs_dir.glob("*.json"), reverse=True) if self.runs_dir.exists() else []:
            run = json.loads(path.read_text(encoding="utf-8"))
            if run.get("status") == "running" or not run.get("results"):
                continue
            run["_saved"] = path.stat().st_mtime  # breaks ties between runs started in the same second
            results = run["results"]
            wrong = [r["question"] for r in results if not r["correct"]]
            # runs from before issues had run ids: their issues move under the run, once
            usage.usage_store.link_issues_to_run(run["id"], wrong, run["run_at"])
            right = len(results) - len(wrong)
            runs.append({
                "id": run["id"], "suite": run.get("suite"), "benchmark": run.get("benchmark"), "short": run.get("short"),
                "status": run["status"], "run_at": run["run_at"], "finished_at": run.get("finished_at"),
                "done": len(results), "total": len(run.get("planned", results)), "correct": right,
                "wrong": len(wrong), "accuracy": round(100 * right / len(results), 1), "_saved": run["_saved"],
            })
        runs.sort(key=lambda r: (r["run_at"], r.pop("_saved")), reverse=True)
        return runs

    def newest(self) -> Optional[Dict]:
        files = sorted(self.runs_dir.glob("*.json")) if self.runs_dir.exists() else []
        return json.loads(files[-1].read_text(encoding="utf-8")) if files else None

    def load(self, run_id: str) -> Optional[Dict]:
        path = self.runs_dir / f"{run_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    # ---------- the work ----------

    def _work(self, questions: List[Dict]) -> None:
        started = time.monotonic() - self.run["elapsed_seconds"]
        try:
            for q in questions:
                if self._stop.is_set():
                    self._finish("stopped", "Stopped. Resume to carry on from the next question.")
                    return
                result = self._answer(q)
                if result is None:  # the provider's limit didn't lift
                    self._finish("stopped", "Stopped: the LLM provider's rate limit didn't lift after "
                                            f"{EVAL_RATE_LIMIT_TRIES} tries. Resume later (a daily limit resets the next day).")
                    return
                with self._lock:
                    self.run["results"].append(result)
                    self.run["current"] = None
                    self.run["elapsed_seconds"] = round(time.monotonic() - started, 1)
                self._save()
            self._finish("finished", None)
        except Exception as e:  # never leave a run looking alive
            self._finish("failed", f"The run stopped on an error: {e}")

    def _answer(self, q: Dict) -> Optional[Dict]:
        """One question in the app's flow; retries when the provider's rate limit is hit."""
        for attempt in range(EVAL_RATE_LIMIT_TRIES):
            tag = f"eval:{self.run['id']}:{q['id']}:{attempt}"
            began = time.monotonic()
            try:
                result, meaning, steps = self._flow(q, tag)
            except Exception as e:  # a bug in Scout fails this question, never the whole run (Version-21)
                print(f"[Eval] {q['id']}: Scout stopped on an error: {e!r}")
                result, meaning, steps = {"error": f"Scout stopped on an error: {e}"}, "", []
            # a 429 shows in the error (meaning, steps) or only in the SQL tries' reasons
            if not is_rate_limited(f"{result.get('error') or ''} {result.get('failure_reason') or ''}"):
                return self._score(q, tag, result, meaning, steps, time.monotonic() - began)
            with self._lock:
                self.run["wait_until"] = time.time() + EVAL_RATE_LIMIT_WAIT_SECONDS
                self.run["current"] = {**(self.run["current"] or {}), "stage": "waiting"}
            self.sleep(EVAL_RATE_LIMIT_WAIT_SECONDS)
            with self._lock:
                self.run["wait_until"] = None
        return None

    def _stage(self, q: Dict, stage: str) -> None:
        with self._lock:
            current = self.run["current"] or {}
            began = current.get("started") if current.get("id") == q["id"] else None
            self.run["current"] = {"id": q["id"], "question": q["question"], "stage": stage,
                                   "index": len(self.run["results"]) + 1, "started": began or time.time()}

    def pipeline_for_db(self, suite: str, db_id: str, model: str = LLM_MODEL_SQL):
        """The app's Scout, or for BIRD a Scout that has only this database (also used by the agent).
        Each model gets its own Scouts, so a run never changes the model of another."""
        databases = SUITES.get(suite, {}).get("databases")
        if not databases and model == LLM_MODEL_SQL:
            return self.pipeline
        with self._lock:
            key = (suite, db_id, model)  # BIRD and corrected BIRD share database names, not descriptions
            if key not in self._pipelines:
                if databases:
                    folder = databases / db_id
                    if not folder.is_dir() or self.pipeline_for is None:
                        raise RuntimeError(f"BIRD database {db_id} isn't available.")
                    scout = self.pipeline_for(folder)
                elif self.practice_for is not None:
                    scout = self.practice_for()
                else:
                    raise RuntimeError("This suite can't run on another model.")
                if model != LLM_MODEL_SQL:  # a new Scout already uses the default
                    for part in (scout.generator, scout.interpreter, scout.planner):
                        part.model_name = model
                self._pipelines[key] = scout
            return self._pipelines[key]

    def _pipeline(self, q: Dict):
        return self.pipeline_for_db(self.run["suite"], q["db_id"], model_of(self.run))

    def _flow(self, q: Dict, tag: str):
        """
        meaning (first option) -> steps (as written) -> SQL; returns (result, meaning, steps).
        If the checks find the meaning asks for a value the data doesn't have, the meaning is asked
        again with the real values, as the app does (at most MAX_MEANING_TRIES meanings in all).
        """
        question = q["question"] + (f"\n(Hint: {q['evidence']})" if q.get("evidence") else "")
        pipeline = self._pipeline(q)
        flow = flow_of(self.run)
        if flow != DEFAULT_FLOW:  # Version-22: Full or Adaptive (backend/text_to_sql/flows.py)
            self._stage(q, "sql")
            result = FlowRunner(pipeline).run(question, tag, flow)
            meaning = "\n".join(f"{a['style']}: {a['meaning']}" for a in result.get("attempts") or [])
            return result, meaning, result.get("steps") or []
        history, result, meaning, steps = [], {}, "", []
        for _ in range(MAX_MEANING_TRIES):
            self._stage(q, "meaning")
            interpreted = pipeline.interpret(question, history=history, question_id=tag, use_saved=False)
            if interpreted.get("error") or not interpreted.get("options"):
                return {"error": interpreted.get("error") or "No meaning was offered."}, meaning, steps
            first = interpreted["options"][0]
            meaning = "\n".join([first["meaning"]] + [f"- {c}" for c in first["choices"]])

            self._stage(q, "steps")
            planned = pipeline.plan(question, meaning, question_id=tag)
            if planned.get("error"):
                return {"error": planned["error"]}, meaning, []
            steps = planned["steps"]

            self._stage(q, "sql")
            result = pipeline.run(question, meaning, question_id=tag, steps=steps, save=False)
            if not result.get("rethink_meaning"):
                break
            history.append({"options": interpreted["options"], "comment": result["rethink_meaning"]})
        return result, meaning, steps

    def _score(self, q: Dict, tag: str, result: Dict, meaning: str, steps: List[str], seconds: float) -> Dict:
        # the gold SQL is read only here, after Scout has answered
        pipeline = self._pipeline(q)
        # both answers are compared whole, as BIRD's evaluation does: the app keeps only the first
        # 200 rows, and an answer without DISTINCT can show only some of its values there (Version-21)
        _, gold_rows = pipeline.database.run_all(q["gold_sql"])
        error = result.get("error")
        rows = []
        if not error and result.get("sql"):
            try:
                _, rows = pipeline.database.run_all(result["sql"])
            except Exception as e:  # ran in the app, but not again whole: fall back to what was kept
                print(f"[Eval] {q['id']}: scoring on the kept rows ({e})")
                saved = pipeline.results.get(result["result_id"]) if result.get("result_id") else None
                rows = saved["rows"] if saved else result.get("rows", [])
        correct = not error and as_set(rows) == as_set(gold_rows)
        if correct:
            outcome = "right"
        elif not error:
            outcome = "wrong rows"
        elif result.get("rethink_meaning"):
            outcome = "value not in data"  # even after the meaning was asked again
        elif result.get("attempts") and any(a.get("error") for a in result["attempts"]):
            outcome = "SQL error"  # SQLite itself rejected the SQL
        elif result.get("attempts") and any(a.get("status") == "rejected" for a in result["attempts"]):
            outcome = "failed checks"  # valid SQL, but Scout's checks rejected every try
        else:
            outcome = "no answer"
        # Version-22: every try's or chain's answer, and whether it was right: was a right answer there to
        # pick? which chain is best? The gold rows are read here, after the answer: never by a vote.
        chains = []
        for number, a in enumerate(result.get("attempts") or [], 1):
            chain = {"chain": a.get("chain") or f"try {number}", "status": a.get("status"), "sql": a.get("sql"),
                     "problems": a.get("problems") or [], "meaning": a.get("meaning"), "answer": a.get("answer"),
                     "retried": bool(a.get("retried")), "style": a.get("style"), "method": a.get("method"),
                     "model": a.get("model"), "sentence": a.get("sentence"), "measure": a.get("measure"),
                     "spec": a.get("spec"), "flags": a.get("flags") or [], "warnings": a.get("warnings") or [],
                     "rows": (a.get("rows") or [])[:5], "correct": False}
            if a.get("status") == "ok" and a.get("sql"):
                try:
                    _, chain_rows = pipeline.database.run_all(a["sql"])
                    chain["correct"] = as_set(chain_rows) == as_set(gold_rows)
                    chain["rows"] = chain_rows[:5]
                except Exception as e:
                    chain["problems"] = [*chain["problems"], f"Couldn't be run again: {e}"]
            chains.append(chain)
        calls = usage.usage_store.calls_for(tag)
        entry = {
            "id": q["id"], "n": q.get("n"), "db_id": q.get("db_id", "retail_store"),
            "difficulty": DIFFICULTY.get(q.get("difficulty"), q.get("difficulty")),
            # evidence = the hint Scout was given (BIRD); trap = what the question tests (practice; not given)
            "question": q["question"], "evidence": q.get("evidence") or "", "trap": q.get("trap") or "",
            "gold_sql": q["gold_sql"], "sql": result.get("sql"), "correct": correct, "outcome": outcome,
            "error": error, "expected_rows": gold_rows[:10], "rows": rows[:10], "chains": chains,
            "confidence": result.get("confidence"), "seconds": round(seconds, 1), **calls,
        }
        if not correct and not self.run.get("check"):
            usage.usage_store.add_issue("eval", q["question"], {
                "eval_id": q["id"], "db_id": entry["db_id"], "suite": self.run["suite"],
                "difficulty": entry["difficulty"], "trap": q.get("trap") or q.get("evidence"),
                "meaning": meaning, "steps": steps, "sql": result.get("sql"), "error": error, "rows": rows[:10],
                "gold_sql": q["gold_sql"], "expected_rows": gold_rows[:10], "run_id": self.run["id"],
            }, run_id=self.run["id"])
        return entry

    def _finish(self, status: str, message: Optional[str]) -> None:
        with self._lock:
            self.run.update(status=status, message=message, current=None, wait_until=None, finished_at=now())
        self._save()
        if self.run.get("check"):  # the Evaluation tab goes back to showing the newest real run
            with self._lock:
                self.run = None

    def _save(self) -> None:
        with self._lock:
            text = json.dumps(self.run, indent=1, default=str)
            run_id = self.run["id"]
        folder = self.runs_dir / CHECKS_FOLDER if self.run.get("check") else self.runs_dir
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{run_id}.json").write_text(text + "\n", encoding="utf-8")
        if SUITES.get(self.run.get("suite"), {}).get("databases") and not self.run.get("check"):
            self._save_predictions()

    def _save_predictions(self) -> None:
        """BIRD's official format, for BIRD's own scripts (an unanswered question gets empty SQL)."""
        with self._lock:
            predictions = {
                # one line per query, as BIRD's files have (text inside quotes is left alone)
                str(r["id"]): re.sub(r"\s*\n\s*", " ", (r.get("sql") or "").strip()) + BIRD_SEPARATOR + r["db_id"]
                for r in self.run["results"]
            }
            run_id = self.run["id"]
        folder = self.runs_dir / "bird_predictions"  # not *.json in runs/, so never read as a run
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{run_id}.json").write_text(json.dumps(predictions, indent=1) + "\n", encoding="utf-8")
