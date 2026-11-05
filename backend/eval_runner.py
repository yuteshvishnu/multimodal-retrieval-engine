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
from backend.config import (
    BIRD_CORRECTED_DIR, BIRD_MINI_DEV_DIR,
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
    """Rows in any order; floats rounded so 399.99 == 399.9900000001 (as in evaluate.py)."""
    return {tuple(round(v, 4) if isinstance(v, float) else v for v in row) for row in rows}


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


def estimate(questions: int) -> Dict:
    """What a run of this many questions should take, from measured averages (config.py)."""
    price_in, price_out = LLM_PRICES.get(LLM_MODEL_SQL, (0, 0))
    tokens = questions * EVAL_TOKENS_PER_QUESTION
    cost = tokens * 0.75 / 1e6 * price_in + tokens * 0.25 / 1e6 * price_out  # about 3/4 of tokens are input
    return {
        "questions": questions, "llm_calls": round(questions * EVAL_CALLS_PER_QUESTION), "tokens": tokens,
        "cost": round(cost, 4), "minutes": round(questions * EVAL_SECONDS_PER_QUESTION / 60, 1),
        "over_daily_limit": bool(PROVIDER_TOKENS_PER_DAY) and tokens > PROVIDER_TOKENS_PER_DAY,
        "daily_limit": PROVIDER_TOKENS_PER_DAY, "provider": PROVIDERS[LLM_PROVIDER]["name"],
    }


def is_rate_limited(error: str) -> bool:
    text = (error or "").lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text


class EvalRunner:
    def __init__(self, pipeline, runs_dir: Path = RUNS_DIR, sleep: Callable[[float], None] = time.sleep,
                 pipeline_for: Optional[Callable[[Path], object]] = None):
        """
        pipeline: the app's Scout, for the practice suite.
        pipeline_for(folder): makes a Scout over just that folder's database (BIRD: one per db_id).
        """
        self.pipeline = pipeline
        self.pipeline_for = pipeline_for
        self._pipelines: Dict[str, object] = {}  # db_id -> its pipeline, made when first needed
        self.runs_dir = runs_dir
        self.sleep = sleep  # tests pass a no-op
        self.run: Optional[Dict] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---------- control ----------

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, suite: str, first: int = 1, last: int = 0, wait: bool = False) -> Dict:
        if self.busy():
            raise RuntimeError("An evaluation is already running.")
        if suite not in SUITES or not SUITES[suite]["path"].exists():
            raise ValueError("That test suite isn't available.")
        questions = pick(load_questions(suite), first, last)
        info = SUITES[suite]
        self.run = {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
            "suite": suite, "benchmark": info["name"], "short": info["short"], "sample": False,
            "status": "running", "message": None, "run_at": now(), "finished_at": None,
            "model": LLM_MODEL_SQL, "provider": PROVIDERS[LLM_PROVIDER]["name"], "mode": "No user: the first meaning and the steps are accepted as written",
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
            "check": True, "question_id": q["id"],
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

    def resume(self, run_id: str, wait: bool = False) -> Dict:
        if self.busy():
            raise RuntimeError("An evaluation is already running.")
        run = self.load(run_id)
        if run is None or run["status"] == "finished":
            raise ValueError("Only a stopped run can be resumed.")
        done = {r["id"] for r in run["results"]}
        questions = [q for q in load_questions(run["suite"]) if q["id"] in run["planned"] and q["id"] not in done]
        self.run = {**run, "status": "running", "message": None, "finished_at": None}
        self._save()
        return self._launch(questions, wait)

    def stop(self) -> None:
        """Stops after the question being answered (a question is never cut in half)."""
        self._stop.set()

    def _launch(self, questions: List[Dict], wait: bool) -> Dict:
        self._stop.clear()
        self._thread = threading.Thread(target=self._work, args=(questions,), daemon=True, name="eval-run")
        self._thread.start()
        if wait:  # tests
            self._thread.join()
        return self.progress()

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

    def pipeline_for_db(self, suite: str, db_id: str):
        """The app's Scout, or for BIRD a Scout that has only this database (also used by the agent)."""
        databases = SUITES.get(suite, {}).get("databases")
        if not databases:
            return self.pipeline
        with self._lock:
            key = (suite, db_id)  # BIRD and corrected BIRD share database names, not descriptions
            if key not in self._pipelines:
                folder = databases / db_id
                if not folder.is_dir() or self.pipeline_for is None:
                    raise RuntimeError(f"BIRD database {db_id} isn't available.")
                self._pipelines[key] = self.pipeline_for(folder)
            return self._pipelines[key]

    def _pipeline(self, q: Dict):
        return self.pipeline_for_db(self.run["suite"], q["db_id"])

    def _flow(self, q: Dict, tag: str):
        """
        meaning (first option) -> steps (as written) -> SQL; returns (result, meaning, steps).
        If the checks find the meaning asks for a value the data doesn't have, the meaning is asked
        again with the real values, as the app does (at most MAX_MEANING_TRIES meanings in all).
        """
        question = q["question"] + (f"\n(Hint: {q['evidence']})" if q.get("evidence") else "")
        pipeline = self._pipeline(q)
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
        calls = usage.usage_store.calls_for(tag)
        entry = {
            "id": q["id"], "n": q.get("n"), "db_id": q.get("db_id", "retail_store"),
            "difficulty": DIFFICULTY.get(q.get("difficulty"), q.get("difficulty")),
            # evidence = the hint Scout was given (BIRD); trap = what the question tests (practice; not given)
            "question": q["question"], "evidence": q.get("evidence") or "", "trap": q.get("trap") or "",
            "gold_sql": q["gold_sql"], "sql": result.get("sql"), "correct": correct, "outcome": outcome,
            "error": error, "expected_rows": gold_rows[:10], "rows": rows[:10],
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
