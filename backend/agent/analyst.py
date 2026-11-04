"""
The Failure Analyst (Version-11): an AI agent. Unlike the rest of Scout, where our code
decides every step, here the LLM decides what to look at next:

    read the failure -> LLM picks a tool -> our code runs it (read-only) -> LLM reads the result
    -> picks the next tool ... -> submit_report: what happened, why, a fix, the corrected SQL

Our code keeps control of everything else:
    - only the read-only tools in tools.py exist; at most AGENT_MAX_TOOL_CALLS per issue,
      then only submit_report is offered
    - the report is checked for its required parts, and its fix is tested by code (fixes.py);
      a failed test goes back to the agent once, so it can look again and correct itself
    - it only suggests: the developer approves or rejects every fix on the Dashboard
    - every step is saved as it happens, so the Dashboard shows the investigation live,
      and every LLM call is priced like any other (stage "agent")
"""
import json
from typing import Dict, List

from backend.agent.fixes import FIX_TYPES, test_fix
from backend.agent.tools import SPECS, ReadOnlyTools
from backend.config import AGENT_MAX_TOOL_CALLS, AGENT_MODEL
from backend.llm import chat, make_client

SOURCE_LABELS = {"eval": "Eval", "user_report": "User report", "sql_error": "SQL error"}
ROOT_CAUSES = ["meaning", "steps", "sql", "data", "description", "retrieval", "other"]

REPORT_SPEC = {"type": "function", "function": {
    "name": "submit_report",
    "description": "Your final report. Call it once you know what went wrong.",
    "parameters": {"type": "object", "properties": {
        "what_happened": {"type": "string", "description": "One or two plain sentences: what Scout did and what came out."},
        "why": {"type": "string", "description": "The cause, backed by what the tools showed."},
        "root_cause": {"type": "string", "enum": ROOT_CAUSES,
                       "description": "Where it broke: the meaning chosen, the steps, the SQL, the data itself, "
                                      "a missing or wrong table description, the tables picked, or other."},
        "fix_type": {"type": "string", "enum": list(FIX_TYPES),
                     "description": "error_book: a new known-error entry; table_description: a note for a table's "
                                    "description; prompt or code: a change for the developer; none: nothing to fix."},
        "fix": {"type": "string", "description": "The suggested fix, in plain words."},
        "fixed_sql": {"type": "string", "description": "A corrected SQLite query that answers the question, if you have one."},
        "error_book_entry": {"type": "object", "description": "Only for fix_type error_book.", "properties": {
            "id": {"type": "string", "description": "short-kebab-case id"},
            "match": {"type": "string", "description": "regular expression matching SQLite's error message"},
            "hint": {"type": "string", "description": "what to tell the model when this error happens"},
            "rule": {"type": "string", "description": "one-line rule for the SQL prompt"},
            "example_bad": {"type": "string"}, "example_fix": {"type": "string"},
        }},
        "description_note": {"type": "object", "description": "Only for fix_type table_description.", "properties": {
            "table": {"type": "string"}, "note": {"type": "string", "description": "one sentence to add to the description"},
        }},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    }, "required": ["what_happened", "why", "root_cause", "fix_type", "fix", "confidence"]},
}}

SYSTEM_PROMPT = (
    "You are the Failure Analyst for Scout, an app that turns questions into SQLite queries "
    "(flow: confirm the meaning -> plain steps -> SQL, 3 tries compared). You get one failure: "
    "a wrong answer on a test question with a known correct answer, a user's report that an "
    "answer is wrong, or SQL that SQLite rejected.\n"
    "Find out why it failed. Check facts with the tools instead of guessing: look at the "
    "tables, the real values in the columns involved (spelling, letter case, date formats, "
    "units), and test your ideas with small queries. Try a corrected query before you report.\n"
    f"You have at most {AGENT_MAX_TOOL_CALLS} tool calls; then call submit_report. "
    "The tools only read data; you cannot change anything, you only suggest.\n"
    "Before you report, run your corrected query with run_query: the report must include it "
    "(fixed_sql), and our code checks it against the correct answer when there is one.\n"
    "Pick the fix that stops this happening again:\n"
    "- error_book: ONLY when SQLite rejected the SQL with an error message. The entry's 'match' is a "
    "regex on that error message (not on the SQL), plus a hint and a one-line rule.\n"
    "- table_description: the data has a trap (spelling, case, format, units) the description doesn't mention.\n"
    "- prompt: the SQL ran but answered the wrong thing (e.g. ties dropped, wrong filter): a rule for "
    "the SQL or steps prompt.\n"
    "- code: a bug in Scout itself. none: Scout was right, or the test question is wrong.\n"
    "Write for a developer: short, specific, plain words."
)


def failure_text(issue: Dict) -> str:
    """The failure, as the agent first reads it."""
    d = issue["details"]
    lines = [f"Source: {SOURCE_LABELS.get(issue['source'], issue['source'])}", f"Question: {issue['question']}"]
    if d.get("comment"):
        lines.append(f"User's comment: {d['comment']}")
    if d.get("trap"):
        lines.append(f"Known trap in this question: {d['trap']}")
    if d.get("meaning"):
        lines.append(f"Meaning confirmed: {d['meaning']}")
    if d.get("steps"):
        lines.append("Steps confirmed:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(d["steps"], 1)))
    if d.get("sql"):
        lines.append(f"Scout's SQL:\n{d['sql']}")
    for a in d.get("attempts", []):
        lines.append(f"Try (round {a.get('round')}):\n{a.get('sql')}\n-> {'; '.join(a.get('problems', []))}")
    if d.get("error"):
        lines.append(f"Error: {d['error']}")
    if d.get("rows") is not None:
        lines.append(f"Scout's first rows: {json.dumps(d['rows'][:5], default=str)}")
    if d.get("gold_sql"):
        lines.append(f"Correct SQL: {d['gold_sql']}")
    if d.get("expected_rows") is not None:
        lines.append(f"Correct first rows: {json.dumps(d['expected_rows'][:5], default=str)}")
    return "\n\n".join(lines)


def step_title(name: str, args: Dict) -> str:
    """A tool call in plain words, for the Dashboard."""
    return {
        "list_tables": "Listed the tables",
        "describe_table": f"Read the description of {args.get('table')}",
        "column_values": f"Checked the values in {args.get('table')}.{args.get('column')}",
        "run_query": "Ran a test query",
        "known_errors": "Read the error book",
    }.get(name, f"Called {name}")


def report_problems(report: Dict) -> List[str]:
    """What a report still lacks; it's sent back to the agent until there's nothing."""
    missing = [k for k in ("what_happened", "why", "fix") if not str(report.get(k, "")).strip()]
    if report.get("fix_type") != "none" and not str(report.get("fixed_sql", "")).strip():
        missing.append("fixed_sql (run a corrected query with run_query first)")
    problems = [f"missing {', '.join(missing)}"] if missing else []
    if report.get("fix_type") not in FIX_TYPES:
        problems.append(f"fix_type must be one of {', '.join(FIX_TYPES)}")
    return problems


class FailureAnalyst:
    def __init__(self, pipeline, store, client=None, model: str = AGENT_MODEL, max_tool_calls: int = AGENT_MAX_TOOL_CALLS):
        self.pipeline = pipeline
        self.store = store
        self.client = client or make_client()
        self.model = model
        self.max_tool_calls = max_tool_calls
        self.tools = ReadOnlyTools(pipeline)

    def investigate(self, issue_id: str) -> None:
        """Runs the whole investigation of one issue (already marked 'investigating'); never raises."""
        try:
            self._investigate(issue_id)
        except Exception as e:
            self.store.add_agent_step(issue_id, "error", "Stopped", str(e))
            self.store.set_issue(issue_id, "failed")

    def _investigate(self, issue_id: str) -> None:
        if self.client is None:
            raise RuntimeError("The LLM API key is not set (LLM_PROVIDER and its key in .env).")
        issue = self.store.issue(issue_id)
        text = failure_text(issue)
        self.store.add_agent_step(issue_id, "start", f"Read the failure ({SOURCE_LABELS[issue['source']]})", text)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]

        used, retried = 0, False
        for turn in range(self.max_tool_calls + 6):
            final = used >= self.max_tool_calls
            try:
                completion = chat(
                    self.client, {"stage": "agent", "round": turn + 1},
                    model=self.model, messages=messages, temperature=0.2,
                    # the agent chooses freely; once its tool calls are used up, only the report is left
                    tools=[REPORT_SPEC] if final else SPECS + [REPORT_SPEC],
                    tool_choice="required" if final else "auto",
                )
            except Exception as e:
                if "did not call a tool" not in str(e):
                    raise
                # Groq refuses a reply that should have been a tool call: ask again
                messages.append({"role": "user", "content": "Call submit_report now, with what you found."})
                continue
            self._count(issue_id, completion)
            message = completion.choices[0].message
            reasoning = (getattr(message, "reasoning", None) or message.content or "").strip()
            if reasoning:
                self.store.add_agent_step(issue_id, "reasoning", "Thinking", reasoning if len(reasoning) <= 1200 else reasoning[:1200] + "…")
            calls = message.tool_calls or []
            if not calls:
                messages.append({"role": "assistant", "content": message.content or ""})
                messages.append({"role": "user", "content": "Use a tool to check, or call submit_report if you know the cause."})
                continue

            messages.append({"role": "assistant", "content": message.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls
            ]})
            for call in calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = None

                if call.function.name == "submit_report" and isinstance(args, dict):
                    problems = report_problems(args)
                    if problems:
                        result = f"Report incomplete: {'; '.join(problems)}. Call submit_report again."
                    else:
                        test = test_fix(issue, args, self.pipeline)
                        if test["passed"] is not False or retried:
                            self._finish(issue, args, test)
                            return
                        # our test of the fix failed: one more chance to find what's still wrong
                        retried = True
                        failed = "; ".join(c["text"] for c in test["checks"] if not c["ok"])
                        self.store.add_agent_step(issue_id, "test_failed", "Tested the fix: failed, sent back to the agent", failed)
                        result = (f"Our test of your fix failed: {failed}. Find what is still different "
                                  "(compare with the correct answer if there is one), then call submit_report again.")
                elif args is None or not isinstance(args, dict):
                    result = "The arguments were not valid JSON."
                elif final or used >= self.max_tool_calls:
                    result = "No tool calls left: call submit_report now."
                else:
                    used += 1
                    result = self.tools.call(call.function.name, args)
                    detail = f"{args['sql']}\n\n→ {result}" if call.function.name == "run_query" and "sql" in args else result
                    self.store.add_agent_step(issue_id, "tool", step_title(call.function.name, args), detail[:2000])
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        raise RuntimeError("The agent did not write a report in time.")

    def _finish(self, issue: Dict, report: Dict, test: Dict) -> None:
        """Saves the report with code's test of its fix."""
        self.store.add_agent_step(issue["id"], "report", "Wrote the report", report.get("why", ""))
        report["test"] = test
        verdict = {True: "passed", False: "failed", None: "needs a person to judge"}[report["test"]["passed"]]
        self.store.add_agent_step(
            issue["id"], "test_failed" if test["passed"] is False else "test", f"Tested the fix: {verdict}",
            "\n".join(c["text"] for c in test["checks"]),
        )
        self.store.set_issue(issue["id"], "report_ready", report=report)

    def _count(self, issue_id: str, completion) -> None:
        from backend.usage import call_cost

        u = completion.usage
        cost = call_cost(self.model, getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)[2]
        self.store.add_agent_usage(issue_id, cost)
