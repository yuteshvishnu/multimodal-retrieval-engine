"""
Version-22: the Full flow (Ask Me Anything) and the votes replayed from its saved chains. A stand-in
Scout: each chain returns the rows we give it, no LLM. The checks are Scout's own, stubbed here.
"""
import re
import sqlite3
import types
from pathlib import Path

from backend.text_to_sql import flows
from backend.text_to_sql.flows import (
    CHAINS, FlowRunner, adaptive_vote, answer_key, confidence_of, fit_accuracies, is_empty, majority_vote, weighted_vote,
)
from backend.text_to_sql.sql_generator import SYSTEM_PROMPT


def scout(answers, calls):
    """answers: chain -> rows, "bad" for SQL that always fails the checks, or "fixable" for SQL
    that fails the first time and returns [[7]] once it's told why."""

    def generate(question, schema, budget, context="", meaning="", temperature=0, tag=None, instruction="",
                 examples="", **kwargs):
        budget.take()
        calls.append(("sql", tag["chain"], meaning, instruction, examples, kwargs.get("feedback", "")))
        return f"SELECT '{tag['chain']}' /* {'retry' if kwargs.get('feedback') else 'first'} */"

    def check_and_run(candidate, tables, now):
        rows = answers[candidate["chain"]]
        if rows == "fixable":
            rows = [[7]] if "retry" in candidate["sql"] else "bad"
        if rows == "bad":
            return {**candidate, "status": "rejected", "problems": ["Segment has no value 'discount'"]}
        return {**candidate, "status": "ok", "rows": rows, "columns": ["n"], "more": False}

    return types.SimpleNamespace(
        generator=types.SimpleNamespace(api_key="k", model_name="m", generate=generate),
        _prepare=lambda q: ([], "schema", "context", None),
        check_and_run=check_and_run,
        database=types.SimpleNamespace(run_all=lambda sql: (["n"], [])),
    )


def runner(answers, calls):
    r = FlowRunner(scout(answers, calls))

    def meaning(chain, question, schema, context, budget, tag):
        budget.take()
        calls.append(("meaning", chain["id"], chain["style"], "", ""))
        return f"{chain['style']} meaning"

    r.meaning = meaning
    return r


def same(rows):
    return {c["id"]: rows for c in CHAINS}


def saved(*answers):
    """Chains as a run saves them: an answer per chain (None: abstained), right if it starts with 'right'."""
    return [{"chain": c["id"], "status": "ok" if a else "rejected", "answer": a, "correct": bool(a and a.startswith("right"))}
            for c, a in zip(CHAINS, answers)]


def test_full_runs_four_chains_each_rewriting_then_writing_sql_in_its_own_way():
    calls = []
    result = runner(same([[391]]), calls).run("How many?")
    assert result["llm_calls"] == 8 and result["rows"] == [[391]] and result["confidence"] == "high"
    assert sorted(c[1] for c in calls if c[0] == "meaning") == ["direct", "explore", "plan", "shape"]
    sql = {c[1]: c for c in calls if c[0] == "sql"}
    assert sql["explore"][2] == "data meaning" and sql["plan"][2] == "answer meaning" and sql["shape"][2] == "hint meaning"
    assert len({c[3] for c in sql.values()}) == 4 and len({c[4] for c in sql.values()}) == 4  # own instruction, own examples
    assert [a["chain"] for a in result["attempts"]] == ["direct", "explore", "plan", "shape"]


def test_adaptive_live_stops_at_four_calls_when_the_first_two_chains_agree():
    calls = []
    result = runner({"direct": [[1]], "explore": [[1]], "plan": [[9]], "shape": [[9]]}, calls).run("How many?", flow="adaptive")
    assert result["llm_calls"] == 4 and result["rows"] == [[1]] and result["confidence"] == "high"
    assert [a["chain"] for a in result["attempts"]] == ["direct", "explore"]


def test_adaptive_live_runs_all_four_when_they_disagree_or_one_fails():
    result = runner({"direct": [[1]], "explore": [[2]], "plan": [[2]], "shape": [[2]]}, []).run("How many?", flow="adaptive")
    assert result["llm_calls"] == 8 and result["rows"] == [[2]] and len(result["attempts"]) == 4
    result = runner({"direct": "bad", "explore": [[2]], "plan": [[2]], "shape": [[3]]}, []).run("How many?", flow="adaptive")
    assert result["llm_calls"] == 9 and result["rows"] == [[2]]  # 8, plus the rejected chain's one retry


def test_adaptive_doesnt_stop_early_on_an_empty_answer():
    result = runner({"direct": [], "explore": [], "plan": [[2]], "shape": [[2]]}, []).run("How many?", flow="adaptive")
    assert result["llm_calls"] == 8 and result["rows"] == [[2]] and len(result["attempts"]) == 4
    result = runner({"direct": [[None]], "explore": [[None]], "plan": [[2]], "shape": [[3]]}, []).run("x", flow="adaptive")
    assert result["llm_calls"] == 8  # only NULLs: nothing was found either
    assert is_empty([]) and is_empty([[None, None]]) and not is_empty([[0]]) and not is_empty(None)
    # the tie above went to the answer with data, not to the earliest chain
    # the replay follows the same rule (a chain's saved rows)
    chains = saved("a", "a", "b", "b")
    chains[0]["rows"] = chains[1]["rows"] = []
    chains[2]["rows"] = chains[3]["rows"] = [[2]]
    decision = adaptive_vote(chains)
    assert decision["calls"] == 8 and decision["winner"]["answer"] == "b"


def test_adaptive_retries_rejected_sql_once_telling_it_why():
    calls = []
    answers = {"direct": "fixable", "explore": [[7]], "plan": [[1]], "shape": [[1]]}
    result = runner(answers, calls).run("How many?", flow="adaptive")
    assert result["llm_calls"] == 5 and result["rows"] == [[7]] and result["confidence"] == "high"  # stopped early
    assert result["attempts"][0]["retried"] and not result["attempts"][1]["retried"]
    retry = next(c for c in calls if c[0] == "sql" and c[5])
    assert retry[1] == "direct" and "discount" in retry[5]
    # still rejected: it abstains, and the most calls are 12
    calls = []
    result = runner(same("bad"), calls).run("How many?", flow="adaptive")
    assert result["llm_calls"] == 12 and result["error"].startswith("No chain")


def test_a_chain_failing_the_checks_abstains_and_there_is_no_retry():
    calls = []
    result = runner({"direct": "bad", "explore": [[5]], "plan": [[5]], "shape": [[5]]}, calls).run("How many?")
    assert result["rows"] == [[5]] and result["llm_calls"] == 8 and result["confidence"] == "high"  # Full: no retry
    assert not any(a["retried"] for a in result["attempts"])
    result = runner(same("bad"), []).run("How many?")
    assert result["error"].startswith("No chain") and "discount" in result["failure_reason"]


def test_answers_are_the_same_rows_as_a_set_as_bird_compares_them():
    assert answer_key([[1], [2], [2]]) == answer_key([[2], [1]])
    assert answer_key([[1.00000001]]) == answer_key([[1.0]])
    assert answer_key([[1]]) != answer_key([[2]])


def test_mixed_uses_each_model_once_with_styles_and_methods_shuffled_repeatably():
    a = flows.mixed_chains("How many customers?")
    assert a == flows.mixed_chains("How many customers?")  # the same question, the same shuffle
    for chains in (a, flows.mixed_chains("Which store sold most?")):
        assert sorted(c["model"] for c in chains) == sorted(m["id"] for m in flows.MIXED_MODELS)
        assert len({c["style"] for c in chains}) == 3 and len({c["method"] for c in chains}) == 3  # no repeats
    assert len(flows.MIXED_MODELS) == 3 and "z-ai/glm-5.3-flash" not in {m["id"] for m in flows.MIXED_MODELS}
    many = [flows.mixed_chains(f"Question {i}") for i in range(40)]
    gpt = "GPT-OSS 120B"
    assert len({next(c["style"] for c in chains if c["id"] == gpt) for chains in many}) == 4  # no model is tied to a style
    assert len({chains[0]["id"] for chains in many}) == 3  # so ties don't favor one model


def test_mixed_sends_each_call_to_its_own_model_and_votes_by_model():
    answers = {m["voter"]: [[1]] for m in flows.MIXED_MODELS}
    calls, models = [], []
    r = runner(answers, calls)
    real = r.p.generator.generate
    r.p.generator.generate = lambda *a, **k: (models.append(k.get("model")), real(*a, **k))[1]
    result = r.run("How many?", flow="mixed")
    assert result["llm_calls"] == 6 and result["rows"] == [[1]] and result["confidence"] == "high"
    assert sorted(models) == sorted(m["id"] for m in flows.MIXED_MODELS)
    assert {a["model"] for a in result["attempts"]} == {m["id"] for m in flows.MIXED_MODELS}
    # weights are learned per model, with 3 voters: two that agree more than the third does
    questions = []
    voters = [m["voter"] for m in flows.MIXED_MODELS]
    for i in range(20):
        answer = ["x", "x" if i % 5 else f"w{i}", "x" if i % 2 else f"v{i}"]
        questions.append([{"chain": v, "status": "ok", "answer": a} for v, a in zip(voters, answer)])
    accuracies = fit_accuracies(questions)
    assert set(accuracies) == set(voters) and accuracies[voters[0]] > accuracies[voters[2]]
    assert fit_accuracies([[{"chain": v, "status": "ok", "answer": "x"} for v in voters]] * 3) is None  # too few


def test_the_same_confidence_rule_for_every_flow():
    assert confidence_of(4, 4) == "high" and confidence_of(2, 2) == "high"  # every chain that answered agrees
    assert confidence_of(4, 3) == "medium" and confidence_of(4, 2) == "medium"
    assert confidence_of(4, 1) == "low" and confidence_of(1, 1) == "low"  # one chain alone is never sure
    assert confidence_of(0, 0) is None


def test_majority_ties_go_to_the_earliest_chain():
    decision = majority_vote(saved("a", "b", "b", "a"))
    assert decision["winner"]["chain"] == "direct" and decision["confidence"] == "medium"
    assert majority_vote(saved("a", "b", "b", "b"))["winner"]["answer"] == "b"
    assert majority_vote(saved("a", "a", "b", "b"), 3)["winner"]["answer"] == "a"  # the first 3 chains only
    assert majority_vote(saved(None, None, None, None)) is None


def test_weights_are_learned_from_agreement_across_the_questions_and_can_break_a_tie():
    """direct and explore are always right; plan and shape wrong on different questions half of the time."""
    questions = []
    for i in range(20):
        plan = "x" if i % 2 else f"w{i}"
        shape = "x" if i % 2 == 0 else f"v{i}"
        questions.append(saved("x", "x", plan, shape))
    accuracies = fit_accuracies(questions)
    assert accuracies["direct"] > accuracies["plan"] and accuracies["explore"] > accuracies["shape"]
    # a tie: without weights the earliest chain wins; with them, the trusted pair does
    assert majority_vote(saved("wrong", "right", "wrong", "right"))["winner"]["answer"] == "wrong"
    accuracies = {"direct": 0.6, "explore": 0.95, "plan": 0.6, "shape": 0.95}
    assert weighted_vote(saved("wrong", "right", "wrong", "right"), accuracies)["winner"]["answer"] == "right"
    assert fit_accuracies(questions[:3]) is None  # too few questions: equal weights
    assert weighted_vote(saved("a", "a", "b", "b"), None)["winner"]["answer"] == "a"


def test_adaptive_stops_at_four_calls_when_the_first_two_chains_agree():
    decision = adaptive_vote(saved("right", "right", "x", "y"))
    assert decision["calls"] == 4 and decision["confidence"] == "high" and decision["winner"]["correct"]
    decision = adaptive_vote(saved("x", "right", "right", "y"))  # they disagree: all 4
    assert decision["calls"] == 8 and decision["winner"]["answer"] == "right"
    decision = adaptive_vote(saved(None, "right", "right", "y"))  # one abstains: all 4
    assert decision["calls"] == 8
    assert adaptive_vote(saved(None, None, None, None)) is None


def test_the_worked_examples_run_on_the_practice_database_and_are_not_about_a_benchmark():
    db = next(Path("data/text_to_sql").rglob("*.sqlite"))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for chain in CHAINS:
        text = chain["sql_examples"]
        queries = re.findall(r"```sql\n(.*?)\n```", text, re.DOTALL) or re.findall(r"^SELECT .*$", text, re.MULTILINE)
        assert len(queries) == 2
        for sql in queries:
            con.execute(sql).fetchall()
    con.close()
    combined = " ".join(c["sql_examples"] + c["meaning_examples"] for c in CHAINS).lower()
    for word in ("gasstations", "yearmonth", "student_club", "women", "consumption"):
        assert word not in combined


def test_meanings_are_told_not_to_change_the_format():
    assert "never how to show it" in flows.MEANING_SYSTEM and "combine" in flows.MEANING_SYSTEM
    assert all(flows.FORMAT_RULE in c["instruction"] for c in CHAINS)


def test_apostrophes_are_doubled_and_the_checks_understand_them():
    import datetime

    from backend.text_to_sql.checks import check_sql

    assert "'O''Brien'" in SYSTEM_PROMPT
    table = {"name": "events", "database": "d", "row_count": 2, "schema": "", "columns": [], "links": set(),
             "profile": {"title": {"values": {"Women's Soccer", "Chess"}}}}
    today = datetime.date(2026, 1, 1)
    assert check_sql("SELECT 1 FROM events WHERE title = 'Women''s Soccer'", [table], today)[0] == []
    assert check_sql("SELECT 1 FROM events WHERE title IN ('Women''s Soccer', 'Chess')", [table], today)[0] == []
    assert "Tennis" in check_sql("SELECT 1 FROM events WHERE title = 'Tennis'", [table], today)[0][0]


def test_flows_never_read_correct_answers():
    import inspect

    source = inspect.getsource(flows.FlowRunner)
    assert "gold" not in source and '"correct"' not in source


# ---------- Version-24: Adaptive + English program ----------

def test_a_sentence_becomes_one_read_query_by_the_program():
    assert flows.sentence_to_sql('get name\n  from users where city is "O\'Hare";') == "SELECT name FROM users WHERE city = 'O''Hare'"


def test_sql_and_writes_cannot_get_through_the_sentence():
    for sentence in ("raw: DELETE FROM users", 'get raw "1" from users', ".tables", "delete everything from users",
                     "drop table users", "get a from b; get c from d", "", "this is not the format"):
        try:
            flows.sentence_to_sql(sentence)
        except flows.SentenceRejected:
            continue
        raise AssertionError(sentence)
    # the word raw inside a value or a backticked name is fine
    assert "'raw'" in flows.sentence_to_sql('get a from b where c is "raw"')
    assert flows.sentence_to_sql("get `raw` from b") == "SELECT raw FROM b"


def test_every_worked_example_is_a_valid_sentence_and_matches_the_sql_chains():
    for chain in flows.ENGLISH_CHAINS:
        sentences = re.findall(r"```text\n(.*?)\n```", chain["examples"]) or [
            line for line in chain["examples"].splitlines() if line and not line.startswith("Question:")]
        assert len(sentences) == 2
        for sentence in sentences:
            assert flows.sentence_to_sql(sentence).startswith("SELECT")
    assert [c["id"] for c in flows.ENGLISH_CHAINS] == [c["id"] for c in CHAINS]


def test_the_model_gets_only_the_query_rules():
    assert "## Queries" in flows.FORMAT_GUIDE and "## Conditions" in flows.FORMAT_GUIDE
    assert "Changing data" not in flows.FORMAT_GUIDE and "`raw " not in flows.FORMAT_GUIDE and "raw:" not in flows.FORMAT_GUIDE


def english_runner(replies, seen):
    """A stand-in model: meanings as JSON, then sentences from replies. Runs the real program."""
    def create(**kwargs):
        system = kwargs["messages"][0]["content"]
        if "strict English format" in system:
            seen.append(kwargs["messages"][-1]["content"])
            content = replies[len(seen) - 1]
        else:
            content = '{"meaning": "m"}'
        usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None)
        return types.SimpleNamespace(usage=usage, choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])

    ran = []

    def check_and_run(candidate, tables, now):
        ran.append(candidate["sql"])
        return {**candidate, "status": "ok", "rows": [[1]], "columns": ["n"], "more": False}

    pipeline = types.SimpleNamespace(
        generator=types.SimpleNamespace(api_key="k", model_name="m", generate=None),
        _prepare=lambda q: ([], "schema", "context", None), check_and_run=check_and_run,
        database=types.SimpleNamespace(run_all=lambda sql: (["n"], [[1]])),
    )
    r = FlowRunner(pipeline)
    r._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    return r, ran


def test_english_flow_stops_early_and_never_asks_the_model_for_sql(monkeypatch):
    monkeypatch.setattr(flows, "chat", lambda client, tag, **kw: client.chat.completions.create(**kw))
    seen = []
    r, ran = english_runner(["```text\nget a from b\n```"] * 2, seen)
    result = r.run("How many?", flow=flows.ENGLISH_FLOW)
    assert result["llm_calls"] == 4 and ran == ["SELECT a FROM b"] * 2 and result["sql"] == "SELECT a FROM b"
    assert {a["sentence"] for a in result["attempts"]} == {"get a from b"}


def test_a_rejected_sentence_gets_one_retry_told_why(monkeypatch):
    monkeypatch.setattr(flows, "chat", lambda client, tag, **kw: client.chat.completions.create(**kw))
    chain = flows.ENGLISH_CHAINS[0]
    seen = []
    r, ran = english_runner(["nonsense words here", "get a from b"], seen)
    done = r.chain(chain, "Q", "schema", "", [], None, flows.CallBudget(3), {"question_id": "q"}, retry=True)
    assert done["status"] == "ok" and done["retried"] and done["sql"] == "SELECT a FROM b" and ran == ["SELECT a FROM b"]
    assert "rejected the sentence" in seen[1]
    # without a second chance, it abstains and nothing is run
    seen = []
    r, ran = english_runner(["nonsense words here", "still nonsense words"], seen)
    done = r.chain(chain, "Q", "schema", "", [], None, flows.CallBudget(3), {"question_id": "q"}, retry=True)
    assert done["status"] == "rejected" and done["sql"] is None and not ran and len(seen) == 2


def voted(chain, answer, rows):
    return {"chain": chain, "status": "ok", "answer": answer, "rows": rows}


def test_an_answer_with_data_outvotes_empty_ones_and_lowers_the_confidence():
    chains = [voted("a", "none", []), voted("b", "null", [[None]]), voted("c", "none", []), voted("d", "right", [[5]])]
    decision = majority_vote(chains)
    assert decision["winner"]["chain"] == "d" and decision["agreeing"] == ["d"] and decision["confidence"] == "low"


def test_most_data_answers_still_win_and_empty_wins_only_when_nothing_else_is_there():
    chains = [voted("a", "none", []), voted("b", "x", [[1]]), voted("c", "y", [[2]]), voted("d", "y", [[2]])]
    decision = majority_vote(chains)
    assert decision["winner"]["answer"] == "y" and decision["confidence"] == "medium"
    all_empty = majority_vote([voted("a", "none", []), voted("b", "none", [])])
    assert all_empty["winner"]["chain"] == "a" and all_empty["confidence"] == "high"


def test_the_weighted_vote_follows_the_same_rule():
    chains = [voted("a", "none", []), voted("b", "x", [[1]])]
    assert weighted_vote(chains, {"a": 0.95, "b": 0.05})["winner"]["chain"] == "b"


def test_the_respectively_rule_reaches_sql_and_sentence_writers():
    from backend.text_to_sql.sql_generator import ANSWER_RULES
    assert "respectively" in ANSWER_RULES and "respectively" in SYSTEM_PROMPT and "respectively" in flows.ENGLISH_SYSTEM


def test_the_granularity_rule_reaches_every_step_that_decides_what_to_compute():
    from backend.text_to_sql import interpreter, planner
    from backend.text_to_sql.sql_generator import GRANULARITY_RULE, GRANULARITY_RULE_PLAIN
    assert GRANULARITY_RULE in SYSTEM_PROMPT and GRANULARITY_RULE in flows.ENGLISH_SYSTEM
    for prompt in (interpreter.SYSTEM_PROMPT, planner.SYSTEM_PROMPT,
                   flows.MEANING_SYSTEM.format(style="s", examples="e")):
        assert GRANULARITY_RULE_PLAIN in prompt


def test_the_metric_and_date_part_rules_reach_their_steps_and_name_no_question():
    from backend.text_to_sql import interpreter, planner
    from backend.text_to_sql import sql_generator as g
    for rule in (g.METRIC_RULE, g.TIME_PART_RULE):
        assert rule in SYSTEM_PROMPT and rule in flows.ENGLISH_SYSTEM
    assert g.METRIC_RULE_PLAIN in interpreter.SYSTEM_PROMPT and g.METRIC_RULE_PLAIN in planner.SYSTEM_PROMPT
    assert g.METRIC_RULE_PLAIN in flows.MEANING_SYSTEM.format(style="s", examples="e")
    assert g.TIME_PART_RULE_PLAIN in planner.SYSTEM_PROMPT and g.TIME_PART_RULE_PLAIN in interpreter.SYSTEM_PROMPT
    assert g.TIME_PART_RULE_PLAIN in flows.MEANING_SYSTEM.format(style="s", examples="e")
    for rule in (g.METRIC_RULE, g.METRIC_RULE_PLAIN, g.TIME_PART_RULE, g.TIME_PART_RULE_PLAIN, g.GRANULARITY_RULE):
        assert not any(word in rule for word in ("SME", "CZK", "EUR", "Consumption", "2012"))


# ---------- Version-24: the measure check ----------

from backend.text_to_sql.measure_check import measure_problem  # noqa: E402


def test_the_measure_check_flags_only_clear_mismatches():
    adds = "SELECT SUM(CASE WHEN c = 'x' THEN t.amount * t.price ELSE 0 END) FROM t"
    for count_sql in ("SELECT COUNT(*) FROM t", "SELECT SUM(c = 'x') - SUM(c = 'y') FROM t",
                      "SELECT SUM(CASE WHEN c = 'x' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) FROM t",
                      "SELECT SUM(IIF(c = 'x', 1, 0)) FROM t"):
        assert measure_problem("entity count", count_sql) is None
        assert measure_problem("money", count_sql) is not None or "COUNT(*)" in count_sql and "SUM" in count_sql
    assert "adds up values" in measure_problem("entity count", adds)
    assert measure_problem("transaction count", adds) and measure_problem("money", adds) is None
    for anything in ("SELECT name FROM t", "not sql at all", ""):
        assert measure_problem("entity count", anything) is None
    assert measure_problem("other", adds) is None and measure_problem("nonsense", adds) is None


def test_a_flagged_chain_gets_one_retry_told_why_and_keeps_its_answer_otherwise():
    asked = []

    def generate(question, schema, budget, context="", meaning="", temperature=0, tag=None, instruction="",
                 examples="", feedback="", model=None, **kwargs):
        budget.take()
        asked.append(feedback)
        return "SELECT COUNT(*) FROM t" if feedback else "SELECT SUM(amount) FROM t"

    ok = lambda c, tables, now: {**c, "status": "ok", "rows": [[1]], "columns": ["n"], "more": False}
    pipeline = types.SimpleNamespace(generator=types.SimpleNamespace(api_key="k", model_name="m", generate=generate),
                                     check_and_run=ok, database=types.SimpleNamespace(run_all=lambda s: ([], [])))
    r = FlowRunner(pipeline)
    r.meaning = lambda *a: (budget_take(a[4]), ("How many customers", "entity count"))[1]
    done = r.chain(CHAINS[0], "Q", "s", "", [], None, flows.CallBudget(3), {"question_id": "q"}, retry=True)
    assert done["sql"] == "SELECT COUNT(*) FROM t" and done["retried"] and "adds up values" in asked[1]
    assert not done["warnings"]
    # without retries (Full, Mixed), the answer stays, with a warning
    asked.clear()
    done = r.chain(CHAINS[0], "Q", "s", "", [], None, flows.CallBudget(3), {"question_id": "q"}, retry=False)
    assert done["sql"] == "SELECT SUM(amount) FROM t" and done["status"] == "ok" and "adds up values" in done["warnings"][0]


def budget_take(budget):
    budget.take()


def test_counting_up_and_down_is_still_counting():
    sql = "SELECT SUM(CASE WHEN a = 'x' THEN 1 WHEN a = 'y' THEN -1 ELSE 0 END) FROM t"
    assert measure_problem("entity count", sql) is None


def test_the_average_rule_reaches_every_step_that_computes():
    from backend.text_to_sql import interpreter, planner
    from backend.text_to_sql import sql_generator as g
    assert g.AVERAGE_RULE in SYSTEM_PROMPT and g.AVERAGE_RULE in flows.ENGLISH_SYSTEM
    for prompt in (interpreter.SYSTEM_PROMPT, planner.SYSTEM_PROMPT, flows.MEANING_SYSTEM.format(style="s", examples="e")):
        assert g.AVERAGE_RULE_PLAIN in prompt
