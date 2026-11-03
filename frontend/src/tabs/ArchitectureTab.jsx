import { Icon } from "../components/ui.jsx";

// Architecture: how Scout is built today, how it started (Version 1), and what each version
// changed: the problem, what we thought, what we added, and what it measured. Static content.

// Block diagrams, drawn as SVG (generated once; styled by the .d-* rules in styles.css)
const V1_DIAGRAM = `<svg class="hld" viewBox="0 0 1200 116" role="img" aria-label="Version 1: the UI sends the question to FastAPI, one LLM call writes a SELECT from the full schema, SQLite runs it read-only, and the UI shows the SQL and rows."><defs><marker id="v1ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="d-head"/></marker><marker id="v1ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="d-head-accent"/></marker></defs>
<rect class="d-box" x="20" y="20" width="200" height="76" rx="10"/>
<text class="d-title" x="120.0" y="52" text-anchor="middle">React UI</text>
<text class="d-sub" x="120.0" y="69" text-anchor="middle">types a question</text>
<rect class="d-box" x="260" y="20" width="200" height="76" rx="10"/>
<text class="d-title" x="360.0" y="52" text-anchor="middle">FastAPI</text>
<text class="d-sub" x="360.0" y="69" text-anchor="middle">/api/sql/query</text>
<rect class="d-box" x="500" y="20" width="200" height="76" rx="10"/>
<text class="d-title" x="600.0" y="52" text-anchor="middle">Groq LLM</text>
<text class="d-sub" x="600.0" y="69" text-anchor="middle">full schema + question</text>
<text class="d-sub" x="600.0" y="84" text-anchor="middle">→ one SELECT</text>
<rect class="d-box" x="740" y="20" width="200" height="76" rx="10"/>
<text class="d-title" x="840.0" y="52" text-anchor="middle">SQLite</text>
<text class="d-sub" x="840.0" y="69" text-anchor="middle">read-only, ≤500 rows</text>
<rect class="d-box" x="980" y="20" width="200" height="76" rx="10"/>
<text class="d-title" x="1080.0" y="52" text-anchor="middle">React UI</text>
<text class="d-sub" x="1080.0" y="69" text-anchor="middle">SQL + result table</text>
<path class="d-arrow" d="M220 58 H258" marker-end="url(#v1ah)"/>
<text class="d-label" x="240.0" y="50" text-anchor="middle">question</text>
<path class="d-arrow" d="M460 58 H498" marker-end="url(#v1ah)"/>
<text class="d-label" x="480.0" y="50" text-anchor="middle">prompt</text>
<path class="d-arrow" d="M700 58 H738" marker-end="url(#v1ah)"/>
<text class="d-label" x="720.0" y="50" text-anchor="middle">SQL</text>
<path class="d-arrow" d="M940 58 H978" marker-end="url(#v1ah)"/>
<text class="d-label" x="960.0" y="50" text-anchor="middle">rows</text></svg>`;

const TODAY_DIAGRAM = `<svg class="hld" viewBox="0 0 1200 730" role="img" aria-label="Scout today: the UI talks to FastAPI, which runs one question through six stages: asked before, pick tables, confirm meaning, check steps, three SQL tries, guards and run. The stages use the Groq LLM, a local embedding model, knowledge files, read-only SQLite databases and caches. Every call and failure is logged; failures become issues the Failure Analyst agent investigates."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="d-head"/></marker><marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="d-head-accent"/></marker></defs>
<rect class="d-box" x="400" y="16" width="400" height="64" rx="10"/>
<rect class="d-tag" x="410" y="25" width="22" height="17" rx="4"/>
<text class="d-tagt" x="421" y="37.5" text-anchor="middle">V1</text>
<rect class="d-tag" x="437.4" y="25" width="22" height="17" rx="4"/>
<text class="d-tagt" x="449" y="37.5" text-anchor="middle">V6</text>
<rect class="d-tag" x="464.79999999999995" y="25" width="30" height="17" rx="4"/>
<text class="d-tagt" x="480" y="37.5" text-anchor="middle">V10</text>
<rect class="d-tag" x="499.4" y="25" width="30" height="17" rx="4"/>
<text class="d-tagt" x="514" y="37.5" text-anchor="middle">V11</text>
<text class="d-title" x="600.0" y="48" text-anchor="middle">React UI</text>
<text class="d-sub" x="600.0" y="65" text-anchor="middle">Ask · Tables · Dashboard · Issues</text>
<path class="d-arrow" d="M560 80 V122" marker-end="url(#ah)"/>
<text class="d-label" x="552" y="106" text-anchor="end">question · pick · comments</text>
<path class="d-arrow" d="M640 122 V82" marker-end="url(#ah)"/>
<text class="d-label" x="648" y="106" text-anchor="start">options · steps · answer</text>
<rect class="d-box" x="400" y="124" width="400" height="56" rx="10"/>
<text class="d-title" x="600.0" y="146" text-anchor="middle">FastAPI</text>
<text class="d-sub" x="600.0" y="163" text-anchor="middle">/interpret · /plan · /query · /feedback · /issues</text>
<path class="d-arrow" d="M600 180 V228" marker-end="url(#ah)"/>
<text class="d-label" x="608" y="210" text-anchor="start">one question</text>
<rect class="d-group" x="16" y="230" width="1168" height="200" rx="14"/>
<text class="d-group-t" x="34" y="254">Scout pipeline · one question</text>
<rect class="d-stage" x="36" y="272" width="168" height="104" rx="10"/>
<rect class="d-tag" x="46" y="281" width="30" height="17" rx="4"/>
<text class="d-tagt" x="61" y="293.5" text-anchor="middle">V12</text>
<text class="d-title" x="120.0" y="324" text-anchor="middle">Asked before?</text>
<text class="d-sub" x="120.0" y="341" text-anchor="middle">saved answers</text>
<text class="d-sub" x="120.0" y="356" text-anchor="middle">embeddings + key words</text>
<rect class="d-stage" x="228" y="272" width="168" height="104" rx="10"/>
<rect class="d-tag" x="238" y="281" width="22" height="17" rx="4"/>
<text class="d-tagt" x="249" y="293.5" text-anchor="middle">V3</text>
<text class="d-title" x="312.0" y="324" text-anchor="middle">Pick tables</text>
<text class="d-sub" x="312.0" y="341" text-anchor="middle">embeddings</text>
<text class="d-sub" x="312.0" y="356" text-anchor="middle">top 4 + bridge tables</text>
<rect class="d-stage" x="420" y="272" width="168" height="104" rx="10"/>
<rect class="d-tag" x="430" y="281" width="22" height="17" rx="4"/>
<text class="d-tagt" x="441" y="293.5" text-anchor="middle">V5</text>
<text class="d-title" x="504.0" y="324" text-anchor="middle">Confirm meaning</text>
<text class="d-sub" x="504.0" y="341" text-anchor="middle">LLM writes ≤3 options</text>
<text class="d-sub" x="504.0" y="356" text-anchor="middle">user picks one</text>
<rect class="d-stage" x="612" y="272" width="168" height="104" rx="10"/>
<rect class="d-tag" x="622" y="281" width="30" height="17" rx="4"/>
<text class="d-tagt" x="637" y="293.5" text-anchor="middle">V10</text>
<text class="d-title" x="696.0" y="324" text-anchor="middle">Check steps</text>
<text class="d-sub" x="696.0" y="341" text-anchor="middle">LLM writes plain steps</text>
<text class="d-sub" x="696.0" y="356" text-anchor="middle">user flags wrong ones</text>
<rect class="d-stage" x="804" y="272" width="168" height="104" rx="10"/>
<rect class="d-tag" x="814" y="281" width="22" height="17" rx="4"/>
<text class="d-tagt" x="825" y="293.5" text-anchor="middle">V4</text>
<text class="d-title" x="888.0" y="324" text-anchor="middle">3 SQL tries</text>
<text class="d-sub" x="888.0" y="341" text-anchor="middle">LLM ×3, run, compare</text>
<text class="d-sub" x="888.0" y="356" text-anchor="middle">confidence high/med/low</text>
<rect class="d-stage" x="996" y="272" width="168" height="104" rx="10"/>
<rect class="d-tag" x="1006" y="281" width="22" height="17" rx="4"/>
<text class="d-tagt" x="1017" y="293.5" text-anchor="middle">V7</text>
<rect class="d-tag" x="1033.4" y="281" width="22" height="17" rx="4"/>
<text class="d-tagt" x="1045" y="293.5" text-anchor="middle">V8</text>
<rect class="d-tag" x="1060.8000000000002" y="281" width="22" height="17" rx="4"/>
<text class="d-tagt" x="1072" y="293.5" text-anchor="middle">V9</text>
<text class="d-title" x="1080.0" y="324" text-anchor="middle">Guards → run</text>
<text class="d-sub" x="1080.0" y="341" text-anchor="middle">read-only · size/time</text>
<text class="d-sub" x="1080.0" y="356" text-anchor="middle">error-book hints</text>
<path class="d-arrow" d="M204 324.0 H226" marker-end="url(#ah)"/>
<path class="d-arrow" d="M396 324.0 H418" marker-end="url(#ah)"/>
<path class="d-arrow" d="M588 324.0 H610" marker-end="url(#ah)"/>
<path class="d-arrow" d="M780 324.0 H802" marker-end="url(#ah)"/>
<path class="d-arrow" d="M972 324.0 H994" marker-end="url(#ah)"/>
<path class="d-arrow d-accent" d="M120 376 V406 H1080 V380" marker-end="url(#ahg)"/>
<text class="d-label d-accent-t" x="600" y="400" text-anchor="middle">saved meaning picked → re-run its SQL, 0 SQL calls (V12)</text>
<rect class="d-box" x="16" y="488" width="220" height="74" rx="10"/>
<text class="d-title" x="126.0" y="519" text-anchor="middle">Groq LLM</text>
<text class="d-sub" x="126.0" y="536" text-anchor="middle">gpt-oss-120b</text>
<rect class="d-box" x="262" y="488" width="180" height="74" rx="10"/>
<text class="d-title" x="352.0" y="519" text-anchor="middle">Embedding model</text>
<text class="d-sub" x="352.0" y="536" text-anchor="middle">mpnet, runs locally</text>
<rect class="d-box" x="468" y="488" width="250" height="88" rx="10"/>
<rect class="d-tag" x="478" y="497" width="22" height="17" rx="4"/>
<text class="d-tagt" x="489" y="509.5" text-anchor="middle">V2</text>
<rect class="d-tag" x="505.4" y="497" width="22" height="17" rx="4"/>
<text class="d-tagt" x="517" y="509.5" text-anchor="middle">V3</text>
<rect class="d-tag" x="532.8" y="497" width="22" height="17" rx="4"/>
<text class="d-tagt" x="544" y="509.5" text-anchor="middle">V6</text>
<rect class="d-tag" x="560.1999999999999" y="497" width="22" height="17" rx="4"/>
<text class="d-tagt" x="571" y="509.5" text-anchor="middle">V9</text>
<text class="d-title" x="593.0" y="532" text-anchor="middle">Knowledge files</text>
<text class="d-sub" x="593.0" y="549" text-anchor="middle">column descriptions · table</text>
<text class="d-sub" x="593.0" y="564" text-anchor="middle">summaries · error book</text>
<rect class="d-box" x="744" y="488" width="200" height="88" rx="10"/>
<rect class="d-tag" x="754" y="497" width="22" height="17" rx="4"/>
<text class="d-tagt" x="765" y="509.5" text-anchor="middle">V1</text>
<rect class="d-tag" x="781.4" y="497" width="22" height="17" rx="4"/>
<text class="d-tagt" x="793" y="509.5" text-anchor="middle">V7</text>
<text class="d-title" x="844.0" y="532" text-anchor="middle">SQLite databases</text>
<text class="d-sub" x="844.0" y="549" text-anchor="middle">attached read-only</text>
<rect class="d-box" x="970" y="488" width="214" height="88" rx="10"/>
<rect class="d-tag" x="980" y="497" width="22" height="17" rx="4"/>
<text class="d-tagt" x="991" y="509.5" text-anchor="middle">V8</text>
<rect class="d-tag" x="1007.4" y="497" width="30" height="17" rx="4"/>
<text class="d-tagt" x="1022" y="509.5" text-anchor="middle">V12</text>
<text class="d-title" x="1077.0" y="532" text-anchor="middle">Caches</text>
<text class="d-sub" x="1077.0" y="549" text-anchor="middle">results (paging) · answers</text>
<path class="d-arrow" d="M126 430 V486" marker-end="url(#ah)"/>
<text class="d-label" x="134" y="462" text-anchor="start">meaning · steps · SQL</text>
<path class="d-arrow" d="M352 430 V486" marker-end="url(#ah)"/>
<text class="d-label" x="360" y="462" text-anchor="start">embed question</text>
<path class="d-arrow" d="M593 488 V432" marker-end="url(#ah)"/>
<text class="d-label" x="601" y="462" text-anchor="start">read into prompts</text>
<path class="d-arrow" d="M844 430 V486" marker-end="url(#ah)"/>
<text class="d-label" x="852" y="462" text-anchor="start">run checked SQL</text>
<path class="d-arrow" d="M1077 430 V486" marker-end="url(#ah)"/>
<text class="d-label" x="1085" y="462" text-anchor="start">save · reuse</text>
<rect class="d-box" x="250" y="640" width="300" height="74" rx="10"/>
<rect class="d-tag" x="260" y="649" width="30" height="17" rx="4"/>
<text class="d-tagt" x="275" y="661.5" text-anchor="middle">V10</text>
<rect class="d-tag" x="294.6" y="649" width="30" height="17" rx="4"/>
<text class="d-tagt" x="309" y="661.5" text-anchor="middle">V11</text>
<text class="d-title" x="400.0" y="677" text-anchor="middle">Usage &amp; issues DB</text>
<text class="d-sub" x="400.0" y="694" text-anchor="middle">every LLM call, cost · failures as issues</text>
<rect class="d-box" x="650" y="640" width="330" height="74" rx="10"/>
<rect class="d-tag" x="660" y="649" width="30" height="17" rx="4"/>
<text class="d-tagt" x="675" y="661.5" text-anchor="middle">V11</text>
<text class="d-title" x="815.0" y="677" text-anchor="middle">Failure Analyst agent</text>
<text class="d-sub" x="815.0" y="694" text-anchor="middle">LLM + read-only tools, ≤8 calls</text>
<text class="d-sub" x="815.0" y="709" text-anchor="middle">code tests its fix · you approve</text>
<path class="d-arrow" d="M455 430 V638" marker-end="url(#ah)"/>
<text class="d-label" x="447" y="620" text-anchor="end">log calls &amp; failures</text>
<path class="d-arrow" d="M550 677.0 H648" marker-end="url(#ah)"/>
<text class="d-label" x="599" y="669.0" text-anchor="middle">issues</text>
<path class="d-arrow" d="M815 640 V578" marker-end="url(#ah)"/>
<text class="d-label" x="823" y="620" text-anchor="start">test queries</text></svg>`;

const PHASES = [
  {
    "title": "Make it work",
    "note": "Versions 1–3 · what the model reads",
    "versions": [
      {
        "no": 1,
        "title": "Baseline",
        "tagline": "question → SQL → table",
        "problem": "Nothing to measure yet. We needed the simplest thing that works end to end, to get a starting score.",
        "idea": "Give the LLM the whole schema and the question, let it write one query, and run it safely.",
        "added": [
          "Full schema + question → one SELECT",
          "Databases opened read-only",
          "UI shows the SQL and the result table"
        ],
        "result": "Worked end to end, but fell into the data's traps: dates, cents, city spellings."
      },
      {
        "no": 2,
        "title": "Database descriptions",
        "tagline": "what the columns mean",
        "problem": "The same data reads more than one way. Is 03/04/2026 March or April? Is 39999 dollars or cents? Are NY and NYC one city? Do cancelled orders count?",
        "idea": "Only the data's owners know. Write the answers down once, in BIRD's format, and put them where the model uses the column.",
        "added": [
          "One description file per table",
          "Each meaning placed beside its column in the prompt",
          "A check that descriptions match the data"
        ],
        "result": "Traps the owners know about stop being guesses. Question wording (\"Q2\", \"top customers\") is still open."
      },
      {
        "no": 3,
        "title": "Scaling to large databases",
        "tagline": "send only the tables that matter",
        "problem": "The whole schema went with every question. Fine for 10 tables; with hundreds the model confuses tables, runs out of room and costs more.",
        "idea": "RAG, applied to schemas: describe each table once, then find the closest tables to each question in milliseconds, with no LLM.",
        "added": [
          "An LLM summary per table, embedded once",
          "Top 4 tables per question + bridge tables for joins"
        ],
        "result": "20/20 questions got the right tables, sending 4.5 of 10 tables instead of all 10."
      }
    ]
  },
  {
    "title": "Make it right",
    "note": "Versions 4–6 · the SQL, the meaning, the tables",
    "versions": [
      {
        "no": 4,
        "title": "Trustworthy SQL",
        "tagline": "check, try 3 times, compare",
        "problem": "A query that runs can still be wrong, and in a real product there's no answer key to check against.",
        "idea": "Add every check code can do (no LLM judging an LLM), and ask the model 3 times: if the results agree, trust them more.",
        "added": [
          "Checks before and after running",
          "3 tries per round, 2 rounds, hard stop at 6 calls",
          "Confidence: high / medium / low",
          "A SQL parser to explain exactly how tries differ"
        ],
        "result": "4 of 5 trap questions right. #20 failed: tries disagreed on ties and on cancelled orders."
      },
      {
        "no": 5,
        "title": "Confirm the meaning first",
        "tagline": "ask before writing SQL",
        "problem": "#20 didn't fail on syntax. It failed on meaning: keep ties? count cancelled orders? More tries just guessed again.",
        "idea": "Only the user knows what they meant. Ask them before any SQL is written.",
        "added": [
          "Up to 3 meanings, each with its choices in plain words",
          "\"None of these\" + a comment → new options"
        ],
        "result": "#20: all 16 rows right, high confidence (was wrong, low). Options sometimes add filters nobody asked for."
      },
      {
        "no": 6,
        "title": "Know your tables",
        "tagline": "see and fix what Scout understands",
        "problem": "A wrong table description misleads every question. The orders description said to leave cancelled orders out, which is why #20 broke.",
        "idea": "Let the user see what Scout believes about each table, and correct it in plain words.",
        "added": [
          "Tables view: description, rows, columns, preview",
          "Fix a description; the LLM rewrites it; save re-embeds",
          "Add or remove a database, no restart"
        ],
        "result": "Two corrections to orders kept together in its new description, tagged Reviewed."
      }
    ]
  },
  {
    "title": "Make it safe and fast",
    "note": "Versions 7–9 · guardrails that don't trust the LLM",
    "versions": [
      {
        "no": 7,
        "title": "Safety (read-only)",
        "tagline": "never change data",
        "problem": "Scout must only read. An LLM is probabilistic, so safety can't depend on it behaving.",
        "idea": "Deterministic rules at every layer, so one miss is caught by the next.",
        "added": [
          "Question: write verbs stopped before any LLM call",
          "Meaning: the LLM must flag a write",
          "SQL: parser allow-list, one plain read only",
          "Database: read-only connection"
        ],
        "result": "17 write queries and 12 write requests blocked; traps like status = 'deleted' still allowed."
      },
      {
        "no": 8,
        "title": "Performance",
        "tagline": "heavy queries",
        "problem": "Some queries read too much, run too long or return more rows than anyone can use.",
        "idea": "Estimate the cost before running, limit it while running, and page the rows.",
        "added": [
          "Size estimate from the query plan: 1M rows max",
          "5 s time limit, 200 rows, 10 per page",
          "Recent results cached for paging",
          "Tries compared by a fingerprint of all rows"
        ],
        "result": "A 4-table cross join estimated at 6.25M rows was stopped before it ran."
      },
      {
        "no": 9,
        "title": "Error book",
        "tagline": "learn from SQL errors",
        "problem": "SQL habits from MySQL or PostgreSQL repeated on every retry: the retry only saw SQLite's raw error.",
        "idea": "Write down each known error once, with how to fix it in SQLite. A person decides what goes in.",
        "added": [
          "15 known errors → rules in the prompt, hints on retry",
          "Every error logged; unknown repeats stop early",
          "A plain failure message"
        ],
        "result": "The failing question answered in 3 calls, where it used to fail all 6."
      }
    ]
  },
  {
    "title": "Make it improve",
    "note": "Versions 10–12 · the user, the failures, the repeats",
    "versions": [
      {
        "no": 10,
        "title": "Check the steps, report issues",
        "tagline": "show how before answering",
        "problem": "Users couldn't see how Scout would answer, and when an answer was wrong nothing was kept to fix it.",
        "idea": "Show the plan before any SQL, in plain words, and collect reports after.",
        "added": [
          "Plain steps; comment on a wrong one, only steps re-run",
          "SQL must follow the steps",
          "Report issue → logged on the Dashboard",
          "Dashboard: calls, tokens, time, cost per question"
        ],
        "result": "#20 with one step corrected: 16/16 rows, only that step changed."
      },
      {
        "no": 11,
        "title": "Failure Analyst",
        "tagline": "an AI agent",
        "problem": "Wrong answers, reports and SQL errors sat in logs. Someone had to dig into each one, and every case is different.",
        "idea": "That's where an agent fits: an LLM that decides what to check, with tools that can only read. Code tests its fix; a person approves.",
        "added": [
          "Failures become issues: eval, user report, SQL error",
          "Agent loop, read-only tools, max 8 calls",
          "Code tests the fix; a failed test goes back once",
          "Approve or reject on the Dashboard"
        ],
        "result": "#20: found both causes after one failed test, 16/16 rows, 5 calls, $0.003."
      },
      {
        "no": 12,
        "title": "Reuse recent answers",
        "tagline": "same question, other words",
        "problem": "The same question in other words went through every step again: 4 to 7 LLM calls for an answer we already had.",
        "idea": "Keep recent answers. When a question is close, offer the saved meaning first; the user's pick confirms it's the same.",
        "added": [
          "10 saved answers; reported ones removed",
          "Match: embeddings + key words + no clash in numbers, months, scope",
          "Picked → saved SQL runs again, no SQL calls",
          "Report-only feedback"
        ],
        "result": "Asked again in other words: 0.45 s, 1 LLM call instead of 5, same 3 orders."
      }
    ]
  }
];

const OPEN = [
  ["Tested on a small database.", "10 tables of 50 rows and 20 questions; the BIRD benchmark hasn't been run."],
  ["The eval skips the newer steps.", "It doesn't confirm the steps, so it scores an older Scout."],
  ["Meanings can add filters.", "Options sometimes add choices the question never raised."],
  ["No score per version.", "We know what each version fixed, not how much each one moved accuracy."],
];

export default function ArchitectureTab() {
  return (
    <section className="arch">
      <div className="page-header">
        <h1>How Scout is built.</h1>
        <p>The design today, how it started, and what each version changed: the problem, what we thought, what we added, and what it measured.</p>
      </div>

      <div className="card arch-diagram">
        <div className="card-header">
          <h3 className="card-title">
            <Icon name="arch" />
            Today (Version 12)
          </h3>
          <span className="arch-legend">
            <span className="arch-vtag">V5</span> version that added it
            <span className="arch-shortcut" /> saved-answer shortcut
          </span>
        </div>
        <div className="card-body">
          <div className="arch-scroll" dangerouslySetInnerHTML={{ __html: TODAY_DIAGRAM }} />
          <p className="muted small">
            One question passes six stages, each added by a version, using the LLM, a local embedding model, the
            knowledge files, read-only databases and caches. When the user picks a saved meaning, Scout skips straight
            to running the saved SQL. Failures are logged as issues, and the Failure Analyst investigates them with
            read-only queries.
          </p>
        </div>
      </div>

      <div className="card arch-diagram">
        <div className="card-header">
          <h3 className="card-title">
            <Icon name="arch" />
            Version 1
          </h3>
        </div>
        <div className="card-body">
          <div className="arch-scroll" dangerouslySetInnerHTML={{ __html: V1_DIAGRAM }} />
          <p className="muted small">
            One LLM call wrote SQL from the whole schema; SQLite ran it read-only. Nothing checked the meaning, the tables
            or the SQL.
          </p>
        </div>
      </div>

      <div className="arch-key">
        <span><i className="arch-dot problem" />Problem</span>
        <span><i className="arch-dot idea" />What we thought</span>
        <span><i className="arch-dot added" />What we added</span>
      </div>

      {PHASES.map((phase) => (
        <div key={phase.title} className="arch-phase">
          <div className="arch-phase-head">
            <h2>{phase.title}</h2>
            <span className="muted">{phase.note}</span>
          </div>
          {phase.versions.map((v) => (
            <article key={v.no} className="arch-version">
              <div className="arch-badge">
                <small>Ver</small>
                <b>{v.no}</b>
              </div>
              <div className="card arch-card">
                <div className="arch-card-top">
                  <h3>{v.title}</h3>
                  <span className="muted">{v.tagline}</span>
                </div>
                <div className="arch-trio">
                  <div className="arch-panel problem">
                    <h4>Problem</h4>
                    <p>{v.problem}</p>
                  </div>
                  <div className="arch-panel idea">
                    <h4>What we thought</h4>
                    <p>{v.idea}</p>
                  </div>
                  <div className="arch-panel added">
                    <h4>What we added</h4>
                    <ul>
                      {v.added.map((a) => (
                        <li key={a}>{a}</li>
                      ))}
                    </ul>
                  </div>
                </div>
                <div className="arch-result">
                  <span className="label">Result</span>
                  {v.result}
                </div>
              </div>
            </article>
          ))}
        </div>
      ))}

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <Icon name="alert" />
            Still open
          </h3>
        </div>
        <ul className="card-body arch-open">
          {OPEN.map(([title, text]) => (
            <li key={title}>
              <strong>{title}</strong> {text}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
