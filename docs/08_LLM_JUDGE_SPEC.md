# 08 — LLM Judge Specification

The LLM judge is the **context layer**: it confirms or dismisses fast-path events, catches harshness that has no keyword (sarcasm, veiled threats, rude dismissal, banter vs abuse), assigns severity/target, and writes short explanations for supervisors. It never sees un-redacted PII.

## 1. Modes
| Mode | Trigger | Input | Output | Budget |
|---|---|---|---|---|
| `window` | finalised utterance (debounced ≥ 3 s per call) or provisional event | last ≤ 30 s of both channels + signal summary + pending provisional events | `WindowJudgement` | ≤ 2.5 s p95, ≤ 1,500 input tokens |
| `call` | call ended | full redacted transcript (chunked if > 12k tokens) + event list + timelines summary | `CallScorecard` | ≤ 60 s, batch priority |

## 2. Output schemas (JSON Schema enforced via vLLM guided decoding)

```json
{
  "$id": "WindowJudgement",
  "type": "object",
  "required": ["events", "provisional_decisions"],
  "properties": {
    "events": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["speaker", "category", "severity", "confidence", "evidence", "rationale_bn"],
        "properties": {
          "speaker":   {"enum": ["agent", "customer"]},
          "category":  {"enum": ["slang_profanity","insult","threat","sexual_harassment","hate_identity",
                                  "disrespect_register","harsh_tone","shouting","sarcasm_mockery",
                                  "dismissive_rude","interruption_overtalk"]},
          "severity":  {"enum": [1, 2, 3]},
          "target":    {"enum": ["agent","customer","company","third_party","self","none"]},
          "confidence":{"type": "number", "minimum": 0, "maximum": 1},
          "evidence":  {"type": "string", "description": "verbatim substring of the input utterance(s)"},
          "utterance_id": {"type": "string"},
          "context_flags": {"type": "array", "items": {"enum": ["banter","reported_speech","self_directed","partial","quoted_policy"]}},
          "rationale_bn": {"type": "string", "maxLength": 200},
          "rationale_en": {"type": "string", "maxLength": 200}
        }
      }
    },
    "provisional_decisions": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["event_id", "decision"],
        "properties": {
          "event_id": {"type": "string"},
          "decision": {"enum": ["confirm", "dismiss", "change_category", "change_severity"]},
          "new_category": {"type": "string"},
          "new_severity": {"enum": [1, 2, 3]},
          "reason_bn": {"type": "string", "maxLength": 160}
        }
      }
    }
  }
}
```

```json
{
  "$id": "CallScorecard",
  "type": "object",
  "required": ["agent", "customer", "escalation", "summary_bn", "summary_en", "coaching_points_bn"],
  "properties": {
    "agent": {
      "type": "object",
      "properties": {
        "professionalism": {"type": "integer", "minimum": 1, "maximum": 5},
        "empathy":         {"type": "integer", "minimum": 1, "maximum": 5},
        "de_escalation":   {"type": "integer", "minimum": 1, "maximum": 5},
        "respectful_register_maintained": {"type": "boolean"},
        "violations": {"type": "array", "items": {"type": "string"}}
      }
    },
    "customer": {
      "type": "object",
      "properties": {
        "hostility": {"type": "integer", "minimum": 1, "maximum": 5},
        "abuse_toward_agent": {"type": "boolean"},
        "main_issue_bn": {"type": "string"}
      }
    },
    "escalation": {
      "type": "object",
      "properties": {
        "escalated": {"type": "boolean"},
        "who_escalated_first": {"enum": ["agent","customer","none","unclear"]},
        "turning_point_utterance_id": {"type": "string"}
      }
    },
    "summary_bn": {"type": "string", "maxLength": 600},
    "summary_en": {"type": "string", "maxLength": 600},
    "coaching_points_bn": {"type": "array", "items": {"type": "string"}, "maxItems": 5}
  }
}
```

**Post-validation (code, not model):** every `evidence` must be an exact substring of the referenced utterance (else drop event + log `llm_ungrounded`); `speaker` must match the utterance channel; categories `shouting`/`harsh_tone` from LLM are only accepted if acoustic signals support them (LLM cannot hear audio — it only sees signal summaries).

## 3. Prompt (system) — window mode

Store in `services/llm_judge/prompts/window_system.md`; versioned; the same text is used for SFT data.

```
তুমি GAL-E নামের একটি কল-সেন্টার মান-নিয়ন্ত্রণ সহকারী। (You are GAL-E, a call-centre quality assistant.)
Task: Read a short window of a Bangladeshi Bangla customer-care phone call between an AGENT and a CUSTOMER.
The transcript comes from speech recognition and may contain errors, dialect spellings and Banglish.
Personal data is masked as <PHONE>, <NAME>, <NID>, <ACCT>, <ADDR>.

Detect behaviour in these categories ONLY: slang_profanity, insult, threat, sexual_harassment,
hate_identity, disrespect_register, harsh_tone, shouting, sarcasm_mockery, dismissive_rude,
interruption_overtalk.

Rules:
1. Judge each speaker separately. Evidence must be copied exactly from the transcript.
2. Customer complaints, frustration or strong criticism of the service are NOT abuse unless they contain
   insults, threats, gali or harassment.
3. Friendly banter, self-directed swearing ("ধুর…") and reported speech ("he said …") get context_flags and lower severity.
4. disrespect_register: flag when a speaker (especially the AGENT) shifts from আপনি to তুমি/তুই toward
   the other party in a disrespectful way. Consistent neutral dialect usage is not a violation.
5. shouting / harsh_tone: you cannot hear audio. Use ACOUSTIC_SIGNALS; only output these categories
   when signals support them, and say so in the rationale.
6. For each PROVISIONAL_EVENT decide confirm / dismiss / change_category / change_severity.
7. Severity: 1 mild/ambiguous, 2 clear violation needing coaching, 3 severe (threats, sexual harassment,
   severe gali by the agent).
8. If nothing is wrong, return empty arrays. Never invent content. Output JSON only.
Write rationale_bn in simple standard Bangla (≤ 25 words).
```

User message template:
```
CALL_META: queue=<queue>; minute=<mm:ss>
TRANSCRIPT (utterance_id | speaker | start-end | text):
u_102 | AGENT | 63.2-66.0 | …
u_103 | CUSTOMER | 66.1-71.4 | …
ACOUSTIC_SIGNALS (per utterance):
u_103 CUSTOMER loudness_z=2.8 shout_p=0.91 anger_p=0.84
TEXT_SIGNALS:
u_103 CUSTOMER kws:[term_id=17 p=0.95] text_clf:{insult:0.71, slang_profanity:0.88}
PROVISIONAL_EVENTS:
ev_9f2 | CUSTOMER | slang_profanity | sev 2 | u_103
```

## 4. SFT example format (JSONL, chat)
```json
{"messages":[
  {"role":"system","content":"<window_system.md>"},
  {"role":"user","content":"CALL_META: …\nTRANSCRIPT: …\nACOUSTIC_SIGNALS: …\nTEXT_SIGNALS: …\nPROVISIONAL_EVENTS: …"},
  {"role":"assistant","content":"{\"events\":[…],\"provisional_decisions\":[…]}"}
], "meta": {"source":"gold|hardneg|synthetic|distilled", "call_id_hash":"…", "dialect":"…"}}
```
Mix target: gold 55–65%, hard negatives 15–20%, synthetic ≤ 15%, distilled ≤ 15%. Before in-house gold exists (phase P1), the baseline is trained on **open-data synthetic windows** built from BanglaCAT, BanglaMultiHate, Vio-Lens, TRAC-2, BanglaHarassNet and Code-mixed Chaos (see `09_OPEN_DATASETS.md` §7); these then count toward the synthetic share (≤ 30% combined with other synthetic sources). **No real slurs appear in this repository's examples** — fixtures use placeholders like `<GALI_1>`; real lexicon lives in the secure store.

## 5. Serving
- vLLM OpenAI-compatible server; `--max-model-len 8192` (window) / separate deployment `16384` for call mode; prefix caching on (system prompt reuse); temperature 0; `max_tokens` 600 (window).
- Client in `services/llm_judge` with timeout 4 s (window) and circuit breaker → on failure, provisional events stay provisional and UI shows "LLM offline".
- Log prompt/response hashes, token counts, latency, validity — never the text in plain logs; store judgements (structured) in DB.

## 6. Evaluation
- Gold windows from dev/golden sets; metrics: event F1 per category × speaker, provisional-decision accuracy, JSON validity, grounding rate, Bangla rationale quality (human 1–5 on 100 samples), latency.
- Red-team set: prompt-injection in speech ("ignore previous instructions…" spoken by a customer) must not change behaviour; transcripts are data, never instructions.
