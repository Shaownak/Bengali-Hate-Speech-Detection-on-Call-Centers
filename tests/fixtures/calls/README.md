# Synthetic fixture calls

Ten **synthetic** stereo 8 kHz calls (agent left, customer right), 2–4 min each. No real call data and no real slurs: abusive words are `<GALI_n>` placeholders from `configs/lexicon/fake_lexicon.yaml` (CLAUDE.md rule 2). See ADR-010.

| dir | scenario | events |
|---|---|---|
| demo_01 | Customer slang and shouting | slang ×2, shouting ×2 |
| demo_02 | Clean call (negative control) | none |
| demo_03 | Agent register drop আপনি → তুমি → তুই | disrespect_register ×2, dismissive_rude |
| demo_04 | Customer insult with interruptions | insult, overtalk ×2 |
| demo_05 | Sustained shouting, no profanity | shouting ×3, dismissive_rude |
| demo_06 | Agent profanity and insult | slang, insult (agent) |
| demo_07 | Customer threat, register drop to তুই | disrespect_register ×2, threat, shouting |
| demo_08 | Heated mutual overtalk | overtalk ×2, shouting ×2 (both sides) |
| demo_09 | Reported speech and self-directed (should not alert) | slang ×2, `alert: false` |
| demo_10 | Banglish code-switching with harassment | disrespect_register, sexual_harassment |

Each dir holds:
- `script.json` (committed): turns, word timings, voices, render parameters. This is the source of truth.
- `expected_events.json` (committed): ground truth (category, speaker, target, span, severity, flags).
- `call.wav` (git-ignored): rendered on demand, bit-for-bit reproducible.

```bash
make fixtures-wav                     # render missing call.wav files
make fixtures                         # regenerate everything after editing tools/fixtures_gen/scenarios.py
make simulate CALL=tests/fixtures/calls/demo_01 SPEED=10
```
