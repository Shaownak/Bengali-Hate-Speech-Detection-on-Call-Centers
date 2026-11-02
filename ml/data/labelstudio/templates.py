"""Label Studio project templates T1–T3 (docs/02 §3.1), generated from the schema enums so the
taxonomy can never drift from ``galecore.schemas`` and docs/02 §3.4."""

from __future__ import annotations

import html

from galecore.schemas import Category, ContextFlag, Target

COLORS = {
    "slang_profanity": "#c0392b",
    "insult": "#d35400",
    "threat": "#8e44ad",
    "sexual_harassment": "#a5006e",
    "hate_identity": "#6d214f",
    "disrespect_register": "#b9770e",
    "harsh_tone": "#7f8c8d",
    "shouting": "#e74c3c",
    "sarcasm_mockery": "#16a085",
    "dismissive_rude": "#2c3e50",
    "interruption_overtalk": "#2980b9",
}
ANNOTATION_FLAGS = [f for f in ContextFlag if f is not ContextFlag.NEUTRAL_REGISTER] + [
    ContextFlag.NEUTRAL_REGISTER
]


def t1_transcript() -> str:
    """Per channel: ASR seed segments as regions; the annotator corrects each region's text."""
    return """<View>
  <Header value="T1 · প্রতিলিপি সংশোধন / Transcript correction — $channel"/>
  <Audio name="audio" value="$audio" hotkey="space" sync="text"/>
  <Labels name="segment" toName="audio" choice="single">
    <Label value="speech" background="#4a90d9"/>
  </Labels>
  <TextArea name="transcription" toName="audio" perRegion="true" editable="true" rows="2"
            placeholder="যা বলা হয়েছে হুবহু লিখুন; ব্যক্তিগত তথ্য &lt;PHONE&gt;, &lt;NAME&gt;… দিয়ে লিখুন"/>
  <Choices name="contains_slang" toName="audio" choice="single" showInline="true">
    <Choice value="contains slang"/>
  </Choices>
</View>
"""


def t2_events() -> str:
    """Stereo (agent L / customer R) waveform; time-range regions per channel labelled with the
    taxonomy; per-region severity, target, context flags, confidence and a note."""
    labels = "\n".join(
        f'    <Label value="{c.value}" background="{COLORS[c.value]}"/>' for c in Category
    )
    targets = "\n".join(f'      <Choice value="{t.value}"/>' for t in Target)
    flags = "\n".join(f'      <Choice value="{f.value}"/>' for f in ANNOTATION_FLAGS)
    return f"""<View>
  <Header value="T2 · ঘটনা চিহ্নিতকরণ / Event spans — call $call_id (left = agent, right = customer)"/>
  <Audio name="audio" value="$audio" splitchannels="true" hotkey="space"/>
  <Labels name="category" toName="audio">
{labels}
  </Labels>
  <View visibleWhen="region-selected">
    <Header value="Severity (1 mild · 2 clear · 3 severe)"/>
    <Choices name="severity" toName="audio" perRegion="true" required="true" choice="single" showInline="true">
      <Choice value="1"/><Choice value="2"/><Choice value="3"/>
    </Choices>
    <Header value="Target"/>
    <Choices name="target" toName="audio" perRegion="true" choice="single" showInline="true">
{targets}
    </Choices>
    <Header value="Context flags"/>
    <Choices name="flags" toName="audio" perRegion="true" choice="multiple" showInline="true">
{flags}
    </Choices>
    <Choices name="confidence" toName="audio" perRegion="true" choice="single" showInline="true">
      <Choice value="sure"/><Choice value="unsure"/>
    </Choices>
    <TextArea name="note" toName="audio" perRegion="true" rows="1" placeholder="note (optional, no personal data)"/>
  </View>
</View>
"""


def t3_call_qa() -> str:
    return """<View>
  <Header value="T3 · কল মূল্যায়ন / Call-level QA — call $call_id"/>
  <Audio name="audio" value="$audio" splitchannels="true"/>
  <Header value="Agent professionalism (1–5)"/>
  <Rating name="professionalism" toName="audio" maxRating="5" required="true"/>
  <Header value="Customer hostility (1–5)"/>
  <Rating name="hostility" toName="audio" maxRating="5" required="true"/>
  <Header value="Was the call escalated by agent behaviour?"/>
  <Choices name="escalated_by_agent" toName="audio" choice="single" required="true" showInline="true">
    <Choice value="yes"/><Choice value="no"/>
  </Choices>
  <Header value="De-escalation quality (1–5)"/>
  <Rating name="de_escalation" toName="audio" maxRating="5" required="true"/>
</View>
"""


TEMPLATES = {"T1": t1_transcript, "T2": t2_events, "T3": t3_call_qa}


INSTRUCTIONS = {
    "T1": "Correct each region's text exactly as spoken (dialect spelling allowed). Type personal data as tags: "
    "<PHONE>, <NID>, <ACCT>, <NAME>, <ADDR>. Tick 'contains slang' if any gali is spoken.",
    "T2": "Label the speaker, not the call: every region sits on one channel (left agent, right customer). "
    "Use the taxonomy of docs/02 §3.4; severity 1 mild, 2 clear violation, 3 severe. Quoted / reported "
    "speech, banter and self-directed swearing get a context flag. Single accidental overlaps are not "
    "interruption_overtalk (needs ≥ 3 in 60 s).",
    "T3": "Rate the whole call after listening to it. Alerts are advisory; ratings never trigger penalties.",
}


def project_title(task: str, batch: str) -> str:
    return f"GAL-E {task} · {batch}"


def instruction_html(task: str) -> str:
    return f"<p>{html.escape(INSTRUCTIONS[task])}</p>"
