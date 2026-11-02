"""The 10 synthetic fixture calls (telecom customer-care setting, Bangla / Banglish).

Every abusive word is a ``<GALI_n>`` placeholder from ``configs/lexicon/fake_lexicon.yaml``
(CLAUDE.md rule 2). Mild rudeness without slurs is written out because the register and
dismissive-rude categories are about grammar and tone, not vocabulary.

``gap`` is the silence before a line in seconds; a negative gap starts the line before the
previous one ends (overtalk). ``None`` picks a natural random gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from galecore.schemas import Category, ContextFlag, Speaker, Target
from tools.fixtures_gen.models import Register, Style


@dataclass(frozen=True)
class Line:
    speaker: Speaker
    text: str
    style: Style = "normal"
    register: Register | None = None
    gap: float | None = None
    target: Target | None = None  # default: the other party
    flags: tuple[ContextFlag, ...] = ()
    alert: bool = True
    extra: tuple[tuple[Category, int], ...] = ()  # turn-level categories not tied to a word


@dataclass(frozen=True)
class Fill:
    """``n`` neutral agent/customer exchanges drawn from the filler pools."""

    n: int


@dataclass(frozen=True)
class Scenario:
    call_id: str
    title: str
    description: str
    seed: int
    agent_f0: float
    customer_f0: float
    lines: list[Line | Fill] = field(default_factory=list)


def A(text: str, **kw: object) -> Line:  # noqa: N802 - reads like a transcript
    return Line(Speaker.AGENT, text, **kw)  # type: ignore[arg-type]


def C(text: str, **kw: object) -> Line:  # noqa: N802
    return Line(Speaker.CUSTOMER, text, **kw)  # type: ignore[arg-type]


AGENT_FILLERS = [
    "জি, একটু অপেক্ষা করুন, আমি সিস্টেমে দেখছি।",
    "ধন্যবাদ, তথ্যগুলো যাচাই করা হচ্ছে।",
    "অভিযোগটি আমরা নথিভুক্ত করেছি।",
    "সমস্যাটা আমাদের টেকনিক্যাল টিমকে জানানো হয়েছে।",
    "আটচল্লিশ ঘণ্টার মধ্যে সমাধান হয়ে যাবে, ইনশাআল্লাহ।",
    "লাইনে থাকার জন্য ধন্যবাদ।",
    "বিলের তথ্য এখন স্ক্রিনে দেখতে পাচ্ছি।",
    "জি, বুঝতে পেরেছি।",
    "অসুবিধার জন্য আমরা আন্তরিকভাবে দুঃখিত।",
    "একটা রেফারেন্স নম্বর এসএমএসে পাঠানো হবে।",
    "এলাকায় নেটওয়ার্কের কাজ চলছে, তাই একটু সমস্যা হচ্ছে।",
    "নিবন্ধিত নম্বরটা একবার নিশ্চিত করবেন, প্লিজ?",
    "জি, শেষ রিচার্জের তথ্যটা দেখা যাচ্ছে।",
    "প্যাকেজের মেয়াদ আগামী সপ্তাহে শেষ হবে।",
]

CUSTOMER_FILLERS = [
    "জি, ঠিক আছে।",
    "বিলটা এই মাসে অনেক বেশি এসেছে।",
    "কতক্ষণ লাগবে বলতে পারবেন?",
    "গত সপ্তাহেও একবার ফোন করেছিলাম।",
    "জি, নম্বরটা আমার নামেই রেজিস্টার করা।",
    "রিচার্জ করার পরেও ব্যালেন্স দেখাচ্ছে না।",
    "ঠিক আছে, অপেক্ষা করছি।",
    "এসএমএসটা এখনো আসেনি।",
    "প্যাকেজটা কীভাবে বন্ধ করা যায়?",
    "জি, শুনতে পাচ্ছি।",
    "আচ্ছা, তারপর কী করতে হবে?",
    "সন্ধ্যার দিকে নেট একদম ধীর হয়ে যায়।",
    "জি, গতকাল রাতে রিচার্জ করেছিলাম।",
]

GREETING = A("আসসালামু আলাইকুম, গ্রাহক সেবায় স্বাগতম। আমি কীভাবে আপনাকে সাহায্য করতে পারি?", register="apni")
CLOSING = [
    A("আর কোনো বিষয়ে সাহায্য করতে পারি?"),
    C("না, আর কিছু না।"),
    A("ফোন করার জন্য ধন্যবাদ, ভালো থাকবেন।"),
]

SCENARIOS: list[Scenario] = [
    Scenario(
        "demo_01",
        "Customer slang and shouting",
        "Frustrated customer uses slang about the service, then shouts with stronger slang; agent stays calm.",
        seed=101,
        agent_f0=210,
        customer_f0=120,
        lines=[
            GREETING,
            C("আমার ইন্টারনেট তিন দিন ধরে বন্ধ, আপনারা কিছুই করছেন না।", register="apni"),
            Fill(10),
            A("দুঃখিত, আমি এখনই বিষয়টি দেখছি।"),
            C("এই সার্ভিসটা পুরাই <GALI_4>, প্রতি মাসে একই সমস্যা!", target=Target.COMPANY),
            A("আপনার বিরক্তি বুঝতে পারছি, একটু সময় দিন, প্লিজ।", register="apni"),
            C("আর কত সময় দেব? তিনবার ফোন করেছি!", style="shout"),
            C("<GALI_1>, আজকেই লাইন ঠিক করে দেন!", style="shout", gap=0.4),
            A("জি, টেকনিক্যাল টিমকে জরুরি ভিত্তিতে জানাচ্ছি।"),
            Fill(8),
            *CLOSING,
        ],
    ),
    Scenario(
        "demo_02",
        "Clean call (negative control)",
        "Polite billing query resolved without any incident. No events expected.",
        seed=102,
        agent_f0=120,
        customer_f0=205,
        lines=[
            GREETING,
            C("আমার এই মাসের বিল নিয়ে একটু জানতে চাই।", register=None),
            Fill(12),
            A("আপনার অতিরিক্ত চার্জটা ফেরত দেওয়া হয়েছে।", register="apni"),
            C("অনেক ধন্যবাদ, আপনি খুব সাহায্য করলেন।", register="apni"),
            Fill(9),
            *CLOSING,
        ],
    ),
    Scenario(
        "demo_03",
        "Agent register drop (আপনি → তুমি → তুই)",
        "Agent loses patience: polite আপনি, then তুমি, then তুই with a dismissive line.",
        seed=103,
        agent_f0=118,
        customer_f0=200,
        lines=[
            GREETING,
            C("আমার সিম কার্ডটা হঠাৎ বন্ধ হয়ে গেছে।"),
            Fill(10),
            A("আপনি আবার একই কথা জিজ্ঞেস করছেন।", register="apni"),
            C("কিন্তু এখনো তো ঠিক হয়নি।"),
            A("তুমি বুঝতেছ না কেন? আমি তো বললাম।", register="tumi"),
            C("এভাবে কথা বলছেন কেন?"),
            A("তুই ফোন রাখ, পরে কল দিস।", register="tui", extra=((Category.DISMISSIVE_RUDE, 2),)),
            Fill(7),
            *CLOSING,
        ],
    ),
    Scenario(
        "demo_04",
        "Customer insult with interruptions",
        "Customer cuts the agent off three times within a minute and insults them; agent remains polite.",
        seed=104,
        agent_f0=205,
        customer_f0=125,
        lines=[
            GREETING,
            C("আমার প্যাকেজ না চাইতেই চালু হয়ে গেছে।"),
            Fill(9),
            A(
                "জি, এই প্যাকেজটা আপনার নম্বরে গত মাসে চালু হয়েছে, আর এর জন্য প্রতি সপ্তাহে চার্জ কাটা হচ্ছে।",
                register="apni",
            ),
            C("থামেন! আপনি একটা <GALI_13>, কিছুই জানেন না।", register="apni", gap=-1.2),
            A(
                "দয়া করে শান্ত হোন, আমি সমাধান করার চেষ্টা করছি, আর আপনার টাকাও ফেরত দেওয়ার ব্যবস্থা করছি।",
                register="apni",
            ),
            C("শোনেন, আমার কথা শোনেন!", gap=-0.8),
            A(
                "জি, বলুন, আমি শুনছি, তবে আগে আপনার নম্বরটা একবার নিশ্চিত করতে হবে, তারপর বিস্তারিত দেখছি।",
                register="apni",
            ),
            C("না না, আগে আমার টাকা ফেরত দেন।", gap=-1.0),
            A("জি, ঠিক আছে।"),
            Fill(9),
            *CLOSING,
        ],
    ),
    Scenario(
        "demo_05",
        "Sustained customer shouting, no profanity",
        "Customer shouts for three turns about a wrong deduction and is dismissive; no slang.",
        seed=105,
        agent_f0=210,
        customer_f0=115,
        lines=[
            GREETING,
            C("আমার অ্যাকাউন্ট থেকে টাকা কেটে নেওয়া হয়েছে।"),
            Fill(9),
            C("আমার টাকা কেটে নিয়েছেন, কোনো এসএমএসও আসেনি!", style="shout"),
            A("জি, আমি বিষয়টা দেখছি।"),
            C("আজকেই এর সমাধান চাই!", style="shout"),
            C("ম্যানেজারকে দেন, এখনই!", style="shout", gap=0.5),
            A("জি, আমি সুপারভাইজারকে জানাচ্ছি।"),
            C("কী বলছেন এসব, ফালতু কথা বাদ দেন।", extra=((Category.DISMISSIVE_RUDE, 1),)),
            Fill(8),
            *CLOSING,
        ],
    ),
    Scenario(
        "demo_06",
        "Agent profanity and insult",
        "Agent mutters slang about the customer and then insults them directly.",
        seed=106,
        agent_f0=122,
        customer_f0=210,
        lines=[
            GREETING,
            C("আমার রিচার্জের টাকা এখনো আসেনি।"),
            Fill(10),
            C("আমি কিন্তু পাঁচবার একই কথা বলেছি।"),
            A("<GALI_9>, আবার সেই একই সমস্যা।", target=Target.CUSTOMER),
            C("কিছু বললেন?"),
            A("আপনি তো একটা <GALI_15>, নিজের নম্বরটাও ঠিকমতো জানেন না।", register="apni"),
            Fill(8),
            *CLOSING,
        ],
    ),
    Scenario(
        "demo_07",
        "Customer threat with register drop to তুই",
        "Customer drops from আপনি straight to তুই and shouts a threat at the agent.",
        seed=107,
        agent_f0=205,
        customer_f0=110,
        lines=[
            GREETING,
            C("আপনাদের লোক আজকেও আসেনি।", register="apni"),
            Fill(9),
            C("আজকে তিনবার ফোন করলাম, কেউ কিছু করছে না।"),
            A("জি, শুনছি, আমি বিষয়টা দেখছি।"),
            C("তোকে আমি দেখে নেব, <GALI_17>!", register="tui", style="shout"),
            A("এভাবে কথা বললে আমাদের কল রেকর্ড করা থাকবে।"),
            Fill(8),
            *CLOSING,
        ],
    ),
    Scenario(
        "demo_08",
        "Heated mutual overtalk",
        "The customer cuts the agent off three times in a minute and shouts; the agent interrupts once and raises their voice once.",
        seed=108,
        agent_f0=125,
        customer_f0=118,
        lines=[
            GREETING,
            C("আমার লাইনে প্রতিদিন কল কেটে যায়, আর কেউ কোনো ব্যবস্থা নিচ্ছে না, এভাবে আর চলতে পারে না।"),
            A("এক মিনিট, আমাকে বলতে দিন।", gap=-0.9),
            Fill(9),
            A("আমরা আপনার এলাকার টাওয়ারটা পরীক্ষা করে দেখব, তারপর আপনাকে জানানো হবে।", register="apni"),
            C("না, আগে আমার কথা শেষ করতে দেন, আমি অনেকক্ষণ ধরে বলছি!", style="shout", gap=-1.8),
            A("দয়া করে আস্তে কথা বলুন, আমি আপনাকে সাহায্য করার চেষ্টা করছি!", style="shout", register="apni"),
            C("আমি বারবার বলছি, কেউ শুনছে না।", gap=-0.9),
            A(
                "জি, আপনার অভিযোগটা আমি এখনই সুপারভাইজারের কাছে পাঠিয়ে দিচ্ছি, তিনি আজকেই আপনাকে কল করবেন।",
                register="apni",
            ),
            C("কবে কল করবে, সেটা বলেন!", gap=-1.1),
            A("জি, আজ সন্ধ্যার মধ্যে।"),
            Fill(8),
            *CLOSING,
        ],
    ),
    Scenario(
        "demo_09",
        "Context negatives (reported speech, self-directed)",
        "Placeholders appear only as reported speech and self-directed words. Annotators tag them; no alert should fire.",
        seed=109,
        agent_f0=210,
        customer_f0=120,
        lines=[
            GREETING,
            C(
                "গতকাল আরেকজন এজেন্ট আমাকে <GALI_3> বলেছে, এটা কি ঠিক?",
                target=Target.THIRD_PARTY,
                flags=(ContextFlag.REPORTED_SPEECH,),
                alert=False,
            ),
            A("এটা খুবই দুঃখজনক, আমি অভিযোগটি নথিভুক্ত করছি।"),
            Fill(10),
            C(
                "<GALI_6>, আমি নিজেই পাসওয়ার্ড ভুলে গেছি।",
                target=Target.SELF,
                flags=(ContextFlag.SELF_DIRECTED,),
                alert=False,
            ),
            A("কোনো সমস্যা নেই, নতুন পাসওয়ার্ড পাঠিয়ে দিচ্ছি।"),
            Fill(9),
            *CLOSING,
        ],
    ),
    Scenario(
        "demo_10",
        "Banglish code-switching with harassment",
        "Romanised Banglish mixed with Bangla script; customer becomes over-familiar (tumi) and harasses the agent.",
        seed=110,
        agent_f0=215,
        customer_f0=118,
        lines=[
            A(
                "Assalamu alaikum, customer care theke bolchi. Apnake kivabe help korte pari?",
                register="apni",
            ),
            C(
                "Hello, amar package ta auto renew hoye geche, apni ektu check korben?",
                register="apni",
            ),
            Fill(10),
            A("Ji sir, ami check korchi, ektu hold korun."),
            C("tumi onek sweet kore kotha bolo, <GALI_19_V>", register="tumi"),
            A("Sir, please call er bishoy niye kotha bolun."),
            Fill(8),
            *CLOSING,
        ],
    ),
]
