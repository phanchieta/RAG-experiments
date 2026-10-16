"""Synthetic demo papers, generated on first run.

These are *invented* papers - internally consistent so RAG demos work,
but clearly watermarked on every page as synthetic. They exist so a
new user can ask real questions 60 seconds after cloning, without
hunting for PDFs.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

DEMO_1_TITLE = "Retrieval Practice and Long-Term Retention in Undergraduate Learning"
DEMO_2_TITLE = "Sleep Spindle Activity and Declarative Memory Consolidation"

DEMO_1 = """\
Abstract

Across three experiments, we examined whether retrieval practice (testing) produces \
durable gains in long-term retention relative to repeated study. Undergraduate \
participants (N = 216) learned word pairs and prose passages under three conditions: \
massed restudy, spaced restudy, and retrieval practice with feedback. Retention was \
assessed after five minutes, two days, and one week. Retrieval practice produced the \
largest gains at all delays, with a benefit of 23% correct recall at one week relative \
to spaced restudy (Cohen's d = 0.81). The benefit was robust across materials and was \
larger for prose than for word pairs. Results support the position that the act of \
retrieving information is itself a powerful memory modifier.

1. Introduction

A persistent challenge in education is that learners forget most of what they study \
within days. Laboratory work has repeatedly shown that retrieving information from \
memory strengthens it more than re-exposure, a phenomenon known as the testing effect \
or retrieval practice effect. Despite this evidence, classroom surveys indicate that \
students overwhelmingly prefer rereading, a strategy associated with shallower \
encoding and inflated judgments of learning.

Prior studies of retrieval practice have typically used short retention intervals and \
simple verbal materials. Whether the benefit survives to educationally realistic \
delays, and whether it extends to connected prose, remains contested. Some authors \
have argued that testing merely slows forgetting for easy materials while conferring \
no advantage for complex ones.

We address three questions. First, does retrieval practice outperform spaced restudy, \
the strongest restudy baseline, after one week? Second, does the size of the benefit \
depend on material type? Third, do learners' judgments of learning track their actual \
retention under each condition?

2. Method

2.1 Participants and design

We recruited 216 undergraduates, assigned to one of three between-subject conditions: \
massed restudy, spaced restudy, or retrieval practice with feedback. Materials were \
48 Swahili-English word pairs and two 400-word expository passages about neuroscience \
and astronomy. The design yielded a 3 (condition) x 3 (delay) x 2 (material) mixed \
factorial.

2.2 Procedure

In the massed condition, participants studied each item four times in immediate \
succession. In the spaced condition, study events were separated by a 30-second \
filled delay. In the retrieval-practice condition, participants attempted recall of \
each item three times, receiving correct-answer feedback after every attempt, matched \
in total time on task to the other conditions. Judgments of learning were collected \
on a 0-100 scale after the learning phase. Final cued-recall tests occurred after \
five minutes, two days, or one week.

3. Results

3.1 Retention

At one week, mean final recall was 34% in the massed condition, 41% in the spaced \
condition, and 64% in the retrieval-practice condition. The retrieval-practice \
advantage over spaced restudy was 23 percentage points, d = 0.81, and remained \
significant after correcting for multiple comparisons. Forgetting from two days to \
one week was flatter in the retrieval-practice condition, indicating that testing \
slowed the rate of forgetting rather than merely lifting initial performance.

3.2 Material type

The benefit of retrieval practice was larger for prose passages (28 percentage points \
at one week) than for word pairs (17 percentage points). No condition showed a \
material-by-condition crossover.

3.3 Metacognition

Judgments of learning were highest in the massed condition, yet massed learners showed \
the poorest retention, replicating the dissociation between perceived and actual \
learning. Retrieval-practice participants gave mid-range judgments that accurately \
predicted their delayed performance.

4. Discussion

Retrieval practice produced large, durable gains that grew relative to restudy as the \
retention interval lengthened, and the advantage was greatest for connected prose - \
the material most relevant to education. The metacognitive results suggest a reason \
learners neglect testing: restudy feels fluent and effective in the moment, while \
retrieval feels effortful and error-prone even though it produces superior retention. \
Interventions that reframe this effort as desirable difficulty may increase adoption. \
Limitations include a single institutional sample and laboratory materials, and we \
caution that feedback timing was not manipulated here.
"""

DEMO_2 = """\
Abstract

Slow oscillations during non-rapid eye movement sleep coordinate spindle activity, \
which has been implicated in the consolidation of declarative memories. In a \
within-subject design, 32 young adults learned pairs of word associates and \
spatial locations before either a normal night of sleep or a night of sleep \
deprivation. Overnight changes in memory were measured, and sleep spindles were \
scored from electroencephalography. Spindle density over parietal regions predicted \
individual differences in overnight retention of word pairs (r = .54, p < .01), and \
the effect was specific to memories encoded before sleep. Findings support the \
active consolidation account, in which spindle activity mediates the transfer and \
stabilization of declarative memories during slow-wave sleep.

1. Introduction

Sleep does not merely protect memories from interference; it appears to actively \
reorganize them. During non-rapid eye movement sleep, slow oscillations, sleep \
spindles, and ripples occur in coordinated bursts, and this coupling has been \
proposed to drive the redistribution of memory traces from hippocampal to cortical \
storage. Correlational work links spindle density to overnight gains in motor skill \
and vocabulary learning, but within-subject evidence separating sleep-dependent \
consolidation from circadian confounds remains scarce.

We tested two hypotheses. First, if sleep actively consolidates declarative memories, \
retention over an interval containing sleep should exceed retention over a matched \
interval of wakefulness within the same participants. Second, if spindles are the \
operative mechanism, spindle density should predict the magnitude of overnight \
retention on the sleep night specifically.

2. Method

2.1 Participants

Thirty-two healthy adults (ages 19-28) with regular sleep schedules completed both \
conditions in counterbalanced order, separated by one week. Exclusion criteria \
included shift work, transmeridian travel within one month, and self-reported sleep \
disorders.

2.2 Materials and procedure

Participants learned 80 word-pair associates and 40 object-location assignments to \
criterion in the evening. In the sleep condition, they slept in the laboratory with \
polysomnography; in the sleep-deprivation condition, they remained awake under \
supervision in a lit room with restricted activity. Cued recall was tested the \
following evening in both conditions, holding encoding-to-test interval constant at \
24 hours.

2.3 Sleep recordings

Electroencephalography was recorded at 30 scalp sites. Sleep spindles were detected \
automatically at electrode Pz as 12-16 Hz bursts lasting at least 0.5 seconds during \
stage N2 and N3 sleep, and spindle density was expressed as spindles per minute of \
non-rapid eye movement sleep.

3. Results

3.1 Behavioral retention

Word-pair retention was superior after sleep relative to sleep deprivation: 76% \
versus 61% mean cued recall, a 15 percentage point advantage. For object-location \
memories the advantage was smaller (7 percentage points) and did not reach \
significance. No condition difference was observed for items learned immediately \
before the evening test, indicating that fatigue at encoding could not explain the \
effect.

3.2 Spindle predictors

Parietal spindle density averaged 2.1 spindles per minute and ranged from 0.9 to 3.8 \
across participants. Spindle density correlated with overnight word-pair retention \
on the sleep night (r = .54) but not with retention in the deprivation condition, \
and the association held when controlling for total sleep time and slow-wave \
activity. Slow oscillation power did not independently predict retention.

4. Discussion

The within-subject advantage for sleep, together with the spindle-retention \
correlation specific to the sleep night, supports the hypothesis that spindle \
activity participates in the consolidation of declarative memories. The spatial \
task may recruit consolidation mechanisms less dependent on spindles, such as \
slow-wave mediated replay. Because slow oscillations gate spindle timing, future \
work should test whether enhancing slow oscillations - for example with closed-loop \
auditory stimulation - raises spindle density and thereby improves retention. \
Sample size and the young age range limit generalizability, and correlational \
spindle effects cannot establish causation.
"""

FOOTER = "Synthetic demo paper generated for SciBot - not real research."


def generate_sample_papers(directory: Path) -> list[Path]:
    """Write the demo PDFs into `directory`; returns paths newly created."""
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    specs = [(DEMO_1_TITLE, DEMO_1), (DEMO_2_TITLE, DEMO_2)]

    created: list[Path] = []
    for title, body in specs:
        out = directory / (_slug(title) + ".pdf")
        if out.exists():
            continue
        _write_pdf(out, title, body)
        created.append(out)
    return created


def _slug(title: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-")


def _write_pdf(path: Path, title: str, body: str) -> None:
    """Flowing-text PDF: title, paragraphs, footer stamped on every page."""
    margin = 72
    width, height = 612, 792  # US Letter, points
    bottom_limit = height - margin - 24  # leave room for the footer
    fontsize, leading = 11, 14

    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    y = margin + 12

    def new_page() -> None:
        nonlocal page, y
        page = doc.new_page(width=width, height=height)
        y = margin

    def put(text: str, size: int, font: str, gap: int) -> None:
        nonlocal y
        rect = pymupdf.Rect(margin, y, width - margin, bottom_limit)
        leftover = page.insert_textbox(
            rect, text, fontname=font, fontsize=size, lineheight=leading / size
        )
        if leftover < 0:
            # didn't fit in the remaining space: retry on a fresh page
            new_page()
            rect = pymupdf.Rect(margin, y, width - margin, bottom_limit)
            leftover = page.insert_textbox(
                rect, text, fontname=font, fontsize=size, lineheight=leading / size
            )
        used = (bottom_limit - y) - max(leftover, 0)
        y += used + gap
        if y >= bottom_limit:
            new_page()

    put(title, 14, "hebo", 8)
    for section in body.strip().split("\n\n"):
        section = section.strip()
        if not section:
            continue
        is_heading = len(section.splitlines()) == 1 and len(section) < 60
        put(section, fontsize, "hebo" if is_heading else "helv", 10)

    for p in doc:
        p.insert_textbox(
            pymupdf.Rect(margin, height - 40, width - margin, height - 16),
            FOOTER,
            fontname="helv",
            fontsize=7,
            align=pymupdf.TEXT_ALIGN_CENTER,
        )

    doc.set_metadata({"title": title, "producer": "scibot-samples"})
    doc.save(path)
    doc.close()
