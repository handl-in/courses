# Module 10 — Hallucinations

::: chapter-opener
<div class="module-num">MODULE 10</div>
<div class="module-title">Hallucinations</div>
<div class="subtitle">The model is calibrated to be helpful. Helpfulness, in the absence of grounding, becomes confident invention.<br>Agents make this worse — and also offer the best tools for fixing it.</div>
<div class="pages">~38 pages · why agents hallucinate, why agents catch it, what to do</div>
:::

::: hook
Sam was reviewing the customer-success bot's traces — the one that had just shipped with cross-session memory in Module 9. The metrics looked great. Repeat-customer NPS up 14 points. "As I have explained" messages way down. Memory operations clean.

Then Sam saw it.

Customer #1142, the original problem case. Ticket #4. The agent had read the customer's profile.md (correctly: *"plan: enterprise (effective ~2026-06-01, prior plan: pro)"*) and timeline.md (correctly: *"2026-05-15: customer announced upcoming upgrade to Enterprise plan"*). All the facts in memory were accurate.

The agent's response opened with: *"I see you've been with us since your Series B funding round, when you upgraded to Enterprise to support your growth..."*

There was no Series B. The customer's funding history wasn't in memory. The customer hadn't mentioned it in any prior ticket. The agent had taken the upgrade timing — May 2026, with no other context — and *decorated* it with a plausible-sounding business event. Series B was statistically the kind of thing that happens to companies upgrading to Enterprise. The model had pattern-matched and produced a plausible sentence that happened to be wholly fabricated.

The customer hadn't replied yet. Sam didn't know if they'd notice. Sam hoped they would notice — because the *next* time, the fabrication might be about something that mattered: a contract term, a regulatory requirement, a compliance constraint.

Memory had grounded the basic fact (when they upgraded). The model had added decoration the user would experience as a confident assertion. The grounding had reduced *some* hallucinations. It had not eliminated them. What was needed wasn't more memory — it was a different discipline entirely.

Sam wrote the line: *"the model is helpful. helpfulness without grounding becomes invention."*
:::

---

## What this module is

Modules 7–9 made Sam's agents context-disciplined. Tight system prompts. Clean tool results. Just-in-time retrieval. Within-trajectory compaction. Cross-session memory. By the end of M9, the customer-success bot had everything it needed — except the discipline of distinguishing what it *knew* from what it was *generating*.

This module is about that distinction. Where hallucinations come from, why agent hallucinations are a different class than chat hallucinations, and what to do about them — at the prompt layer, the architecture layer, and the verification layer.

The structure:

- **Section 1** — what hallucination is and why models do it (the helpfulness bias)
- **Section 2** — the categories of hallucination you'll see in production
- **Section 3** — agent-specific failure modes (memory decoration, tool result confabulation, action hallucination, citation fabrication)
- **Section 4** — Anthropic's canonical techniques: the "I don't know" out, direct quotes, citations, chain-of-thought
- **Section 5** — structural patterns: grounding-first prompts, retrieval shape, output structure
- **Section 6** — verification: post-generation checks, entity verification, self-RAG critics
- **Section 7** — the calibration problem: refusal vs helpfulness, when "I don't know" is the wrong answer too
- **Section 8** — putting it together: the customer-success bot, post-fix
- **Section 9** — the framework

By the end of this module:

- You'll recognize the hallucination categories that apply to your agent specifically (chat agents, tool-using agents, memory-enabled agents each have different patterns)
- You'll have implemented at least three of Anthropic's documented techniques in working code
- You'll know when grounding is enough and when verification is required
- You'll understand the trade-off between aggressive refusal and useful helpfulness — and how to land on the right side for your domain

This is the start of the reliability arc. M11 covers reflection and self-correction. M12 covers guardrails (input filtering, output policies). Together, M10–M12 turn Sam's agents from "well-architected" into "well-architected and trustworthy under adversarial or edge-case conditions."

::: pullquote
A 2025 Stanford analysis put hallucination rates between 22% and 94% depending on task and grounding. OpenAI's evals report rates dropping below 2% in retrieval-grounded tasks. The range is the story: hallucinations are not an inevitable model property — they're a function of how you ask, what you ground in, and what you verify.
:::

---

## Section 1 — What Hallucination Is, and Why

The Lakera 2026 definition is clean: *an AI hallucination occurs when a large language model produces output that looks plausible but is factually wrong or unsupported by evidence.*

Two parts of that definition matter. **Plausible** — the output passes a fluency check; nothing about the language flags a problem. **Unsupported by evidence** — the claim isn't grounded in anything the model legitimately had access to.

The mechanism is well understood at this point. Language models are trained to predict tokens that are *likely* given the preceding context. Likely is not the same as *true*. When the model has strong evidence (grounded retrieval, verbatim source, well-formed prior turn), likely tracks true closely. When the model lacks evidence, likely tracks *what would sound coherent* — which often means making something up.

This isn't a bug in the architecture. It's a consequence of the training objective combined with deployment in a setting where the model is expected to produce an answer.

Anthropic's framing in their official "Reduce hallucinations" doc is direct: *Claude tries to be helpful. It interprets every question as something it should answer, even when it genuinely lacks the information to do so. This helpfulness bias is the root cause of a huge percentage of hallucinations.*

The model could say "I don't know" but doesn't, because it was rewarded during training for producing useful-looking output. Producing "I don't know" feels like a failure to help; producing a confident-sounding answer feels like succeeding. The training pressure pushes toward confident output even when confidence isn't warranted.

That's why the *single most effective* mitigation — the one Anthropic puts first in their docs — is giving the model an explicit out. We'll get to that in Section 4.

### Three things hallucination is *not*

Worth disentangling before we go further:

**Hallucination is not lying.** A model that hallucinates isn't trying to deceive — it doesn't have intent in that sense. The February 2026 Disentangling Deception and Hallucination paper specifically distinguishes the two: hallucination is "knowledge gap filled with fabrication"; deception is "knowledge present but expressed misleadingly." Different mechanisms, different fixes.

**Hallucination is not a confidence problem.** Models that produce hallucinated output often report high confidence on it. The internal probability distribution doesn't reflect the gap between what's grounded and what's invented. You can't ask the model "are you sure?" and expect a useful answer.

**Hallucination is not solved by bigger models.** Frontier models hallucinate less than smaller ones, but the gap narrows as benchmarks saturate. Stanford's 2026 AI Index reports the top 5 models clustering between 10–20% hallucination rates on open-ended benchmarks; the gap to a 100B-parameter open-source model is 5–10 points, not 50. Throwing more capability at the problem helps but doesn't fix it.

The fix is at the *system* level: prompt structure, grounding sources, verification gates, refusal calibration. Section 4 onward.

::: nodumbq
**Q: If frontier models hallucinate 10–20% of the time, how is anyone shipping anything?**

The numbers cited are on *open-ended generation* benchmarks where the model has to produce content from training-data knowledge. With proper grounding (RAG over an authoritative corpus, structured tool results, well-curated context), production rates drop dramatically. OpenAI's own evals report sub-2% rates on retrieval-grounded tasks — a 5–10× improvement. The teams shipping reliably are the ones doing the grounding work. The teams shipping unreliably are the ones treating the model as a knowledge engine instead of a reasoning engine.
:::

---

## Section 2 — Categories of Hallucination in Production

A taxonomy that maps to the fixes in later sections:

**1. Factual fabrication.** The model invents a fact: a date, a number, a name, an event. Sam's "Series B funding round" is this category. The model produced a sentence that sounded right; the specific entity in it didn't exist.

**2. Citation fabrication.** A subtype of factual fabrication, but specifically about *sources*. The model invents references — paper titles, URLs, court cases, expert quotes — that don't exist. The Mata v. Avianca case in 2023 is the canonical example: a lawyer submitted a brief with six fabricated legal citations generated by ChatGPT. Adversarial benchmarks show citation-fabrication rates as high as 94%; this is not a rare edge case.

**3. Misattribution.** The model attributes a real fact to the wrong source. The fact is true; the source it cites is real; but the source doesn't actually contain the fact. Worse than fabrication in some ways — it passes simple "does the cited source exist?" checks but fails substantive verification.

**4. Inferential overreach.** The model takes grounded facts and extrapolates to claims that don't follow. The customer's plan tier is grounded; the agent's claim that they upgraded "for compliance reasons" extends beyond what was given. This is where most production hallucinations actually live — it's not pure invention, it's invention *on top of* a small kernel of truth.

**5. Memory decoration.** A specific subtype of inferential overreach for agents with memory. The agent reads a true fact from memory and embellishes when surfacing it. Sam's bot did this: the upgrade fact was real, the Series B framing was decoration. This is sneaky because the agent is technically grounded — it's just decorating beyond the grounding.

**6. Tool result confabulation.** The agent receives a tool result and reports something *adjacent to* but not actually in the result. The tool returned "user is in Plan B"; the agent says "user is in Plan B Enterprise tier" — the "Enterprise tier" wasn't in the result. Common when tool results are noisy or partial.

**7. Action hallucination.** Agent-specific: the agent reports an action was taken that wasn't, or describes the result of an action that didn't happen. "I've sent the email to the customer" — when no email tool was called, no email exists. Often shows up when agents hit safety refusals or tool failures and recover with confabulation.

**8. Schema confabulation.** The agent invents structured fields that weren't in the source. Asked to extract a JSON record from a document, it adds plausible-looking values for fields the document didn't contain. Common in extraction pipelines where the schema is constraining but the source is sparse.

For each of these, there's a different mitigation. The framework in Sections 4–7 will address all of them, but the priorities differ by which categories your specific agent encounters most.

::: brain
Sam's customer-success bot in the cold open hallucinated. Which category was it? What about a coding agent that says "I've fixed the bug" when the test still fails? What about a research agent citing "Smith et al. 2023" for a real claim, but the actual paper that supports the claim is by Jones et al?

(Sam's bot: memory decoration — true fact embellished. The coding agent: action hallucination — agent reports an outcome that didn't happen. The research agent: misattribution — fact correct, source wrong. Three different categories, three different fixes. Section 5 will cover the structural patterns; Section 6 the verification.)
:::

---

## Section 3 — Why Agent Hallucinations Are Different

Chat models hallucinate. We've known this since GPT-3. What's changed in the agent era is that the failure surface has *expanded*: agents have tools, memory, and trajectories that each introduce new opportunities for invention.

The HEAL paper (June 2025) makes the framing explicit for embodied agents: *"hallucinations in embodied agents stem from a failure to ground user-provided task instructions in the observed physical environment."* The example: a robot told to "put the knives in the dishwasher" when there is no dishwasher; the agent hallucinates a dishwasher into the plan and tries to "press buttons on a bare wall." The hallucination is no longer a textual error — it's an action with physical consequences.

For non-embodied agents, the principle generalizes: agent hallucinations stem from a failure to ground reasoning in the *actual state* of the system. That state includes:

**1. The state of memory.** What the agent has stored, what's true vs decorated. (Sam's bot.)

**2. The state of tool results.** What was returned, what wasn't. (Tool result confabulation.)

**3. The state of past actions.** What's been done, what's been attempted, what failed. (Action hallucination.)

**4. The state of the world the agent is acting on.** The customer's account, the codebase, the database, whatever the agent is interacting with.

Each of these is a layer where the agent can drift. And each compounds: an agent that confabulates a tool result on turn 3 and then reasons over that confabulation on turn 12 has built a tower of inference on a fabricated foundation. The downstream output looks coherent; the foundation is rotten.

### The compounding problem

A specific way agent hallucinations escalate:

```
Turn 3: Tool returns {"plan": "pro"}
Turn 3 agent: "User is on Pro plan. They might benefit from Enterprise features."
[Agent's own statement enters the conversation history.]

Turn 7: Agent reads earlier turns.
Turn 7 agent: "Since the user expressed interest in Enterprise features, let me 
check pricing." [Confabulation: user expressed no such interest. The agent
inferred it on turn 3 as a possibility, then read it back as a fact.]

Turn 12: Agent's plan now treats Enterprise interest as established.
Turn 12 agent: "I'll prepare the Enterprise upgrade summary. The user is
ready to convert."

[Compounding hallucination. Each turn was locally plausible. The end state
is wholly disconnected from the actual user state.]
```

This pattern — the agent's *own statements* re-entering its context as if they were facts — is one of the most insidious sources of agent hallucination. Module 8's compaction and Module 11's reflection both help; the structural fix in this module is to keep agent inferences clearly separated from grounded facts (Section 5).

### Why agents also offer the best tools for catching hallucinations

The flip side: agents have something chat models don't — the ability to *check*. An agent can call a verification tool, look up a fact, query memory, run a test. The same architectural feature that creates new failure surfaces also creates new mitigation surfaces.

Section 6's verification patterns lean on this. A chat model can be asked to self-critique its output (Anthropic's Best-of-N technique), but that's reasoning over the same context that produced the error. An agent can do something better: actually go check.

That's the productive framing for the rest of the module. Agent hallucinations are *worse* in their consequences (they cause actions, not just text) but *more tractable* in their fixes (you can verify in ways chat can't).

::: pullquote
The same architectural feature that creates new failure modes — agents have tools, memory, trajectories — also creates new mitigation surfaces. The agent that can hallucinate a fact can also call the tool that disproves it.
:::

---

## Section 4 — Anthropic's Canonical Techniques

Six techniques from Anthropic's official "Reduce hallucinations" documentation. Each is straightforward to apply; the combination is what produces meaningful reduction.

### Technique 1: Allow "I don't know"

The single most effective mitigation, and the one Anthropic puts first.

The mechanism: by default, the model treats every prompt as a request for an answer. Saying "I don't know" feels like a failure. Explicitly authorizing it changes the calibration.

```python
SYSTEM_PROMPT = """
You are a customer-success agent. Answer customer questions accurately based
on the customer context provided and the tools available.

CRITICAL: If you don't know something, or if the available context doesn't
support a claim you would otherwise make, say so explicitly. Examples of
good responses:
- "I don't have that information in our records."
- "I'm not certain — let me check by calling the lookup tool."
- "The context doesn't show me what plan they were on before. Should I
  pull their full history?"

NEVER fabricate facts to fill a gap. Saying "I don't know" or "I need to
check" is always preferred over inventing a plausible-sounding answer.
""".strip()
```

The instruction is doing several things:

- **Explicit permission** — the model knows uncertainty is acceptable
- **Concrete examples** — the model has phrasings to use, not just an abstract principle
- **Active alternative** — "let me check by calling the lookup tool" gives the agent a productive next step instead of just stalling
- **Negative example** — "NEVER fabricate" makes the wrong behavior explicit

Anthropic's interactive prompt engineering tutorial uses a clean example: ask Claude how many albums Beyoncé has released. If you ask without an "I don't know" option, the model often gives a confident but wrong number (the canonical wrong answer in the tutorial is "8" when the correct answer at that time was "7"). Add the explicit out, and the model either gives the right number or says it's not certain — but stops fabricating.

This single change, in production, often cuts hallucination rates by 30–50%. It's the highest-ROI thing in this entire module.

### Technique 2: Direct quotes for factual grounding

For tasks involving long documents (Anthropic specifically recommends >20K tokens), instruct the model to extract verbatim quotes *before* reasoning over the content.

The mechanism: when the model has to anchor its reasoning in actual text from the source, fabrication becomes harder — the fabricated content would have to exist verbatim in the source, which is a stronger constraint than "would sound plausible."

```python
PROMPT_WITH_QUOTE_GROUNDING = """
Below is a 30-page customer service transcript. The user wants a summary of
the customer's stated concerns.

Before writing the summary:
1. Extract 3–5 verbatim quotes from the transcript that capture the customer's
   primary concerns. Quote them exactly, in <quotes>...</quotes> tags.
2. Then, write the summary, referencing the quotes you extracted.

If you cannot find verbatim quotes that support a concern you would otherwise
include, do not include it. The summary must be grounded in the quoted text.

Transcript:
{long_transcript}

User question: What are the customer's primary concerns?
""".strip()
```

The structural shift: the model has to commit to *what's in the source* before reasoning *about* the source. Pure fabrications don't survive the quote-extraction step — they'd have to invent quotes too, which doesn't pass casual review.

### Technique 3: Verify with citations

Make the model's response auditable. Every claim cites the source for that claim. After generation, you (or a downstream verifier) can check whether the cited source actually supports the claim.

```python
PROMPT_WITH_CITATIONS = """
Answer the user's question based on the documents provided. For every claim
in your answer, cite the source using <cite>doc_id:line_number</cite> tags.

If you cannot cite a source for a claim, do not include it in your answer.
If the documents don't support an answer, say so.

After your initial answer, review each claim. If you cannot find an exact
supporting passage, REMOVE the claim or rephrase it as uncertain.
""".strip()
```

Two things to notice:

- **Citation as commitment.** The model has to point to evidence; "no evidence" forces "no claim."
- **The retraction step.** Asking the model to *review* its own answer and remove un-citable claims catches one round of fabrication that the citation requirement alone might miss.

This technique pairs especially well with retrieval systems where document IDs are stable and the source text is available for downstream verification (Section 6).

### Technique 4: Chain-of-thought verification

Ask the model to explain its reasoning step by step before the final answer. Faulty logic and ungrounded leaps tend to show up in the reasoning trace, where they can be detected.

```python
PROMPT_WITH_COT = """
Question: {user_question}

Before answering:
1. Identify what facts you need to answer this question.
2. For each fact, note where you would find it (memory, tool, document, or
   "this is general knowledge").
3. If any fact requires information you don't have access to, note that.
4. Then answer based only on facts you can verify.

Walk through your reasoning, then give the final answer.
""".strip()
```

CoT prompting was first popularized for reasoning tasks; its effect on hallucination is a useful side benefit. The model that has to *show* its work is more likely to notice when it doesn't have the information needed to do the work.

The downside: longer outputs, more tokens, slower responses. CoT is a quality lever, not a default. Apply it where stakes warrant the cost.

### Technique 5: Best-of-N verification

Run the model on the same prompt N times. Compare outputs. Inconsistencies signal hallucination.

The mechanism: a hallucinated detail tends to be *unstable* across runs — the model invents different details each time. A grounded fact tends to be *stable* — the model produces the same answer.

```python
import asyncio


async def best_of_n(prompt: str, *, model: str, n: int = 5) -> dict:
    """
    Run the prompt N times. Return the consistent components and flag
    inconsistencies for human review.
    """
    responses = await asyncio.gather(*[
        client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            # Non-zero temperature to surface inconsistencies
            temperature=0.7,
        )
        for _ in range(n)
    ])
    texts = [r.content[0].text for r in responses]
    return analyze_consistency(texts)


def analyze_consistency(texts: list[str]) -> dict:
    """
    For each claim that appears in any response, count how many responses
    contain it. High-agreement claims are likely grounded; low-agreement
    claims are likely hallucinations.
    """
    # Implementation: extract claims (entities, facts), cluster by similarity,
    # report agreement rate. Frameworks like RAGAS, TruLens, DeepEval do this.
    ...
```

Best-of-N is expensive (5× the inference cost). Reserve it for high-stakes outputs where the cost of an undetected hallucination outweighs the cost of N inferences.

### Technique 6: Iterative refinement

Use the model's outputs as inputs to a follow-up prompt that explicitly checks the previous output.

```python
async def verify_response(question: str, initial_response: str) -> str:
    verification = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                f"Original question: {question}\n\n"
                f"Initial response: {initial_response}\n\n"
                "Review the initial response. For each factual claim:\n"
                "1. Is it explicitly supported by the available context?\n"
                "2. If not, is it general knowledge that can be verified?\n"
                "3. If neither, mark it as 'unsupported.'\n\n"
                "Output a corrected response with unsupported claims removed "
                "or marked as uncertain."
            ),
        }],
    )
    return verification.content[0].text
```

Iterative refinement is essentially a single-step version of M11's reflection pattern. M11 will deepen this; for hallucination specifically, even one verification pass catches a meaningful fraction of fabrications.

::: nodumbq
**Q: Should I just always use all six techniques?**

No. Each adds tokens, latency, and (for Best-of-N) significant cost. The right combination depends on your stakes and constraints:
- **Low stakes, latency-sensitive:** Technique 1 (allow "I don't know") + careful prompt structure.
- **Medium stakes:** Add Technique 2 or 3 (quotes or citations) where you have grounded sources.
- **High stakes:** Add Technique 6 (iterative refinement) for post-generation verification.
- **Critical, expensive-to-be-wrong:** Add Technique 5 (Best-of-N) and human review.
Stack them deliberately, not by default.
:::

---

## Section 5 — Structural Patterns

The six techniques in Section 4 are about prompts. The structural patterns in this section are about *how the agent is built*. Bigger leverage; bigger investment.

### Pattern 1: Grounding-first prompts

Put the source material *before* the question, and explicitly mark its boundaries.

The Anthropic-recommended structure:

```
<context>
[Retrieved documents, memory contents, tool results — clearly marked]
</context>

<question>
[The actual user question]
</question>

Answer the question using only the information in <context>. If the
context does not contain the information needed, say so explicitly.
```

Why ordering matters: the model attends most carefully to what comes immediately before the response position. Putting the question last (closest to where the answer will be generated) and the source first creates the right attention pattern — the question is fresh; the source is in scope.

Anthropic's recommendation in their long-document handling documentation: *"Put documents first, question last. This improves retrieval performance."* The same recommendation reduces hallucination because it's fundamentally a grounding pattern.

### Pattern 2: Separate grounded facts from inference

The compounding hallucination problem from Section 3 (agent's own inferences re-entering context as facts) has a structural fix: tag what's grounded vs what's inferred.

```python
RESPONSE_TEMPLATE = """
Respond using this structure:

<grounded>
[Facts directly from the customer context, memory, or tool results.
Cite the source for each.]
</grounded>

<inferred>
[Conclusions you're drawing from the grounded facts. Mark each as
"likely" or "possible" rather than asserting it as fact.]
</inferred>

<answer>
[Your final response to the customer, using grounded facts confidently
and inferred conclusions cautiously.]
</answer>
"""
```

The structure forces the model to distinguish what it knows from what it's concluding. When that response gets compacted (M8) or re-read in a later turn, the structure persists — future-Claude can see that "Series B funding" was tagged as `<inferred>` rather than `<grounded>`, and know not to treat it as fact.

This pattern is heavyweight; not every interaction warrants the structure. It earns its place when the agent's outputs feed back into its own context across turns.

### Pattern 3: Strict output schemas

Schema confabulation (the "model invents JSON fields that weren't in the source" failure) is best fixed by making the schema strict and the source-not-found behavior explicit.

```python
EXTRACTION_PROMPT = """
Extract the following fields from the document. For each field:
- If the document explicitly contains the value, return it.
- If the document does not contain the value, return null.
- DO NOT infer values from related information. DO NOT pattern-match
  to plausible-looking values.

Schema:
{
  "company_name": string | null,
  "founding_year": integer | null,
  "headquarters_city": string | null,
  "ceo_name": string | null,
  "annual_revenue_usd": integer | null
}

Document:
{document_text}

Output the JSON object directly. No prose.
""".strip()
```

The discipline:
- Each field has an explicit `null` option for "not present"
- The instruction explicitly forbids inference
- Pattern-matching to plausible values is named as a forbidden behavior

For extraction tasks, this structure plus a downstream schema validator (Pydantic, JSON Schema) catches schema confabulation reliably. The model can still get individual values wrong, but it stops inventing fields wholesale.

### Pattern 4: Tool result discipline

Tool result confabulation comes from the agent reasoning over partial or noisy tool results. The fix happens at the tool layer (Module 4) and at the reasoning layer:

**At the tool layer:** tools should return structured, complete results with explicit "not found" / "not applicable" values rather than empty strings or omitted fields.

**At the reasoning layer:** the agent's prompts should instruct it to surface tool results verbatim before drawing conclusions:

```python
SYSTEM_PROMPT_WITH_TOOL_DISCIPLINE = """
When you receive a tool result, before incorporating it into your response:
1. State the result verbatim, in <tool_result>...</tool_result> tags.
2. Then state what you can conclude from the result.
3. If a field in the result is null, missing, or empty, do not infer
   what it would have been. State the gap explicitly.

Example:
<tool_result>
{"plan": "pro", "tier": null, "billing_cycle": "annual"}
</tool_result>
The customer is on the Pro plan with annual billing. The tier field is
null in our records — I'd need to confirm with the customer or check
another source if tier matters for the response.
""".strip()
```

The verbatim restatement is doing the work. The model that has to write down what the tool returned is more likely to notice when it later reasons about something that *wasn't* returned.

### Pattern 5: Memory provenance and decoration prevention

The Sam-customer-#1142 case from the cold open: the model decorated a grounded fact with an ungrounded embellishment. The fix is at the memory layer (M9) plus the reasoning layer.

Memory layer (review from M9): every entry has provenance — who stated it, when, from what source. Memory entries from "user-direct statement" weighted higher than from "inferred from tool result."

Reasoning layer: explicit instruction not to embellish.

```python
MEMORY_USAGE_GUIDANCE = """
When using information from memory:
- Surface the fact as it appears in memory. Do not embellish.
- If memory says "plan: enterprise (effective ~2026-06-01, prior plan: pro)",
  do NOT say "upgraded after their Series B" or "growth-driven upgrade" or
  any other framing not present in memory.
- The customer's history is what's in memory. Anything beyond that is
  fabrication, even if it sounds plausible.
""".strip()
```

The decoration prevention is fundamentally a refusal-to-helpfully-elaborate. The model that's trained to be helpful wants to provide context; this prompt explicitly trains it against that instinct for memory-derived facts.

::: gotcha
A subtle failure mode: you put all five structural patterns in your system prompt, which is now 4,000 tokens of carefully-crafted hallucination prevention. The agent's actual task instructions are now competing for attention with all this scaffolding. The Module 7 "right altitude" lesson applies: distill the structural patterns into the *minimum* set that addresses *your specific* hallucination categories. Memory decoration is your problem? Use Pattern 5; skip the others. Schema confabulation is your problem? Use Pattern 3. Don't apply patterns prophylactically.
:::

---

## Section 6 — Verification: Catching Hallucinations After the Fact

Prompt-level techniques (Section 4) and structural patterns (Section 5) reduce hallucination rates. They don't eliminate them. For high-stakes applications, post-generation verification catches the residual.

Three verification patterns, increasing in cost and rigor.

### Pattern A: Entity verification

For a generated response, extract the named entities (people, organizations, dates, financial figures, citations) and check them against the source material. Invented entities trigger a failure.

```python
async def verify_entities(
    response: str,
    source_context: str,
) -> dict:
    """
    Extract entities from the response. For each, check whether it appears
    in the source context. Return a structured report.
    """
    entities = await extract_entities(response)
    verified = []
    unverified = []
    for entity in entities:
        if entity_appears_in(source_context, entity):
            verified.append(entity)
        else:
            unverified.append(entity)
    return {
        "verified": verified,
        "unverified": unverified,
        "verification_passed": len(unverified) == 0,
    }


async def extract_entities(text: str) -> list[dict]:
    """
    Use a smaller model (Haiku) to extract entities. Cheap; specialized.
    """
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                "Extract all named entities from the following text: people, "
                "organizations, dates, dollar amounts, citations. Return as "
                "JSON: [{type, value}, ...].\n\n"
                f"Text:\n{text}"
            ),
        }],
    )
    return json.loads(response.content[0].text)
```

The verifier is doing exactly one thing — checking entities against the source. It's not a general "is this response correct?" verifier. The narrow scope makes it reliable.

For the Sam case, this would catch "Series B" (not in any source) but accept "Enterprise plan since June 2026" (in memory). The hallucination gets caught precisely because it's a fabricated entity, not an inference over real entities.

### Pattern B: Self-RAG critic

Anthropic's iterative refinement (Technique 6) generalized: a separate model call dedicated to critiquing the first one.

```python
async def self_rag_critic(
    user_question: str,
    initial_response: str,
    source_context: str,
) -> dict:
    """
    Run a separate model call to critique the initial response.
    The critic is given the source and asked to assess each claim.
    """
    critique = await client.messages.create(
        model="claude-opus-4-7",  # Stronger model as critic
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                "You are a fact-checking critic. The user asked a question. "
                "An assistant provided a response. The source material the "
                "assistant should have used is provided.\n\n"
                f"User question: {user_question}\n\n"
                f"Source material:\n{source_context}\n\n"
                f"Assistant response:\n{initial_response}\n\n"
                "Identify each factual claim in the assistant's response. For "
                "each claim, state whether it is:\n"
                "- SUPPORTED: directly stated in source material\n"
                "- INFERRED_REASONABLE: not stated but follows logically\n"
                "- INFERRED_OVERREACHING: not stated and doesn't strictly follow\n"
                "- UNSUPPORTED: not stated and contradicts or is unrelated\n\n"
                "Return JSON: {claims: [{text, status, reasoning}]}"
            ),
        }],
    )
    return json.loads(critique.content[0].text)
```

The critic uses a stronger model than the original generator. It's reading the response with a single focused task: identify unsupported claims. The cost: roughly equivalent to running the original prompt again, on a stronger tier — so 1.7× to 5× cost depending on which models.

For high-stakes outputs (legal advice, medical guidance, financial recommendations), the critic is justified. For routine outputs, it's overkill.

### Pattern C: Multi-agent verification

A specialized fact-checking agent runs alongside the primary agent. The fact-checker has access to authoritative sources (the company's database, primary documents, web search where applicable) and the primary agent's output. It signals when it can't verify.

The pattern is similar to M3's evaluator-optimizer workflow, applied to factual verification specifically.

```python
async def multi_agent_verified_response(user_question: str):
    # Primary agent generates a response
    primary_response = await primary_agent(user_question)

    # Fact-checking agent runs in parallel with access to sources
    fact_check_result = await fact_checking_agent(
        question=user_question,
        proposed_response=primary_response,
    )

    if fact_check_result.confidence < FACT_CHECK_THRESHOLD:
        # Either retry with stricter constraints, or escalate
        return retry_or_escalate(user_question, fact_check_result)

    return primary_response
```

This is heavyweight — two agents per request. Reserved for the highest-stakes settings: legal, medical, regulatory compliance, financial advice. Overkill for customer success bots; appropriate for systems where a hallucination has six-figure consequences.

::: postmortem
**The Brief That Cited a Paper That Didn't Exist**

A research-summary agent was helping users prepare academic-style briefs. Its outputs included citations — paper titles, authors, journals, years. Most citations were real and accurate.

A user submitted a brief generated by the agent to a peer who responded: *"I tried to find the Smith et al. 2024 paper you cited on neural-network interpretability. It doesn't exist. Smith publishes in this area but never wrote that paper."*

The team pulled the trace. The citation had been confidently inserted: real author, real area, real-looking journal, plausible year. The model had pattern-matched to "what a citation in this domain would look like" and produced a fabrication that passed a casual reader.

Two fixes shipped:

1. **Citation extraction and verification.** Every citation in agent output got extracted (Pattern A above). For each, the agent had to either point to the source document it came from in the retrieval results, or the citation got removed.

2. **Explicit instruction in the system prompt:** "Citations must be drawn from the documents provided in your retrieval context. Do not generate citations from training-data knowledge. If the user wants a citation that isn't in the retrieval context, say so."

The team also added a post-generation step: every citation in the output was checked against an academic-paper API. Citations that weren't findable triggered a re-run with the citation removed.

Citation-fabrication rate dropped from ~12% to <1% over two weeks. The pattern wasn't novel — Anthropic's docs had recommended citation-grounded generation since 2023 — but the team had been shipping with general-purpose prompts. The specific class of error required a specific class of mitigation.

**Lesson:** general hallucination mitigation isn't enough for citation-heavy outputs. Citations are a specific category that demands a specific verification pattern. Same for dates, financial figures, legal claims. Audit your output type for which categories matter; build the verification for those specifically.
:::

---

## Section 7 — The Calibration Problem: Refusal vs Helpfulness

A failure mode the techniques in Sections 4–6 *create*: over-refusal. An agent that's been heavily prompted to say "I don't know" when uncertain, to retract un-citable claims, to refuse extraction when fields are missing, can become unusably cautious.

The customer asks a perfectly reasonable question that the agent could answer with reasonable inference. The agent says "I don't have that information" and refuses to engage. The customer reasonably gets frustrated.

The aimagicx 2026 guide flagged this directly: hallucination reduction correlates with increased refusal rates. You can drive hallucination to near-zero by refusing to answer almost anything. That's not the goal.

The right calibration depends on stakes. Three regimes:

**Low-stakes regime (customer support, content recommendations, exploratory chat).** Default toward helpfulness. Use Technique 1 ("I don't know") for genuinely uncertain claims; otherwise let the agent answer with the natural variance models have. Hallucination here is annoying but recoverable. Rates of 5–10% on inferential claims are tolerable.

**Medium-stakes regime (technical support, code generation, structured data extraction).** Apply Patterns 1, 3, 4 from Section 5. Add Technique 6 (iterative refinement) for higher-stakes outputs. Target 1–3% hallucination on grounded claims; tolerate higher rates on inferential ones with appropriate caveats.

**High-stakes regime (legal, medical, regulatory, financial).** Full verification stack. Patterns 1–5 + entity verification + self-RAG critics + human-in-the-loop for the highest-impact outputs. Target near-zero on factual claims; tolerate "I cannot help with this without a human reviewing" on cases the system can't ground confidently.

The calibration trap teams fall into: applying the high-stakes regime to low-stakes use cases. Now your customer support bot refuses to acknowledge anything not literally in the customer record. The customer asks "can I export my data?" — a clearly-yes question with a clearly-yes answer per public docs — and the bot says "I don't have specific information about export capabilities for your account; please contact support." The hallucination rate is zero. The bot is also useless.

::: brain
A medical-information agent and a casual restaurant-recommendation agent both have hallucination problems. Should they get the same fix? Why or why not?

(No. Different stakes, different fixes. The medical agent should be in the high-stakes regime: full verification, conservative refusal, human escalation. A hallucinated drug interaction warning has six-figure-or-worse consequences. The restaurant agent is low-stakes: a wrong recommendation about whether a place takes reservations is annoying but recoverable. Apply Technique 1 ("I don't know" if not sure) and reasonable structural patterns; don't bury the experience in verification overhead. The cost of an incorrect restaurant recommendation does not warrant a fact-checking agent.)
:::

### The Goldilocks zone for refusal

A working principle: **the agent should refuse confident-sounding claims it can't ground, but not refuse questions it can reasonably answer.**

Practically, this looks like:

- **Strong refusal** for factual claims (named entities, specific dates, citations, dollar amounts) that aren't in the agent's grounded sources.
- **Caveated answer** for inferential claims that follow reasonably from grounded facts. ("Based on the upgrade pattern, the customer is likely growth-stage, though I don't have specifics on their funding history.")
- **Confident answer** for genuine general knowledge that doesn't depend on customer-specific facts. ("Yes, our Enterprise plan includes SSO support.")

The third category is where teams over-refuse. The agent should not say "I don't have information about whether the Enterprise plan includes SSO" if SSO is in fact a documented Enterprise feature in the agent's product knowledge base. That's not honesty; that's failure.

Calibration is iterative. Ship with reasonable defaults. Watch user feedback for both directions: hallucinations on one side, frustrating refusals on the other. Adjust prompts and verification gates to land in the zone where the agent is honest *and* useful.

---

## Section 8 — Putting It Together: Sam's Customer-Success Bot, Post-Fix

Returning to the cold open. Sam's bot had hallucinated "Series B funding round" — memory decoration on a grounded fact. What does the fix look like?

Sam applied the M10 framework:

**1. Identified the category.** Memory decoration — embellishment on top of grounded facts.

**2. Picked the right techniques.** Anthropic Technique 1 (allow "I don't know" / "I don't have that info"). Pattern 5 (memory provenance and decoration prevention from Section 5). Pattern A entity verification (Section 6) for high-stakes responses.

**3. Updated the system prompt:**

```
You are a customer-success agent. The customer may have history with us across
multiple support tickets.

When using information from memory or context:
- Surface facts as they appear. Do not embellish or add framing not present.
- For example, if memory says "plan: enterprise (effective ~2026-06-01)", do
  NOT add "after their Series B" or "growth-driven upgrade" or any framing
  not in memory.
- The customer's history is what's in memory. Anything beyond that is
  fabrication, even if it sounds plausible.
- If you don't have information about why something happened, why a customer
  did something, or other context not in memory, say so or simply omit it.
  "Welcome back" is better than "Welcome back since your Series B."

Acceptable: "I see you upgraded to Enterprise in June. How can I help?"
Acceptable: "I don't have details on what prompted the upgrade — let me know
if context is helpful."
Unacceptable: "I see you upgraded to Enterprise after your funding round."
(unless funding round is explicitly in memory)
```

**4. Added an entity-verification post-check** for outputs containing dates, dollar amounts, named events, or citations:

```python
HIGH_STAKES_ENTITY_TYPES = {"date", "money", "named_event", "citation", "person_name"}


async def respond_with_entity_check(
    customer_id: str,
    user_message: str,
    memory_context: str,
) -> str:
    response = await primary_response(customer_id, user_message, memory_context)
    entities = await extract_entities(response)
    risky = [e for e in entities if e["type"] in HIGH_STAKES_ENTITY_TYPES]
    if risky:
        verification = await verify_entities_against_source(risky, memory_context)
        if verification.unverified:
            # Re-run with explicit instruction not to mention the unverified entities
            response = await primary_response(
                customer_id,
                user_message,
                memory_context,
                exclusions=verification.unverified,
            )
    return response
```

**5. Tested on the original failure case.** Re-ran the Customer #1142 ticket #4 scenario. New response: *"Welcome back. I see you've been on Enterprise since June. How can I help?"* Clean. No fabricated funding round. No decoration. The customer would not be left wondering if their conversation history had been mis-recorded.

**6. Measured the impact.** Two weeks after rollout: in a sample of 200 randomly-selected production responses, manually annotated by Sam for hallucinations:

- Pre-fix: 18 responses contained at least one ungrounded inference (9%)
- Post-fix: 4 responses contained at least one ungrounded inference (2%)
- Refusal/non-answer rate: increased from 1.5% to 2.3% — the trade-off
- Customer satisfaction on the same response set: unchanged (the refusals were on cases where the original agent had hallucinated; either way, the customer wasn't getting actionable info)

The trade-off is real but worth it. The bot is now *honest* — it doesn't decorate. The slightly-higher refusal rate is on cases where the previous behavior was confidently wrong; saying "I don't have that info" is strictly better than fabricating.

::: code-exercise
**Exercise 10.1 — Hallucination audit.**

Pick an agent in your stack. Sample 30–50 production responses (or test responses, if you don't have production traffic yet). For each:

1. **Identify hallucination categories.** Which of the eight categories from Section 2 appear? For each occurrence, note which one.
2. **Assess severity.** How impactful is each hallucination? Annoying, problematic, or actively dangerous?
3. **Map to fixes.** For each frequent category in your sample, which techniques (Section 4) and structural patterns (Section 5) are best suited? Which require verification (Section 6)?
4. **Pick the highest-leverage fix.** What's the single change that addresses the largest fraction of your hallucinations? (Almost always Technique 1 — allow "I don't know" — if you haven't already done it.)
5. **Ship the fix; re-sample.** Did the rate drop? Did refusal/non-answer rate go up? Is the trade-off acceptable for your stakes?

The goal is not zero hallucinations. The goal is hallucinations that are *appropriate to your stakes* and detected/handled when they matter most.
:::

---

## Section 9 — The Framework

Adding hallucination discipline to a system you're building:

**1. Diagnose by category.** Audit a sample of outputs. Which of the eight categories show up? What's the severity? Don't apply general mitigations; apply category-specific ones.

**2. Apply Technique 1 first.** Allow "I don't know." Single highest-ROI change in this entire module. Most teams haven't done this explicitly; doing it cuts hallucinations 30–50% in many cases.

**3. Match stakes to regime.** Low-stakes → light prompt techniques. Medium-stakes → structural patterns plus iterative refinement. High-stakes → full verification stack including entity checks and critics.

**4. Ground the prompts structurally.** Documents-first, question-last. Verbatim quote extraction for long sources. Citation-required outputs where claims need to be auditable.

**5. Separate grounded from inferred.** Especially for agents whose outputs feed back into context. Tagged outputs prevent compounding hallucination.

**6. Verify the categories that matter.** Entity verification for fabrication-prone outputs (citations, dates, money). Self-RAG critics for high-stakes generation. Multi-agent verification for the small set of cases where it's justified.

**7. Calibrate refusal vs helpfulness.** Aggressive refusal is a failure mode too. Watch user-experience metrics, not just hallucination rates. Goldilocks zone: refuse the confident-sounding claims you can't ground; answer the questions you can reasonably answer.

**8. Iterate.** Hallucination patterns shift as your agent's surface area grows (new tools, new memory categories, new user behaviors). Quarterly audits keep the calibration current.

The discipline this enforces: **the model's helpfulness bias is treated as a known failure mode that requires structural mitigation, not a feature you trust by default.** The agent is honest by design, not by hope.

---

## Recap: Module 10 in eight bullets

::: bullet-points
- Hallucination is the model producing plausible-sounding but ungrounded output. The mechanism is the helpfulness bias from training: producing an answer feels like succeeding, even when the answer isn't grounded.
- Eight production categories: factual fabrication, citation fabrication, misattribution, inferential overreach, memory decoration, tool result confabulation, action hallucination, schema confabulation. Different categories need different fixes.
- Agent hallucinations are a distinct class. Tools, memory, and trajectories introduce new failure surfaces (memory decoration, action hallucination, tool result confabulation) that chat models don't have. Compounding effects make agent hallucinations especially insidious.
- Anthropic's six canonical techniques: allow "I don't know," direct quotes for grounding, citations for auditability, chain-of-thought verification, Best-of-N consistency, iterative refinement. Stack deliberately based on stakes.
- Structural patterns: grounding-first prompts (documents first, question last), separate grounded from inferred output, strict output schemas with explicit nulls, tool result verbatim discipline, memory provenance and decoration prevention.
- Verification patterns increase in cost: entity verification (cheap, narrow), self-RAG critics (medium, broad), multi-agent verification (expensive, comprehensive). Match the verification cost to the stakes of being wrong.
- Refusal is a failure mode too. Over-applied verification produces uselessly cautious agents. Calibrate to the Goldilocks zone: refuse ungroundable factual claims; answer questions you can reasonably address.
- Allow "I don't know" is the single highest-leverage change. Most teams haven't done it explicitly. Doing it before the rest of the module's techniques typically cuts hallucination rates by a third or more.
:::

---

::: sam-arc
**Sam, after the hallucination audit.**

A month after the M9 rollout, Sam ran the M10 audit across the company's eight memory-enabled agents. The pattern was consistent: rich grounded memory produced agents that were *more confident* than before — more likely to elaborate, more likely to decorate, more likely to invent context to make their grounded responses feel natural.

Six of the eight agents had measurable memory-decoration rates between 5% and 12% of responses. Two had citation fabrication rates above 8% (these were the two agents producing structured outputs like research summaries). One had action hallucination on 3% of responses where a tool failure had occurred and the agent had recovered with a confabulated success message.

Sam wrote a one-page hallucination playbook for the platform team:

1. Apply Anthropic Technique 1 (allow "I don't know") in every system prompt. Default.
2. For agents using memory, add the decoration-prevention pattern explicitly.
3. For agents producing structured outputs (extraction, JSON, citations), add schema strictness and entity verification.
4. For high-stakes agents (legal, financial, medical, anything where wrong has six-figure consequences), full verification stack including a self-RAG critic.
5. Audit quarterly. Watch both hallucination rates and refusal rates; calibrate.

Two weeks of fixes across the fleet. The before-and-after: hallucination rates roughly halved across the board. Refusal rates up modestly (1–3 percentage points). Customer satisfaction up — turns out customers preferred "I don't have that info" to confidently-wrong answers.

Sam noticed something deeper: the M10 work was the first reliability work where the *fix* was structural and the *measurement* was operational. M2's hardened loop you couldn't measure — it just worked or didn't. M9's memory had clear before/after metrics (NPS, "as I have explained" rate). M10 had even sharper metrics: hallucination rate, refusal rate, the trade-off curve. The reliability arc was going to live or die on this kind of measurement discipline.

Sam's arc this module: **the helpfulness bias is the enemy.** The model is trying to be useful. Without grounding, useful becomes invented. Production-quality agents need explicit structural mitigation against their own helpfulness — and ongoing measurement to keep the calibration honest. The next two modules (M11 reflection, M12 guardrails) extend this discipline to two adjacent surfaces: catching mistakes mid-trajectory, and constraining behavior at the boundaries.
:::

---

## What's next

Module 11 is reflection and self-correction. The pattern: agents that catch their own mistakes mid-trajectory. M3's evaluator-optimizer workflow lifted into the agent loop. When done well, dramatic quality improvements. When done poorly, just expensive.

The connection to M10: hallucination prevention happens at generation time; reflection happens after. The agent that produced a hallucinated response in M10 could, with reflection, catch it before delivering it. The two are complementary, not redundant.

Module 12 is guardrails — input filtering, output checks, topical and behavioral constraints. The cousin of Module 5's protocol-level security, applied at the per-call layer. The agent that's been told "don't fabricate citations" still needs an output-time check that verifies it didn't. M12 is where those checks live.

After M12, the reliability arc is complete. M13–15 take the system multi-agent (when complexity is justified, the architectures, the patterns). M16–17 are operations (sagas, observability). M18 is the capstone.

For now: take the audit exercise from Section 8. Sample 30 outputs from your agent. Categorize the hallucinations. Apply Technique 1 if you haven't. Watch the rate move. The first reliability instrument is honesty — about what your agent knows, and about what it's making up.
