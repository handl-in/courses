# Module 7 — Context Engineering Principles

::: chapter-opener
<div class="module-num">MODULE 7</div>
<div class="module-title">Context Engineering Principles</div>
<div class="subtitle">The model has an attention budget. Every token you stuff in spends some of it.<br>What goes in the context window is the next architectural lever.</div>
<div class="pages">~38 pages · the natural progression of prompt engineering</div>
:::

::: hook
Wednesday, late afternoon. Sam was running the deal-research workflow on a particularly complex target — a holding company with twelve subsidiaries — when the brief came back oddly thin. The synthesizer had skipped the financial profile section. Just… omitted it. The brief was otherwise fine.

Sam pulled the trace. The synthesizer had received a 180,000-token context. Three parallel research workers had each returned ~40K tokens of findings. The orchestrator had added its plan. The system prompt was 8K. Tool definitions another 5K. A few rounds of intermediate context had piled up.

The synthesizer's prompt buried the financial-profile section requirement on line 412 of a 700-line user message. The model had attended to lines 1-200 carefully. By line 412, it was skimming. By line 600, it had latched onto whatever pattern looked most relevant and was producing the brief from there.

The model wasn't broken. The context was. There were 180,000 tokens of mostly-relevant information pointing in subtly different directions, and the actual instruction — *write the brief covering these specific sections* — was structurally indistinguishable from the surrounding noise.

Sam re-ran the synthesizer with a leaner context: the original instruction first, plus a curated 40K-token summary of the research instead of the full 120K. The brief came back complete. Better than complete — the synthesis was sharper because the model wasn't trying to hold 120K tokens of overlapping research in working memory.

Sam wrote a one-liner in their notes: *"the context isn't a bag you fill. It's an attention budget you spend."*

That night they read the Anthropic essay on context engineering. Highlighted half of it. Showed up Thursday morning ready to refactor.
:::

---

## What this module is

The first six modules treated the context window as a place where things go. The system prompt, the tool definitions, the conversation history, the tool results — all of it landed in the context window, and the model figured it out. That assumption worked for short trajectories with focused tasks. It stopped working as Sam's deal-research system handled longer trajectories with more parallel work.

Anthropic's framing in their September 2025 engineering post — *"context engineering as the natural progression of prompt engineering"* — is the lens for the rest of the book. Prompt engineering is about writing the right words. Context engineering is about deciding which words, retrieved from where, organized how, with what *not* in there, are most likely to produce the behavior you want.

This module covers the principles. Module 8 covers compaction and summarization (what to do when the budget overflows). Module 9 covers memory across runs (state that persists between trajectories). Together they form an arc: M7 is what to put in, M8 is what to take out, M9 is what to remember.

By the end of this module:

- You'll think of the context window as a finite attention budget, not unlimited storage
- You'll know the four levers for spending that budget well: system prompt, tools, examples, retrieval
- You'll understand the just-in-time vs pre-loaded retrieval trade-off and when each is right
- You'll have written a system prompt at the "right altitude" — the Goldilocks zone between brittle scripts and vague handwaves
- You'll be able to diagnose context rot (the Sam case) and know what to fix when it shows up

We're following Anthropic's own framing closely in this module because their post is the canonical reference and the field has converged on the vocabulary. Where the post leaves things at "principle," this module adds production code, specific patterns, and the failure modes that show up when you ship.

---

## Section 1 — Context as an Attention Budget

The mental model from the cold open is the right one. Every token you put in the context window draws from a finite attention budget. It's not "free until you hit the limit" — it's diminishing returns from the first token onward, with the curve getting steeper as the context grows.

Why? Two structural reasons.

**Architecture.** Transformers work on n² pairwise relationships across the context. Every token attends to every other token. As n grows, the model's ability to capture these relationships gets stretched thin. The math doesn't lie: a 100K-token context has 10 billion pairwise relationships; a 10K-token context has 100 million. The model's parameters don't scale with the context; the same fixed capacity is being asked to handle two orders of magnitude more relationships.

**Training distribution.** Models are trained on token sequences with a strong skew toward shorter contexts. Long contexts are *handled* by techniques like position encoding interpolation, but the model has fewer specialized parameters for context-wide dependencies than it has for local ones. Long-range reasoning is competent, but it's never as sharp as short-range reasoning — even on the same content.

The empirical phenomenon is called *context rot* — researchers at Chroma have published needle-in-a-haystack-style benchmarks showing that recall accuracy decreases as token count grows, across every model tested. Some degrade gently; some degrade sharply; none escape it.

The practical consequence: at any given moment, your model has an attention budget to spend on the current token's prediction. Every irrelevant token in the context is competing for that budget against the relevant ones. The bigger the haystack, the harder the needle.

::: pullquote
Context is a finite resource with diminishing marginal returns. Every new token introduced depletes the attention budget by some amount. The optimization isn't "fit more in"; it's "find the smallest set of high-signal tokens that produces the desired behavior."<br>— paraphrased from Anthropic's "Effective Context Engineering for AI Agents"
:::

### What this changes in practice

A few principles fall out of this framing immediately:

**Long context isn't free.** Sonnet 4.6 and Opus 4.7 both support 1M-token contexts. That doesn't mean using 1M tokens is the right call; it means the technical option exists. For most tasks, performance peaks well before the context limit and degrades from there.

**Token efficiency is a quality metric, not just a cost metric.** A 10K-token context that produces the right answer beats a 100K-token context that produces the same answer at the same cost — because the next conversation, on top of that, will be better with less stale baggage.

**More retrieval ≠ more useful.** Stuffing a RAG system to return 50 documents instead of 10 doesn't usually improve quality. It often degrades it because the model has more competing signals to attend to.

**Tool result discipline is context discipline.** Module 4's tool-result truncation patterns aren't just about bandwidth — they're about preserving the model's attention budget for the actual task. A tool that returns 80K tokens of customer record (from M4's postmortem) doesn't just cost money; it makes the next decision worse.

::: nodumbq
**Q: Doesn't this contradict the "1M context" pitch? If long context is bad, why ship it?**

It's not bad — it's *expensive in attention even when it's free in tokens*. The 1M context window is a useful capability for tasks that genuinely need it: full-codebase analysis, long-document review, complex multi-file refactoring. For those tasks, the alternative (chunking and re-retrieving) is worse. The capability earns its place when the task genuinely needs it. The mistake is using long context as a *default* — packing in everything you have because you can. Use long context when the task requires it; design for tighter context when it doesn't.

**Q: How do I know when context rot is hitting my system?**

The diagnostic is usually: the model produces output that's structurally fine but misses a specific instruction or fact that's clearly in the context. Like Sam's case — the synthesizer skipped the financials section despite being told to include it, because the instruction was buried at line 412. If you're seeing "the model didn't do X even though I told it to," and X was specified deep in a long context, you're looking at context rot, not model failure. The fix isn't a stronger model; it's a tighter context.
:::

---

## Section 2 — The Four Components of Context

Anthropic's essay structures context into a few components that are useful to name explicitly. We'll work with five:

**1. The system prompt.** The framing the agent reads before any user input. Defines role, tone, available tools, output requirements.

**2. Tool definitions.** Schemas and descriptions for every tool the model can call. Loaded into context once per request.

**3. Examples.** Few-shot demonstrations of desired behavior. Not always present, but powerful when used well.

**4. Message history.** The accumulating conversation — user messages, assistant responses, tool calls, tool results.

**5. Retrieved or attached content.** Documents pulled from RAG, files attached by the user, MCP resources injected by the host.

Each component has its own discipline. The system prompt has the "right altitude" problem. Tools have the bloat-and-overlap problem. Examples have the over-cherry-picking problem. Message history has the staleness problem. Retrieved content has the "too much vs not enough" problem.

The next sections take each in turn. The unifying principle: for each component, the question isn't "what's the most I can fit in?" but "what's the smallest set of high-signal tokens that gets the job done?"

---

## Section 3 — The System Prompt at the Right Altitude

The most consequential context-engineering decision in any agent system. The system prompt is the framing the agent re-reads on every turn (it's at the start of context every time the model is called). Get it wrong and every subsequent decision is shaped by the wrong frame.

Anthropic's essay names the failure modes precisely:

**One extreme: brittle hardcoded prompts.** Engineers cram complex if-else logic into prompts to elicit exact behavior. *"If the user asks about pricing, respond with the pricing format. If they ask about a feature, check the feature list. If they ask about both, prioritize pricing unless the feature is in category X..."* These prompts work for the cases they were written for and break for everything adjacent. Maintenance is brutal; every new edge case adds a new conditional.

**Other extreme: vague handwaves.** *"You are a helpful customer support assistant. Answer questions accurately."* The model gets no signal about what "accurately" means in this context, what tone is appropriate, what's out of scope. It defaults to its training distribution, which often isn't what the team intended.

**Right altitude: specific enough to guide, flexible enough to handle the cases not enumerated.** The Goldilocks zone. Concrete signals about role, scope, output structure, and edge cases that are likely. Heuristics rather than scripts.

A practical structure:

```python
SYSTEM_PROMPT = """
You are a deal-research synthesizer for an enterprise sales team. Your job is
to turn structured research findings into a 1500-word brief that an account
executive can read in five minutes before a customer call.

<output_structure>
The brief has these sections, in this order:
1. Executive Summary (3-4 sentences)
2. Financial Profile (revenue, growth, runway when known)
3. Leadership (current CEO/CFO, recent changes)
4. Market Position (positioning vs competitors)
5. Recent Developments (last 90 days)
6. Key Risks (3-5 items relevant to a deal)

Each section is 150-300 words. Do not skip sections. If a section's findings
are sparse, write what you have and note the gap explicitly.
</output_structure>

<voice_and_tone>
Confident, factual, no hedging or filler. Write for a busy reader who already
knows the basics of B2B SaaS. Avoid superlatives ("revolutionary", "world-class")
unless quoting source material directly.
</voice_and_tone>

<facts_and_evidence>
Use only the findings provided. Do not invent figures, names, or events. If
findings are silent on a topic, say so rather than filling the gap.

When findings disagree (e.g., two sources cite different revenue figures),
report both and flag the disagreement; do not pick one silently.
</facts_and_evidence>

<edge_cases>
- If findings indicate match_quality is "ambiguous" or "none" for the target
  company, return a brief stating this; do not produce a confident-sounding
  brief based on uncertain matches.
- If findings are present but very sparse (fewer than 5 facts total), return
  a brief noting this and recommending escalation rather than producing low-
  quality output.
</edge_cases>
""".strip()
```

The structure here:

- A one-paragraph framing that names the role, the consumer, and the success criterion (briefable in 5 minutes)
- Tagged sections for the things that need precision: structure, voice, fact discipline, edge cases
- No procedural conditionals; no "if the user asks X, do Y" scripts
- Edge cases listed as specific, named handlings rather than scattered through the body

The XML-style tags (`<output_structure>`, `<voice_and_tone>`, etc.) help the model find the right section when it's reasoning about a particular concern. They aren't strictly required — modern Claude handles unstructured prompts well — but they remain useful for system prompts that span multiple concerns.

::: brain
A team's system prompt is 4,000 tokens long with detailed conditional handling for every edge case they've encountered. They keep adding to it. Quality is fine; latency is creeping up. What's the diagnosis?

(The prompt has crossed from "right altitude" into "brittle hardcoded." Every conditional addition makes the prompt more rigid and less generalizable. The cost: every new case requires a new conditional; the prompt grows monotonically; eventually, conditionals interact in ways that break old behavior. The fix: refactor to heuristics. Replace the "if X then Y" rules with a description of the principle that *generates* those rules, and trust the model to apply it to novel cases. Test on real traffic, including the edge cases that originally drove the conditionals. You'll typically lose 30-60% of the prompt size with no quality regression.)
:::

### Iteration discipline

The right altitude isn't found in one draft. The discipline:

1. Write the minimal prompt — role, scope, output structure. Test on real tasks.
2. For each failure mode, ask: is this a prompt problem, or a model/data problem? Don't fix data problems with prompt patches.
3. When it is a prompt problem, prefer adding heuristics to adding conditionals. "Be cautious about X" beats "if X then Y."
4. Periodically prune. Delete every line that wasn't load-bearing on a recent failure mode. Re-test.

Sam's deal-research synthesizer prompt started at 200 tokens, grew to 1,200 after early failures, then settled at 800 after pruning. Every line in the final 800 was earning its place. Every removed line had been added defensively for a case that had since been handled differently.

::: gotcha
A common pattern: a team's system prompt grows by 50% every quarter as edge cases get patched in. Six months later, the prompt is 6,000 tokens of accreted history. The fix isn't a refactor session right before launch — it's a quarterly prompt audit. Reserve an hour per quarter to review the system prompt against current behavior. Delete anything that's no longer earning its place. The model will tell you what's still load-bearing if you re-test after pruning.
:::

---

## Section 4 — Tools as Context

Module 4 covered tool design from the contract perspective. Every word in this module about tools applies in addition: *tools are part of context, and they spend attention budget like everything else.*

The attention costs of tools:

**Tool descriptions are read every turn.** Every API call to the model includes the full tool list with descriptions. A 5K-token tool definition × 30 turns = 150K tokens spent on tool definitions alone (before caching, anyway).

**Tool overlap creates ambiguity.** If two tools could plausibly handle the same request, the model spends attention on the disambiguation. Three search tools with subtly different scopes is worse than one search tool with clear scope.

**Tool count strains decision-making.** Anthropic's guidance, echoed across production systems: agents with more than ~10-15 tools start showing meaningful tool-selection regressions. The model's attention is distributed; deciding among 30 tools is harder than deciding among 7.

The practical implications:

**Curate the tool set per agent.** Not every agent needs every tool. The deal-research workflow's synthesizer doesn't need a Slack-posting tool, even if Slack-posting is useful elsewhere in the company. Per-agent tool allow-lists keep context lean.

**Use the tool search pattern for large catalogs.** When an agent genuinely needs access to many tools (a Claude Code-style coding agent, for instance), don't load all of them. Load a "search-tools" tool that returns relevant tool descriptions on demand. The Anthropic tool search feature, available on the Claude Developer Platform, does exactly this.

**Cache tool definitions.** Prompt caching applies to the tool list. If your tools don't change between calls (the usual case), cache them — the cached read is roughly 10% the cost of the uncached read. Module 8 covers caching mechanics in depth; for now, know that it applies here.

**Watch for tool-result bloat.** From Module 4: tool results live in context for the rest of the trajectory. A tool that returns 20K tokens of result, called five times, is 100K tokens of context the model is dragging around forever. Truncate aggressively at the tool layer; the marker text Module 4 covered isn't just for honesty, it's for context efficiency.

::: nodumbq
**Q: Should I worry about tool definitions if I'm using prompt caching?**

Less, but not zero. Caching helps with cost (90% savings on cached input) but doesn't help with the *attention* budget — the model still attends to the cached tokens, even if you're paying less for them. So a 5K-token tool definition cached is still 5K tokens spent in the attention budget. Caching is a cost optimization, not a context optimization. Both matter; both have separate disciplines.
:::

---

## Section 5 — Examples (Few-Shot) Done Well

Few-shot prompting is one of the most reliable ways to steer behavior, and one of the most commonly misused. The misuse pattern: a team writes a prompt, hits an edge case, adds an example showing the right behavior for that case, hits another edge case, adds another example, and so on. After a few months the prompt has 14 examples covering every edge case the team has hit, and the prompt is 8K tokens long.

The result is *worse* than starting over. The 14 examples are usually narrow, idiosyncratic, and inconsistent in how they teach the underlying behavior. The model learns to pattern-match against the specific examples rather than the general principle they were supposed to demonstrate.

Anthropic's framing: *"work to curate a set of diverse, canonical examples that effectively portray the expected behavior of the agent."*

The discipline:

**Examples should teach the principle, not paper over edge cases.** A good example shows how the agent should reason about a class of cases, not how to handle one specific case.

**Few examples, well-chosen, beat many examples.** 3-5 diverse examples that span the meaningful range of behavior usually beat 15 examples that all illustrate similar ground.

**Edge cases that aren't generalizable belong in heuristics, not examples.** "If the user mentions a specific competitor, mention market positioning" is a heuristic, not an example. State it explicitly in the prompt; don't try to teach it through demonstration.

**Diversity matters more than density.** Three examples covering three different shapes of input/output beat ten examples on similar-shaped inputs. The model generalizes from variation.

A practical pattern for picking examples:

```python
EXAMPLES = [
    # Example 1: Standard case, success
    {
        "input": "Write a brief on Acme Corp.",
        "findings": "<typical findings>",
        "output": "<expected brief>",
    },
    # Example 2: Sparse findings, partial output
    {
        "input": "Write a brief on Bravo Industries.",
        "findings": "<sparse findings, only 4 facts>",
        "output": (
            "[Brief states the available facts, then explicitly notes:]\n"
            "Note: research returned limited data on Bravo Industries. "
            "Consider escalating before customer call."
        ),
    },
    # Example 3: Conflicting findings
    {
        "input": "Write a brief on Charlie Inc.",
        "findings": "<two sources disagree on revenue>",
        "output": (
            "[Brief reports both figures, e.g.:]\n"
            "Charlie Inc. reported $12M ARR per their Series B announcement; "
            "third-party data services estimate $18M. Sources disagree; "
            "verify with primary source before pricing discussions."
        ),
    },
]
```

Three examples spanning three meaningfully different cases: standard, sparse, conflicting. The model now has clear signals for each — without 15 examples and without the "every edge case has its own example" anti-pattern.

::: pullquote
Examples are the "pictures" worth a thousand words for an LLM. Three diverse, canonical examples beat fifteen narrow ones — every time.
:::

---

## Section 6 — Just-in-Time Retrieval

The biggest shift in context engineering between 2024 and 2026 has been the move from pre-loaded retrieval to just-in-time retrieval. The vocabulary is Anthropic's; the pattern is now mainstream.

**Pre-loaded retrieval (the older pattern):** Before the agent runs, embed-and-retrieve the most relevant documents. Stuff them into the system prompt or the first user message. The agent reasons over the retrieved set.

**Just-in-time retrieval (the agentic pattern):** The agent has tools that let it *fetch information on demand*. Instead of pre-loading 20 documents that *might* be relevant, the agent gets pointers (file paths, query helpers, list-and-search tools) and pulls the specific pieces it needs as it goes.

Both still exist. The shift is *which is the default*.

Anthropic's framing — and this is the part that lands hardest — is that just-in-time retrieval mirrors human cognition. *"We generally don't memorize entire corpuses of information, but rather introduce external organization and indexing systems like file systems, inboxes, and bookmarks to retrieve relevant information on demand."*

You don't keep your inbox in your head. You search it.

### When pre-loaded retrieval is right

Some content genuinely belongs in context up front:

- **Stable, frequently-needed reference material.** A coding agent's project conventions, an API agent's auth-token handling, a customer-support agent's escalation policy. Things every turn might need.
- **The user's immediate request and obvious context.** The current customer record for a support session; the deal in progress for a deal-research call.
- **Small reference sets where exhaustive pre-loading is cheaper than search.** If your knowledge base is 2000 tokens, loading it is fine; the cost of a search-and-retrieve dance is more than the cost of just having it.

The pattern: **pre-load the things every interaction will need. Retrieve everything else just-in-time.**

### When just-in-time wins

Most cases. Specifically:

- **Large corpuses where most content isn't relevant to a given query.** The codebase, the document library, the message archive. Pre-loading would dilute attention; the agent should search.
- **Dynamic content.** A pre-load is a snapshot; the agent's search reflects current state. For anything that changes during the agent's run (or between runs), just-in-time wins.
- **Branching exploration.** When the agent doesn't know what it needs until it sees what it found. The shape of the search depends on the input. You can't pre-compute.

Sam's deal-research workflow is a hybrid:

```python
# Pre-loaded into the orchestrator's context:
# - Deal info (target company name, deal stage, AE handling it)
# - Brief structure template
# - Recent customer notes (small, almost always relevant)

# Just-in-time, retrieved during the workflow:
# - Detailed company facts (via company_lookup tool)
# - Competitor analyses (via competitor_research tool)
# - Market context (via market_research tool)
# - Existing briefs on similar companies (via brief_search tool)
```

The 5K of pre-loaded deal info gets attended to on every step. The 50K of potentially-relevant research only enters context when (and if) the orchestrator decides it needs it.

### Implementing just-in-time well

The just-in-time pattern is only as good as the agent's ability to navigate the retrieval space. Three things have to be true:

**1. The agent has good tools for navigation.** A `list_files` + `read_file` pair beats a vector search for many code-search cases, because the model can see the structure (folder names, file names, sizes) and pick intelligently. For document-heavy cases, well-described search tools matter — Module 4's tool description discipline applies here in full.

**2. The agent has heuristics about where to look.** Folder structure, naming conventions, timestamps — Anthropic calls this "progressive disclosure." A file named `auth.py` in `src/middleware/` tells the agent something different than `auth.py` in `tests/`. The metadata is context the agent can use without loading the content.

**3. The agent isn't expected to find a needle in a haystack blindly.** If your retrieval space is genuinely unstructured and unsearchable, the agent will struggle. The fix is at the data layer (organize, index, name better), not the agent layer.

```python
# A just-in-time retrieval pattern: agent gets pointers, not content

@tool
async def list_briefs(
    target_company: str | None = None,
    industry: str | None = None,
    since_date: str | None = None,
    limit: int = 20,
) -> dict:
    """
    List recent deal briefs matching the filters. Returns metadata (company name,
    industry, date, deal stage), not full content. Use get_brief(brief_id) to
    fetch a specific brief's content.

    Use this WHEN: looking for relevant past briefs to inform the current research.
    Filter narrowly; broad listings are slow and crowd context.
    """
    briefs = _query_brief_metadata(target_company, industry, since_date, limit)
    return {
        "briefs": [
            {
                "brief_id": b.id,
                "company": b.company_name,
                "industry": b.industry,
                "date": b.date.isoformat(),
                "deal_stage_at_writing": b.deal_stage,
            }
            for b in briefs
        ],
        "total": len(briefs),
    }


@tool
async def get_brief(brief_id: str) -> dict:
    """
    Fetch the full content of a specific brief. Use sparingly — full briefs are
    1500+ words. List first with list_briefs(), then fetch only the briefs you
    actually need.
    """
    brief = _load_brief_content(brief_id)
    return {
        "brief_id": brief_id,
        "company": brief.company_name,
        "content": brief.full_text,
    }
```

The pattern: cheap metadata listing, expensive full retrieval. The agent sees the listing (which is small), picks the briefs it actually needs (usually 1-3), fetches just those. A workflow that previously pre-loaded all 20 candidate briefs into context (~30K tokens) now uses ~2K tokens of metadata + 1-3K per fetched brief = ~5K total. The model's attention is on the few relevant items, not the 17 distractors.

::: brain
You're building an agent that needs to answer questions about a 500-document legal corpus. Pre-load with embedding search? Just-in-time with a search tool? Hybrid?

(Almost certainly hybrid, leaning just-in-time. Pre-load the small, always-relevant pieces — the case description, the relevant statutes if known, the user's specific question. Give the agent a search tool over the 500 documents, plus a fetch tool. The agent searches based on what it learns, fetches the specific paragraphs that matter, doesn't drown in 500 documents of pre-loaded context. The pure pre-load fails because most documents won't be relevant to most queries. The pure just-in-time can fail if the agent doesn't have enough framing to know what to search for; that's why the framing parts get pre-loaded.)
:::

---

## Section 7 — Message History and the Trajectory Problem

The message history is the part of context that grows monotonically as the agent runs. Every turn adds the model's previous response and any tool results. By turn 30, the message history can dominate the context — 80%+ of tokens are messages, 20% are everything else.

Three principles for managing message history:

**1. Stale tool results are the biggest waste.** A tool result from turn 3 that's already been processed is consuming attention budget on turn 30, doing no work. The Anthropic Developer Platform's recent tool-result-clearing feature (covered in Module 8) handles this systematically; even without it, the discipline is "tool results have a useful life."

**2. The agent's own reasoning trace can dilute its current focus.** When the model has produced 20 turns of reasoning, the most recent reasoning often contradicts or refines earlier reasoning. The earlier reasoning is now noise; it's pointing the model toward conclusions it has since revised.

**3. Periodic compaction is the right default for long trajectories.** When the trajectory exceeds some threshold (Module 8 will cover specifics), summarize the message history into a compact form and continue with the summary plus the most recent activity. We won't deep-dive here; M8 is the home for compaction.

For Module 7, the principle is: **message history grows. You will have to manage it. Plan for it.**

The simplest pattern is a turn cap with graceful degradation. Module 2's hardened loop already had this — `max_turns`. The implication for context engineering: at the turn cap, the agent doesn't just stop; it hands off (to a summary, to a sub-agent, to the user) with what it has. Module 8 details the handoff mechanics.

---

## Section 8 — Retrieved and Attached Content

The fifth component of context: documents the user attached, MCP resources the host injected, retrieved chunks from a RAG system. These are the things the agent didn't ask for — they're handed to it.

The discipline here is mostly *yours, not the agent's*. The agent has to work with what it gets. You're deciding what it gets.

A few patterns that show up repeatedly:

**Attached documents: respect the structure.** When the user attaches a 50-page PDF, don't dump the raw text into context. Section it. Add headers. Preserve table boundaries. Number paragraphs if the agent might need to cite them. The structure costs almost no extra tokens and makes the model's downstream reasoning much sharper.

**RAG retrieval: filter, don't fan out.** Most retrieval systems return the top-K chunks by relevance score. Resist the urge to set K=20 because "more context is better." K=5-10 with high relevance is almost always better than K=20 with marginal relevance. If your relevance scoring isn't strong enough to make K=5 work, fix the relevance scoring before increasing K.

**MCP resources: attached only when the agent needs them.** From Module 5: resources are host-decided context attachments. Don't auto-attach everything; attach what's relevant to the current query. A `current_pipeline` resource for a deal-research agent is fine when the conversation is about pipeline; pointless when the agent is researching a specific company.

**Source citations preserve provenance.** Tag retrieved content with where it came from. Module 11's hallucination disciplines depend on this; without source-tagging, the agent has nothing to cite when asked.

```python
# Example: structured retrieved content rather than concatenated text

retrieved_context = """
<sources>
<source id="1" title="Mercer Industries 10-K" date="2025-12-31" type="filing">
[Relevant excerpt from the 10-K, ~500 tokens]
</source>

<source id="2" title="Mercer Industries press release" date="2026-03-15" type="news">
[Relevant excerpt, ~200 tokens]
</source>

<source id="3" title="Internal CRM record" date="2026-04-22" type="internal">
[Relevant excerpt, ~150 tokens]
</source>
</sources>

When citing these sources, reference them by id (e.g., "per source 1"). If
sources disagree, surface the disagreement and cite both.
""".strip()
```

The structured form preserves provenance, signals to the agent that these are external sources to cite, and uses about 5% more tokens than the raw concatenated text. The downstream quality lift — the agent can now reason about source credibility, dates, and conflicts — is significant.

::: gotcha
A common RAG bug: the retrieval system returns the top 10 chunks, but the chunks are formatted as one big concatenated blob. The agent has no idea which chunk supports which claim, can't tell when chunks disagree, and has no way to cite. The fix is structural: keep chunks separate, tag them, signal their boundaries. The cost is small; the quality gain is large.
:::

---

## Section 9 — Diagnosing Context Problems

When something's wrong in production, how do you tell if it's a context problem vs a model/prompt/tool/data problem?

Five diagnostic patterns:

**Symptom 1: The model misses an instruction that's clearly in the prompt.** Likely context rot. Check where in the context the missed instruction lives. If it's deep in a long context, move it earlier or make the context shorter.

**Symptom 2: The model produces inconsistent behavior across runs that "should" produce the same result.** Likely attention budget thrashing — the model's attention is being pulled by different parts of the noisy context on different runs. Tighten the context.

**Symptom 3: The model invents content that contradicts retrieved sources.** Could be context (the source got buried, the model didn't attend to it) or could be a hallucination problem (Module 11). Test by tightening the context — if the contradiction disappears, it was attention; if it persists, it's hallucination.

**Symptom 4: The model handles short cases well but degrades on long ones.** Almost always context rot. Apply just-in-time retrieval, summarize the history, scope the tool set per task.

**Symptom 5: The model gets confused about which tool to use.** Tool overlap or tool count problem. Audit the tool set; consolidate overlapping tools; remove rarely-used tools from the per-task allow-list.

::: code-exercise
**Exercise 7.1 — Audit your context.**

Pick an agent or workflow you've shipped. Pull a real production trace.

1. **Count the tokens by component.** System prompt, tools, examples, message history, retrieved/attached content. What proportion is each?
2. **Identify the high-value sections.** Which parts of the context are load-bearing for the model's output? Which are noise?
3. **Find the longest single block.** Tool result, retrieved document, message — whichever is biggest. Could you replace it with a smaller, more focused version (truncation, just-in-time alternative, structured filter)?
4. **Find the staleest content.** What's in context now that was relevant 10 turns ago and isn't relevant anymore? (Module 8 covers compaction; Section 7 here notes the principle.)
5. **Pick one change.** Implement it. Re-run on the same inputs. Compare quality and cost.

The goal isn't to optimize one trace — it's to build the calibration. Most production agents have 30-50% of their context consuming attention budget on stale or low-value content. The audit shows you where.
:::

---

## Section 10 — When to Use a Long Context Anyway

A short, honest section. Context engineering isn't always about minimizing — sometimes the right call is *use the long context*.

The cases:

**Single-pass document analysis.** A 200-page legal contract being reviewed once. Loading the whole thing into a 200K-token Sonnet/Opus context, asking "what are the unusual clauses?", getting the answer. The model gets full attention to the document at the cost of one big call. Cheaper and faster than chunking and aggregating.

**Whole-codebase reasoning.** A coding agent answering "where is this function used?" needs the whole codebase visible. Just-in-time search-and-fetch can work, but for global queries, full visibility is sometimes simpler and produces sharper output.

**One-shot heavy-context generation.** "Write a 5,000-word literature review based on these 30 papers" benefits from the model seeing all 30 papers at once. Chunking would lose cross-paper synthesis.

The rule: **long context is appropriate when the task genuinely requires global reasoning over a coherent body of content, and a single high-quality output beats a multi-call workflow.**

The disqualifiers: agentic loops with growing context (just-in-time wins), tasks where you only need a fraction of the content (retrieve the fraction), tasks that decompose cleanly (decompose).

The pricing matters too. Sonnet 4.6 and Opus 4.7 support 1M-token context at standard rates — no surcharge. That's the technical option. Use it when the task earns it; don't default to it just because it's there.

---

## Section 11 — Putting It Together

The context engineering discipline, applied to a system you're building:

**1. Audit current context usage.** Token-count by component for a representative trace. Identify the proportions and the largest blocks.

**2. Tighten the system prompt to the right altitude.** Prune brittle conditionals; replace with heuristics. Test on real failures; remove anything that wasn't load-bearing in recent issues.

**3. Curate the tool set per agent.** Per-agent allow-lists. Consolidate overlapping tools. Cap definitions in count (10-15 max for most agents) and length (use Module 4's discipline — clear, not bloated).

**4. Examples: 3-5, diverse, canonical.** Drop anything that just papers over an edge case. Move edge cases into prompt heuristics.

**5. Default to just-in-time retrieval.** Pre-load only what's universally relevant. Give the agent good navigation tools. Use lightweight identifiers (paths, IDs, summaries) over inline content.

**6. Structure retrieved content.** Source-tag everything. Preserve boundaries between chunks. Tag dates and provenance.

**7. Plan for trajectory length.** For agents that run more than 10-20 turns, design the compaction strategy now (Module 8). Don't wait for the context window to break.

**8. Diagnose context problems explicitly.** When the model behaves badly, check whether it's a context issue (stale tool results, rotted attention, buried instruction) before assuming it's a model or prompt issue.

The principle that runs through all of this: **the context is a designed artifact, not a bag**. You decide what's in it. The model's quality is a function of that decision as much as it is of the model itself.

---

## Recap: Module 7 in eight bullets

::: bullet-points
- Context is a finite resource with diminishing marginal returns. Every token spent depletes the model's attention budget, n² pairwise relationships in the transformer notwithstanding.
- Context rot — degraded recall as token count grows — emerges across every model. Long context isn't free; it's expensive in attention even when free in tokens.
- The system prompt should live at the "right altitude" — specific enough to guide, flexible enough to handle cases not enumerated. Brittle conditional scripts and vague handwaves are both failure modes.
- Tools are part of context. Per-agent allow-lists, ~10-15 tools max, consolidate overlap, watch for tool-result bloat. Module 4's tool design discipline is also context discipline.
- Few-shot examples: 3-5 diverse, canonical examples beat 15 narrow ones. Examples teach principles; edge cases that don't generalize belong in heuristics.
- Just-in-time retrieval is the default in 2026. Pre-load only what's universally relevant; give the agent navigation tools and lightweight identifiers; let it pull what it needs as it works.
- Message history grows monotonically. Stale tool results are the biggest waste; periodic compaction (Module 8) is the right default for long trajectories.
- Diagnose context problems explicitly. Buried instructions, attention thrashing, source-citation gaps, tool-overlap confusion — each has a different fix. Don't apply a model-level patch to a context-level problem.
:::

---

::: sam-arc
**Sam, after the late-Wednesday context audit.**

Sam spent Thursday morning in the synthesizer prompt. Pruned 1,800 tokens of accreted edge-case handling that was now in heuristic form. Restructured the user-message construction to put the *instruction* first and the *findings* second, with explicit source tagging. Cut the orchestrator's pre-loaded research from "everything" to "the things every brief actually uses, plus pointers to fetch the rest."

Thursday afternoon: re-ran the holding-company case from Wednesday's failure. Synthesizer received a 60K-token context instead of 180K. Brief came back complete — six sections, including the financials section that had been silently dropped before. Latency improved 40%. Quality, by the team's own evaluator rubric, improved measurably.

Friday morning: Sam ran the same audit across the rest of the deal-research workflow's prompts. Found the orchestrator was pre-loading every past brief on similar companies into context (~80K tokens). Replaced with a `list_briefs` tool. Found the brief evaluator's prompt had 11 examples that had been added one-at-a-time over six weeks. Pruned to 4 canonical ones. Found a tool that was being given every consumer's preference profile (~5K tokens) when only the active consumer's preference was relevant (~200 tokens).

Each individual fix was small. The cumulative effect across the system was a 60% reduction in average context size, a 25% latency improvement, and a measurable quality lift on long deals (the cases that had been suffering from context rot most).

Sam's arc this module: **the bottleneck moved.** The agent loop, the tools, the protocols, the model selection — all of those needed work, and got work, in earlier modules. The current bottleneck for Sam's system is what's *in* the context window. The next bottleneck (Module 8) will be what to *do* when the trajectory gets long enough that even careful upstream context engineering isn't enough.

Each module's discipline doesn't *replace* the previous ones; it adds another layer of attention. By Module 9, Sam's system will be well-shaped (M3), well-tooled (M4), well-protocoled (M5), well-priced (M6), well-fed (M7), well-pruned (M8), and well-remembered (M9). The book builds — and so does Sam.
:::

---

## What's next

Module 8 covers what to do when the context engineering principles from this module aren't enough — when trajectories get long, when tool results pile up, when the agent needs to think across a horizon longer than its context can hold.

Three techniques: compaction (summarize and reinitialize), structured note-taking (write to memory outside the context window), and a brief preview of multi-agent architectures (which we'll deep-dive in Module 13). Plus the recent tool-result-clearing feature on the Claude Developer Platform that automates the most common cleanup pattern.

After Module 8, Module 9 covers memory across runs — state that persists between separate trajectories, so a customer-success agent remembers what it learned about a customer last week. By the end of the M7-M9 arc, you'll have the complete picture: what to put in, what to take out, what to remember.

Sam's deal-research system, post-Module 7, has tight, well-curated context per call. By Module 9 it'll also be smart about what it carries between calls. Then Module 10 starts the reliability arc — hallucinations, reflection, guardrails — which is the layer above context.

For now: take the audit exercise from Section 9. Open one production trace. Count the tokens. Find the bloat. Ship the trim. Watch the quality move.
