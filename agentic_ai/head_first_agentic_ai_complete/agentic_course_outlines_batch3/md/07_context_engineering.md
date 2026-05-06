# Module 7 Outline — Context Engineering Foundations

::: chapter-opener
<div class="module-num">MODULE 07 — OUTLINE</div>
<div class="module-title">Context Engineering Foundations</div>
<div class="subtitle">Prompt engineering was the warm-up.<br>This is the discipline that decides whether your agent ships.</div>
<div class="pages">Target length: ~36 pages</div>
:::

## What this module is

Context engineering is the discipline that emerged in 2025 as engineers realized: building agents that work for one turn is prompt engineering; building agents that work over hundreds of turns and dozens of tool results is something else entirely. The framework that won — write, select, compress, isolate — comes from LangChain (Lance Martin, June 2025) and is now used at IBM, MongoDB, Letta, Mem0, Anthropic, and across the field.

This module establishes the framework, the four memory types every agent eventually needs (episodic, semantic, procedural, working), and the design principles that make the next two modules (compression, memory systems) possible. Heavy on concepts and decision frameworks; the algorithms and code-heavy build come in Module 8 and Module 9.

::: hook
"Your agent has a 200K context window. Sounds enormous. Then turn 12 dumps a 30K log file. Turn 14 fetches three 8K web pages. Turn 19 retrieves 12 documents from your knowledge base. Turn 22 the model is reasoning over 180K tokens of mostly garbage — paying $0.54 per turn — and missing the one fact that mattered, which is buried on line 43 of turn 7's tool result."
:::

---

## Section 1 — From Prompt Engineering to Context Engineering (≈3 pages)

The shift. Prompt engineering optimized one input (a prompt) for one output (a completion). It implicitly assumed the prompt is what the model sees.

In agents, that assumption breaks. Each turn, the model sees:

- The system prompt (rarely changes)
- Tool definitions (Module 4)
- The user's original message
- Every previous turn's assistant message
- Every previous turn's tool calls and results
- Whatever your agent code injected (retrieved memories, current state, etc.)

The "prompt" — singular — has become "the context window" — a dynamic, growing, multi-source assemblage that you, the engineer, *compose* every turn. That composition is context engineering.

::: pullquote
Prompt engineering is what to write. Context engineering is what to keep, what to fetch, what to summarize, what to throw away — every turn, forever.
:::

What changed in 2024-2025 to make this a discipline:

1. **Long agent horizons.** Agents running for 50+ turns blew past naive context management.
2. **Tool result bloat.** Agents that fetch web pages, read files, query databases produce huge intermediate context.
3. **Cost reality.** A 200K-context Sonnet call costs ~$0.60. At dozens of calls per task, this is real money.
4. **Quality degradation.** Long contexts measurably degrade reasoning ("context rot" — well-documented). More context isn't better.

::: nodumbq
**Q: Doesn't Claude Sonnet 4.6 have a 1M context window now? GPT-5.4 too? Doesn't that solve this?**

No, for three reasons. (1) Cost: a 1M token call at $3/MTok input is $3 per call. (2) Latency: TTFT scales with input length. A 500K-token call has noticeably worse first-token latency. (3) Quality: long-context degradation is real and measurable. Even models marketed as "1M coherent" perform meaningfully better on focused 50K context than dumped 500K context. Bigger windows raise the ceiling on what's *possible*; they don't change what's *optimal*.

**Q: Is context engineering just a fancy name for managing the conversation history?**

It includes that. It also includes: deciding what to fetch from external memory, when to compress old turns, when to spawn a sub-agent with isolated context, when to write a fact to long-term memory, when to discard a tool result entirely. Conversation history is one input among many.
:::

---

## Section 2 — The Write/Select/Compress/Isolate Framework (≈5 pages)

LangChain's framework, which became the field standard. Four verbs that describe everything you can do with context:

**Write** — persist information *outside* the context window so it's available later but doesn't consume tokens now. Examples: scratchpads, long-term memory stores, notes-to-self files, episode logs.

**Select** — pull information *into* the context window only when needed. Examples: retrieval (RAG-style), tool calls that fetch state, conditional prompt sections.

**Compress** — reduce the size of what's already in context while preserving meaning. Examples: summarization of old turns, dropping verbose tool outputs, replacing raw data with computed conclusions.

**Isolate** — split context across separate processes or sub-agents so each only sees what it needs. Examples: orchestrator-workers (Module 3), sub-agent delegation, multi-agent architectures (Module 14).

Most production systems combine all four. A research agent might:
- **Write** intermediate findings to a scratchpad after each search
- **Select** the most relevant 3-5 findings when synthesizing
- **Compress** old search results that didn't pan out
- **Isolate** sub-research-tasks to separate sub-agents

::: brain
Your agent's context just hit 150K tokens. You can spend the next hour applying any one strategy. Which do you reach for first?

(Probably **isolate** — spinning off a sub-agent for the next subtask gives you a fresh window. Then **compress** if isolation isn't an option. **Select** assumes you have a retrieval system in place. **Write** doesn't help with the immediate problem; it helps with the next problem.)
:::

We do a deep example: a customer support agent that answers questions across a 200-document knowledge base. We show the naive version (dump everything in context) and walk through applying each of the four strategies in turn, watching context size and quality both improve.

::: postmortem
**The Chatbot That Got Dumber Every Day**

A team built a Slack bot that helped engineers debug production issues. Naive design: every conversation in a channel was fed to the bot as context. Days 1-3 worked great. By day 14, context was 80K tokens of mostly stale conversation. By day 30, the bot was confidently mixing details from issues weeks apart, citing wrong customers, and giving advice based on solved problems.

Cause: zero context engineering. No write (no separate fact memory), no select (every message included regardless of relevance), no compress (full transcript), no isolate (one bot, one growing context).

Fix involved all four: write extracted facts to a memory store, select only conversation from the past 4 hours plus retrieved facts, compress with a 24-hour rolling summary, isolate per-thread instead of per-channel.

**Lesson:** the four strategies aren't optional. They're the minimum scaffolding for any agent that runs for more than a session.
:::

---

## Section 3 — The Three (or Four) Memory Types (≈5 pages)

The memory taxonomy borrowed from cognitive science (Tulving 1972) and adapted by Letta, Mem0, LangMem, and others. The three memory types every production agent eventually distinguishes:

**Working memory** — what's in the current context window right now. Cleared between sessions or compressed continuously. Analog: human short-term memory, RAM.

**Episodic memory** — specific past events. "On 2026-04-15, the user asked about refund policy and I gave them this answer." Used for: continuity, personalization, learning from past interactions. Analog: human autobiographical memory.

**Semantic memory** — facts and preferences. "User works at Acme Corp." "User prefers TypeScript over JavaScript." "Company policy is no refunds after 30 days." Used for: stable knowledge that should persist. Analog: human factual knowledge.

**Procedural memory** — *how* to do things. Not facts, but learned workflows. "When asked to debug Python imports, first check `__init__.py` files, then PYTHONPATH, then virtual env." Used for: skill accumulation. Analog: human motor/skill memory. Mem0 added this as a first-class type in v1.0.0; Letta uses core memory blocks for similar effect.

A fourth type that's emerging:

**Reflective memory** — explicit lessons learned from successes and failures, separate from episodic. "Last time I tried this approach to refund handling, the customer escalated. Try X first." This shows up under various names (reflection memory, retrospective memory, experience memory) and is closely tied to Module 11.

::: brain
Map each to write/select/compress/isolate from Section 2:
- Working memory ↔ ?
- Episodic ↔ ?
- Semantic ↔ ?
- Procedural ↔ ?

(Working = whatever's currently in context, the substrate compress and isolate operate on. Episodic, semantic, procedural all live in *write*-stores and become *select* candidates. Different memory types are storage; the four strategies are operations.)
:::

::: nodumbq
**Q: Do I need all three memory types from day one?**

No. Most production agents start with semantic memory only ("remember user preferences") and add episodic later when they need continuity ("what did we discuss yesterday?"). Procedural memory matters most for agents that should learn workflows over time — coding agents, research agents, ops agents.

**Q: How is episodic memory different from just keeping conversation history?**

Conversation history is raw. Episodic memory is *consolidated* — extracted, summarized, indexed. The Feb 2026 paper "Episodic Memory is the Missing Piece for Long-Term LLM Agents" (arXiv:2502.06975) argues that consolidation — converting events into compact, reusable representations — is the key mechanism. Storing every message verbatim is hoarding; episodic memory is filing.
:::

---

## Section 4 — Where Context Comes From: Source Mapping (≈4 pages)

Before you engineer context, know what's flowing into it. We map every source:

**Static (rarely changes):**
- System prompt
- Tool definitions
- Behavioral guidance / persona

**Per-task:**
- User's original query
- Initial state passed in (user ID, session ID, etc.)
- Pre-loaded reference material (documents the user attached)

**Dynamic (changes every turn):**
- Previous assistant messages
- Tool call inputs
- Tool call results — often the largest source
- Retrieved memories (from semantic/episodic stores)
- Sub-agent return values

**Injected:**
- Time-sensitive context (current date, user timezone)
- Permission/role context
- Recent activity context (last login, recent purchases)

We build a context-source diagram for a representative agent and label each source with: typical token cost, frequency of change, criticality to the task. Then we go through and ask of each: write, select, compress, or isolate?

```
┌─────────── CONTEXT WINDOW (this turn) ──────────────┐
│                                                       │
│  System prompt                          400 tok       │
│  Tool definitions (12 tools)          1,800 tok       │
│  ─────                                                │
│  User query                              50 tok       │
│  ─────                                                │
│  Turn 1 assistant + tool call           120 tok       │
│  Turn 1 tool result (web_fetch)       4,200 tok ← compress?
│  Turn 2 assistant + tool call            90 tok       │
│  Turn 2 tool result (db query)        2,300 tok       │
│  Turn 3 assistant + tool call            80 tok       │
│  Turn 3 tool result (web_fetch)      18,500 tok ← compress!
│  ...                                                  │
│  Retrieved memories (5 facts)           400 tok       │
│  ─────                                                │
│  TOTAL                              ~30,000 tok       │
└──────────────────────────────────────────────────────┘
```

::: gotcha
Tool results are almost always the dominant source after a few turns. They're also where most teams ignore context engineering — every other source gets attention; tool results just accumulate. The compression module (Module 8) targets this directly.
:::

---

## Section 5 — Design Principles (≈4 pages)

Six principles to apply when designing an agent's context:

**1. Treat context as a resource, not a free input.** Every token has cost (real $) and benefit (relevance to task). Optimize the ratio.

**2. Write before you need to.** Persist facts to memory the moment they appear, not when context fills up. Late writes lose information.

**3. Select on demand, not on speculation.** Don't pre-fetch "everything that might be relevant." Let the agent fetch when it knows it needs it. Speculative loading is how naive RAG bloats context.

**4. Compress lossy, keep originals lossless.** Compressed context is what the agent sees. Original tool results, raw data, full conversations stay in your logs / memory store / scratchpad. You can always re-fetch; you can't undelete.

**5. Isolate when ownership boundaries change.** A sub-task that needs different tools, different persona, or runs in parallel — that's a candidate for isolation. Same task, same tools, same context — keep it in one agent.

**6. Measure before optimizing.** Most teams do context engineering by gut feel. The teams that ship instrument first: token count per source per turn, retrieval hit rates, post-compression task success.

::: brain
"Compress lossy, keep originals lossless." Why is this principle non-obvious enough to call out?

(Because most engineers' first instinct is to throw away the raw tool result once it's been summarized. Then debugging an agent that got the wrong answer becomes impossible — you can't see what the model saw before the summary. Keep originals in logs/storage; only show summaries in context.)
:::

::: pullquote
Context isn't what you have. Context is what you choose, every turn, to put in front of the model.
:::

---

## Section 6 — A Reference Architecture (≈5 pages)

We design the context-engineered agent we'll extend in Modules 8-9. The architecture has six layers:

```
            ┌──────────────────────────────────────┐
            │       AGENT LOOP (Module 2)          │
            └──────────┬───────────────────────────┘
                       │ assembles each turn's context
                       ▼
            ┌──────────────────────────────────────┐
            │      CONTEXT BUILDER (this module)   │
            │  - merges static + dynamic sources   │
            │  - applies retrieval                 │
            │  - applies compression               │
            │  - injects memory                    │
            └──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬─────────────┐
        ▼              ▼              ▼             ▼
   ┌─────────┐   ┌──────────┐   ┌──────────┐  ┌──────────┐
   │WORKING  │   │ EPISODIC │   │ SEMANTIC │  │PROCEDURAL│
   │MEM (RAM)│   │  STORE   │   │  STORE   │  │  STORE   │
   └─────────┘   └──────────┘   └──────────┘  └──────────┘
                       │
                       ▼
            ┌──────────────────────────────────────┐
            │   COMPRESSION ENGINE (Module 8)      │
            └──────────────────────────────────────┘
```

**The Context Builder** is the new component. Every turn it:

1. Pulls static context (system prompt, tools)
2. Pulls dynamic context (working memory / message history)
3. Decides what to retrieve (select from memory stores)
4. Decides what to compress (compression engine, Module 8)
5. Decides what to write (extract facts/episodes, send to stores)
6. Returns the assembled `messages` list to the agent loop

```python
class ContextBuilder:
    def __init__(
        self,
        memory: MemorySystem,
        compressor: Compressor,
        budget: ContextBudget,  # token budget for this turn
    ):
        self.memory = memory
        self.compressor = compressor
        self.budget = budget

    async def build(
        self,
        system: str,
        tool_schemas: list[dict],
        working_memory: list[dict],
        user_id: str,
        task_id: str,
    ) -> list[dict]:
        # Always-include
        context = []
        
        # Step 1: select relevant memories
        retrieved = await self.memory.retrieve(
            query=self._derive_query(working_memory),
            user_id=user_id,
            task_id=task_id,
            max_tokens=self.budget.memory_budget,
        )
        if retrieved:
            context.append({"role": "user", "content": self._format_memories(retrieved)})

        # Step 2: compress old working memory if needed
        compressed_working = await self.compressor.compress(
            working_memory,
            target_tokens=self.budget.working_budget,
        )
        context.extend(compressed_working)

        # Step 3: extract anything worth writing to long-term memory
        facts = await self.memory.extract_writeable(working_memory)
        if facts:
            await self.memory.write(facts, user_id=user_id, task_id=task_id)
        
        return context
```

We'll fill in `MemorySystem` (Module 9) and `Compressor` (Module 8). For now: the agent loop calls `ContextBuilder.build()` instead of building messages itself.

::: code-exercise
**Exercise 7.1 — Wire the Context Builder.**

Take the hardened agent from Module 2. Replace the inline `messages` construction with calls to a `ContextBuilder`. For now, the builder is a passthrough — no compression, no memory. The point is to introduce the seam. Modules 8 and 9 fill it in.
:::

---

## Section 7 — Anti-Patterns (≈3 pages)

Common ways context engineering goes wrong.

**Speculative dumping.** "Let me include everything that might be relevant." Reality: you've increased cost, reduced quality, and now have to debug why the model is paying attention to the wrong thing.

**Over-summarization.** Compressing every tool result to one sentence. The model loses the specifics it needs to answer follow-ups.

**Memory-as-trash-can.** Writing everything to long-term memory "just in case." Then retrieval surfaces irrelevant junk. Be selective at write time, not just at read time.

**Isolation as silver bullet.** Spawning sub-agents whenever context grows. Each sub-agent has its own startup cost, its own context, and you've now multiplied your token spend. Isolate when it makes the work simpler, not when context feels uncomfortable.

**No instrumentation.** "We do context engineering." But you can't show me a graph of token cost per source per turn. You're not engineering; you're hoping.

**Format inconsistency.** Memories formatted one way, tool results formatted another, system prompts a third. The model wastes effort orienting. Pick a format (markdown sections with clear headers usually wins), use it everywhere.

::: postmortem
**The Sub-Agent Cascade That 10x'd the Bill**

A team noticed their main agent's context was getting big. They added "isolate as much as possible" as a heuristic — every subtask spawned a sub-agent. Sub-agents spawned sub-sub-agents. By the time a single user query resolved, 17 agents had run. Bill quintupled. Latency tripled. Quality unchanged because the sub-agents weren't doing meaningfully isolated work — they were just smaller copies of the same agent fighting over the same context.

Fix: rule that isolation must justify itself by either (a) parallelism (Module 3 sectioning) or (b) genuinely different tools/persona for the sub-task. Most "isolation" reverted to compression instead.

**Lesson:** isolation isn't free. It's a structural change with a structural cost.
:::

---

## Section 8 — What's Next (≈1 page)

Module 8 builds the **Compression Engine**: anchored iterative summarization, ACON's failure-driven optimization, budget-aware compression, distilled compressors. Code for all four. Benchmarks on real workloads.

Module 9 builds the **Memory System**: vector store for semantic, knowledge graphs for entity-heavy queries, episodic store with consolidation, procedural memory as agent-rewritable instructions. Mem0/Letta/Zep architectural comparison. Layered memory pattern.

After Modules 8-9, your agent has full context engineering. By Module 10 it also handles hallucinations; by Module 12 it has guardrails for off-topic drift.

---

## Recurring elements used in this module

- 1 chapter opener
- 1 hook
- 2 No Dumb Questions (one in Section 1, one in Section 3)
- 3 Brain Power prompts
- 2 Production Postmortems (Slack bot drift, sub-agent cascade)
- 1 Watch It! / Gotcha block
- 1 Code Exercise (wire the Context Builder)
- 2 Pullquotes
- 1 architecture diagram
- 1 context-source breakdown diagram
- 1 Bullet Points recap

::: sam-arc
Sam's agent works on simple queries. PM tries it on the company's actual support workload — multi-turn debugging sessions that span hours. The agent loses track of what was discussed three turns ago, retrieves wrong docs, and racks up $0.40 per turn after turn 10. Sam realizes prompt-tuning won't fix this. Sam builds a Context Builder by Section 6 and refactors the agent. Performance improves but real wins wait for Modules 8-9. Sam's arc this module: **the bottleneck moved**.
:::

::: page-budget
S1 (From prompt to context eng): 3p
S2 (Write/Select/Compress/Isolate): 5p
S3 (Memory types): 5p
S4 (Source mapping): 4p
S5 (Design principles): 4p
S6 (Reference architecture): 5p
S7 (Anti-patterns): 3p
S8 (Next): 1p
Recurring elements: 6p
TOTAL: ~36 pages
:::

::: sources
**Must verify when drafting:**

- LangChain / Lance Martin "Context Engineering for Agents" (June 2025) — primary source for write/select/compress/isolate framework
- "Episodic Memory is the Missing Piece for Long-Term LLM Agents" (arXiv:2502.06975, Feb 2026) — for the consolidation argument
- "Memory in the Age of AI Agents" survey (arXiv:2512.13564, Dec 2025) — current taxonomy and field state
- Mem0 v1.0.0 procedural memory release notes — for procedural-memory-as-first-class
- Letta documentation on tiered memory (core / archival / recall)
- Zep / Graphiti documentation for graph-based memory
- LangMem SDK release notes (early 2025)
- Anthropic / Claude context length pricing (verify $3/MTok for Sonnet 4.6 input, current model strings)
- "Context Rot" research — find the citation; this matters for the long-context-doesn't-solve-it argument
- IBM Zurich "Cognitive Tools Framework" — referenced; verify

**Stable knowledge:**
- Tulving 1972 memory taxonomy (semantic, episodic, procedural)
- The general principle that context composition matters
- Anti-patterns (speculative dumping, etc. — these are universal)

**Cross-references:**
- Module 8 implements the compression engine
- Module 9 implements the memory system
- Module 11 (reflection) ties to reflective/retrospective memory
- Module 14 (multi-agent) is where isolation becomes architectural
:::

::: bullet-points
### Module 7 in eight bullets

(filled at draft time)

- Context engineering is the discipline of composing what the model sees every turn — across many sources
- Write/select/compress/isolate is the canonical four-strategy framework
- Working, episodic, semantic, procedural — the four memory types every production agent eventually distinguishes
- Tool results dominate context after a few turns; they're where most teams ignore engineering
- Six design principles, including: compress lossy, keep originals lossless; measure before optimizing
- The Context Builder is a real component sitting between the agent loop and memory/compression
- Anti-patterns: speculative dumping, over-summarization, memory-as-trash-can, isolation cascade
- Bigger context windows raise the ceiling but don't change what's optimal — engineering still matters
:::
