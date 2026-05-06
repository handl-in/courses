# Module 9 Outline — Memory Systems and Context Graphs

::: chapter-opener
<div class="module-num">MODULE 09 — OUTLINE</div>
<div class="module-title">Memory Systems and Context Graphs</div>
<div class="subtitle">Compression decides what to forget.<br>Memory decides what to remember — and how to find it again.</div>
<div class="pages">Target length: ~38 pages (architecture-heavy + code-heavy)</div>
:::

## What this module is

Compression (Module 8) discards information from the working window. Memory is what catches it. Without memory, an agent that compresses is just throwing facts away faster.

This module builds a layered memory system from scratch and compares it to the production frameworks engineers actually use: **Mem0** (hybrid vector + graph, open-source), **Letta / MemGPT** (tiered memory inspired by OS virtual memory), **Zep / Graphiti** (temporal knowledge graphs), **LangMem** (LangGraph-integrated, three memory types). We cover the surprising 2026 finding that a plain filesystem (CLAUDE.md style) hits 74% on memory benchmarks, beating many vector-store libraries.

By the end the reader can: implement working/semantic/episodic/procedural memory in code, decide vector vs graph vs filesystem for their workload, integrate a memory system into the Module 7-8 Context Builder, and avoid the most common failure mode (memory bloat — writing too much, retrieving too much, drowning the agent).

::: hook
"Your agent forgets everything between sessions. The user is annoyed. You add a 'memory.' Now your agent remembers everything. The user's previous angry message comes back to haunt every conversation. You added storage. You didn't add memory."
:::

---

## Section 1 — What Memory Actually Means for Agents (≈3 pages)

Memory is misnamed. It's not "the agent stores things and remembers them later." It's a system of:

- **Capture** — what gets recorded? (The opposite of compression's "what gets discarded.")
- **Storage** — where does it live? (Vector store? Graph? Plain files?)
- **Indexing** — how is it organized for retrieval?
- **Retrieval** — given the current context, what's relevant?
- **Consolidation** — when does raw episodic data become consolidated semantic knowledge?
- **Forgetting** — when does old information get pruned?

A "memory system" is all six. Most teams build the first three and discover the agent's quality degrades anyway, because retrieval and consolidation are where the real engineering lives.

::: pullquote
A memory system that stores everything and retrieves naively is worse than no memory. It pollutes context with noise that distracts the agent and costs you tokens.
:::

::: nodumbq
**Q: Isn't this just RAG with extra steps?**

RAG (retrieval-augmented generation) is one technique inside memory. RAG was designed for: take a user question, retrieve relevant documents, generate an answer. Agent memory is broader: retrieve relevant past interactions, retrieve learned facts about the user, retrieve procedural knowledge about how to handle this kind of task, plus retrieve documents. RAG is the document-retrieval slice of agent memory.

**Q: Why don't I just stuff everything into a long context window?**

Three reasons (also covered in Module 7): cost, latency, quality degradation. Plus a fourth specific to memory: across sessions, "everything" grows unbounded. There is no context window size that holds a year of conversation history times millions of users. You always need an external store; the question is just how good the retrieval is.
:::

---

## Section 2 — The Three (Plus One) Memory Types Revisited (≈4 pages)

Module 7 introduced these conceptually. Now we build them.

**Working memory** — the current context window. Already covered (Modules 7-8). Not a separate store; it's the substrate the others feed into.

**Semantic memory** — facts and preferences. Long-lived, low write rate, high read rate.

```python
class SemanticMemory:
    def __init__(self, store: VectorStore):
        self.store = store
    
    async def write_fact(self, user_id: str, fact: str, source: str, ttl: int | None = None):
        embedding = await embed(fact)
        await self.store.upsert(
            id=f"sem:{user_id}:{hash(fact)}",
            vector=embedding,
            metadata={"user_id": user_id, "fact": fact, "source": source, "ttl": ttl, "ts": now()},
        )
    
    async def retrieve(self, user_id: str, query: str, k: int = 5) -> list[Fact]:
        query_emb = await embed(query)
        results = await self.store.search(
            vector=query_emb,
            filter={"user_id": user_id},
            k=k,
        )
        return [Fact(**r.metadata) for r in results if not self._expired(r.metadata)]
```

What gets stored: "User prefers TypeScript." "User's company uses PostgreSQL 14." "User has admin access to repo X." Mostly extracted from conversations by an LLM extractor (we cover this in Section 4).

**Episodic memory** — specific past events. Higher write rate than semantic, queries often involve time and entities.

```python
class EpisodicMemory:
    def __init__(self, store: VectorStore, time_index: TimeIndex):
        self.store = store
        self.time_index = time_index
    
    async def record_episode(self, user_id: str, event: Episode):
        # event has: timestamp, summary, entities involved, outcome
        embedding = await embed(event.summary)
        await self.store.upsert(
            id=f"ep:{user_id}:{event.id}",
            vector=embedding,
            metadata=event.dict(),
        )
        await self.time_index.add(user_id, event.timestamp, event.id)
    
    async def retrieve_by_query(self, user_id, query, k=5):
        # semantic similarity
        ...
    
    async def retrieve_by_time(self, user_id, start, end, limit=20):
        # time-window query
        ...
    
    async def retrieve_by_entity(self, user_id, entity, k=5):
        # entity-mention search
        ...
```

The Letta paper-equivalent insight: episodic memory needs *three* retrieval modes — semantic (vector), temporal (time index), and entity (graph or inverted index). Naive vector-only retrieval misses the queries users actually ask ("what did I ask about last Thursday?").

**Procedural memory** — how to do things. Lowest write rate, highest stability, often hand-curated and machine-augmented.

```python
class ProceduralMemory:
    """Learned workflows and tool-use patterns."""
    
    def __init__(self, store: KeyValueStore):
        self.store = store
    
    async def write_skill(self, skill: Skill):
        # skill has: trigger pattern, instructions, examples, success rate
        await self.store.put(f"proc:{skill.name}", skill.dict())
    
    async def retrieve_skills_for(self, situation: str, k: int = 3) -> list[Skill]:
        # match situation against trigger patterns
        ...
```

Procedural memory often manifests as agent-rewritable system prompts (LangMem's pattern: agents update their own instructions based on outcomes). Or as a library of skill templates the agent can pull in (Letta's core memory blocks; the Anthropic Skills convention).

**Reflective memory** — explicit lessons. Discussed in Module 7 as emerging; covered in depth in Module 11 (reflection). Lives alongside episodic but indexed differently.

::: brain
Look at the four memory types. Which has the highest read-write ratio? Which is most expensive to query? Which would you cache aggressively?

(Procedural: written rarely, read every turn — cache it in the agent's process. Episodic: written every turn, read selectively — needs good indexing. Semantic: written occasionally, read often — vector store sweet spot. Reflective: somewhere between procedural and episodic.)
:::

---

## Section 3 — Storage Backends: Vector, Graph, Filesystem (≈5 pages)

The substrate matters. Three backends dominate.

**Vector stores.** Pinecone, Weaviate, Qdrant, ChromaDB, Postgres+pgvector, Mem0's default. Embeddings + similarity search. Sweet spot: semantic similarity queries on unstructured text.

Strengths: fast similarity at scale; zero-schema setup; works with any text.

Weaknesses: no native temporal queries; no relational reasoning; embedding drift over time; can return semantically-similar-but-wrong answers (hallucination via retrieval).

**Knowledge graphs.** Neo4j, TypeDB, Mem0's graph layer (Mem0g), Zep's Graphiti. Nodes (entities) and edges (relations). Sweet spot: multi-hop reasoning, entity-centric queries, temporal sequences.

```cypher
// "What did Alice do with the refund I gave her last quarter?"
MATCH (u:User {email: "alice@example.com"})-[:HAS_INTERACTION]->(i:Interaction)
WHERE i.timestamp >= $quarter_start
MATCH (i)-[:RELATED_TO]->(r:Refund)
RETURN i, r ORDER BY i.timestamp;
```

Strengths: relationship reasoning; precise temporal queries; structured constraints.

Weaknesses: schema design is hard; entity extraction is itself an LLM problem; updates are complex (consistency); often overkill for simple use cases.

**Filesystem.** CLAUDE.md, MEMORY.md, USER.md, agent skill folders, plain markdown files. Letta's surprising 2026 result: a plain filesystem hits 74% on standard memory benchmarks, beating many specialized libraries.

Strengths: trivial to operate; human-readable; version-controllable; works with any tool that can read files.

Weaknesses: no semantic similarity (the agent has to know to grep for the right thing); doesn't scale to massive history (millions of episodes don't fit); no built-in retrieval API.

::: pullquote
Letta benchmarked their tiered memory against a plain filesystem. The filesystem won 74% to vector-store-equivalents' lower scores. The lesson isn't "use a filesystem." It's "your fancy memory system might be losing to a flat file. Benchmark."
:::

**The hybrid pattern that wins in 2026** (per the Digital Applied 2026 client survey):

```
Vector for fast fuzzy recall  +  Episodic buffer (rolling summary)  +  Graph for entity reasoning
```

Each component handles what it's best at. The agent (or a routing layer) picks which to query based on the question type.

::: postmortem
**The Knowledge Graph That Cost Six Months and Returned the Same Answers as Vector Search**

A team building an enterprise agent assumed they'd need a knowledge graph from day one. Spent six months designing schemas, wrote entity extractors, built ingestion pipelines, picked Neo4j, hired a graph specialist.

Final benchmark: graph queries returned the right answer 78% of the time. Vector search returned the right answer 76% of the time. The 2-point difference came from a handful of multi-hop entity questions; everything else was effectively semantic similarity in disguise.

Cost: six months of engineering for marginal gain. They scrapped most of the graph and kept it for the entity-heavy 5% of queries.

**Lesson:** start with the simplest backend that could work. Graduate to graphs only when you've measured that your queries genuinely need relational reasoning. "We might need it eventually" is how teams build six-month projects that recover six points of accuracy they could have gotten for free.
:::

---

## Section 4 — Capture: What to Write, When to Write It (≈4 pages)

The first failure mode in memory systems is over-capture. Every conversation generates dozens of "facts" — most of them stale, low-signal, or specific to one moment. Storing all of them makes retrieval slower, noisier, and more confusing.

The capture pipeline:

```python
class FactExtractor:
    def __init__(self, llm):
        self.llm = llm

    async def extract(self, conversation: list[dict]) -> list[Fact]:
        prompt = EXTRACTION_PROMPT.format(conversation=format_messages(conversation))
        resp = await self.llm.complete(prompt, max_tokens=1000)
        candidates = parse_facts(resp)
        return [f for f in candidates if self._is_durable(f)]

    def _is_durable(self, fact: Fact) -> bool:
        # Heuristic / LLM judgment: is this a stable fact or a momentary state?
        # "User is angry" — momentary, don't store.
        # "User uses Python 3.12" — durable, store.
        ...
```

Good capture rules:

- **Extract claims, not interactions.** "User said X" is weak; "X is true" is strong (when the user can be trusted as a source). Store the claim, link it to the source if traceable.
- **Prefer durable over momentary.** "User is frustrated" probably shouldn't persist; "User is allergic to peanuts" should.
- **Tag with source and timestamp.** Always. Conflicts will arise; you need provenance.
- **Don't extract what's already in the prompt.** If "user lives in NYC" is in the system prompt every turn, don't also store it as a fact.
- **Detect and resolve conflicts at write time.** If new fact contradicts old, flag — don't silently overwrite (Mem0's conflict detector does this).

::: gotcha
Don't run the extractor on every turn. It's an expensive LLM call. Run it: at session end, at compression time, or every N turns. Most facts that need extracting are stable across the conversation; extracting once per session is usually enough.
:::

---

## Section 5 — Retrieval: The Hard Part (≈5 pages)

Most memory failures look like hallucinations but are retrieval misses. The agent doesn't have the right information in context — not because it isn't stored, but because retrieval didn't surface it.

Retrieval design choices:

**Query construction.** What query do you embed/search for? Options:
- The user's last message
- A rewrite of the last message (LLM-generated query)
- The accumulated conversation summary
- A combination

The right answer depends on workload. For "what is my favorite editor?" the user's question itself works fine. For "what did we discuss about that bug last week?" the question alone is too vague — you need the user's likely intent expanded.

**Top-K selection.** How many results? Too few: missed relevance. Too many: context bloat + distraction.

A practical rule: k=3-5 for tight contexts, k=10 for exploratory; always re-rank.

**Re-ranking.** Vector search retrieves by similarity. Re-ranking applies a stronger (slower) signal — often a cross-encoder LLM — to reorder the top-N. Production systems often do: retrieve top 50 → re-rank to top 5. Mem0 added re-ranking as a default in v1.0.0; Cohere Rerank, BGE-Reranker, and others are common choices.

**Filter before search.** Don't search the whole store. Filter by user_id, session_id, time window first. Massive speedup and quality improvement.

**Hybrid retrieval.** Combine vector + keyword (BM25) + graph traversal. Mem0's graph mode (Mem0g) reports 68.4% LLM Score vs 66.9% for vector-only on multi-hop questions.

```python
class HybridRetriever:
    def __init__(self, vector_store, graph_store, reranker):
        self.vector = vector_store
        self.graph = graph_store
        self.reranker = reranker
    
    async def retrieve(self, query: str, user_id: str, k: int = 5) -> list[Fact]:
        # Parallel retrieval from both stores
        vector_task = self.vector.search(query, user_id, k=20)
        graph_task = self.graph.find_related(query, user_id, k=20)
        vector_results, graph_results = await asyncio.gather(vector_task, graph_task)
        
        # Dedupe
        candidates = dedupe(vector_results + graph_results)
        
        # Re-rank
        ranked = await self.reranker.rerank(query, candidates)
        
        return ranked[:k]
```

::: code-exercise
**Exercise 9.1 — Build the layered memory system.**

Implement: SemanticMemory (vector), EpisodicMemory (vector + time index), ProceduralMemory (key-value), HybridRetriever (vector + filter + re-rank). Wire into the Context Builder from Modules 7-8.

Test on the same multi-turn conversation set used for compression benchmarks. Measure: retrieval hit rate (when relevant memories are present, are they returned?), context bloat (how many irrelevant memories make it through?).
:::

---

## Section 6 — Consolidation: From Episodes to Knowledge (≈4 pages)

The Feb 2026 paper "Episodic Memory is the Missing Piece for Long-Term LLM Agents" argues consolidation is *the* mechanism for long-term reasoning. Storing every event is hoarding; consolidating events into reusable representations is memory.

The consolidation pipeline:

```
[raw episodes accumulate over time]
        │
        ▼
[periodic consolidation job]
        │ groups related episodes
        │ extracts patterns, recurring themes, lessons
        ▼
[generates consolidated semantic facts]
        │
        ▼
[stored in semantic memory; original episodes optionally pruned]
```

When to run consolidation:
- End of each session (consolidate that session into 1-3 semantic facts)
- Daily / weekly batch (cross-session consolidation)
- On retrieval, if too many similar episodes are returned (consolidate them on the fly)

```python
class Consolidator:
    def __init__(self, llm, semantic: SemanticMemory, episodic: EpisodicMemory):
        self.llm = llm
        self.semantic = semantic
        self.episodic = episodic

    async def consolidate_session(self, user_id: str, session_id: str):
        episodes = await self.episodic.retrieve_by_session(user_id, session_id)
        if not episodes:
            return
        
        prompt = CONSOLIDATION_PROMPT.format(
            episodes=format_episodes(episodes),
        )
        consolidated = await self.llm.complete(prompt, max_tokens=500)
        facts = parse_consolidated_facts(consolidated)
        
        for fact in facts:
            await self.semantic.write_fact(
                user_id=user_id,
                fact=fact.statement,
                source=f"consolidated:session:{session_id}",
            )
```

The consolidation prompt is where the work happens. Bad consolidation: "User did X, then Y, then Z" (just a summary). Good consolidation: "User prefers approach X for problem class Y; tried Z and abandoned it" (a reusable inference).

::: brain
Consolidation generates semantic facts from episodic events. But semantic facts can become wrong over time — preferences change, situations evolve. How do you handle stale consolidated knowledge?

(Source-tag every consolidated fact with the episodes it came from. When new contradicting episodes arrive, the conflict detector (Mem0 pattern) flags them. The consolidation job either overwrites the fact or adds a "as of date X" version.)
:::

---

## Section 7 — Frameworks: Mem0, Letta, Zep, LangMem (≈4 pages)

A current map of the production frameworks. (Verify all details when drafting; this changes monthly.)

**Mem0.** Open-source (~48K GitHub stars), best for chatbot/personal-assistant memory. v1.0.0 made async writes default and added procedural memory as first-class. Hybrid vector + graph (Mem0g). Conflict detection. Hosted SaaS or self-hosted.

```python
from mem0 import Memory

m = Memory()
m.add("User prefers TypeScript", user_id="alice")
results = m.search("what languages does the user know?", user_id="alice")
```

Strengths: easy to start; good defaults; good docs.
Weaknesses: opinionated structure; harder to deeply customize.

**Letta (formerly MemGPT).** Tiered memory inspired by OS virtual memory: core memory (always in context), archival memory (vector store), recall memory (full conversation history, on-demand retrieval). Agents actively manage their own memory via tools.

Strengths: very explicit memory model; agents have first-class memory awareness; strong for long-running agents.
Weaknesses: more complex setup; the agent has to learn to use memory tools effectively.

**Zep / Graphiti.** Production-grade temporal knowledge graphs. Best for entity-heavy enterprise workloads (CRM, healthcare, legal). Combines vector search with graph traversal and temporal reasoning.

Strengths: rich relational queries; temporal awareness baked in.
Weaknesses: schema/ontology investment required; overkill for simple chat.

**LangMem (LangChain).** SDK launched early 2025. Three memory types (episodic, semantic, procedural) with first-class procedural memory (agents updating their own system instructions). Requires LangGraph's StateGraph.

Strengths: integrates with LangChain ecosystem; strong procedural memory story.
Weaknesses: tied to LangGraph; not a standalone library.

**The simple option: filesystem + small custom code.**

```
/memory/
  /alice/
    USER.md           # core facts about the user
    SKILLS.md         # learned workflows (procedural)
    /episodes/
      2026-04-15.md   # daily episode summaries
      2026-04-16.md
```

Letta's benchmark: this beats many vector-store libraries. CLAUDE.md + grep is a real production pattern.

::: pullquote
Pick the simplest backend that could plausibly work. Most memory deployments fail not from picking the wrong framework, but from over-investing in the framework before they understood the workload.
:::

---

## Section 8 — Forgetting (≈3 pages)

The least-discussed memory operation. Memory that grows without bound becomes useless. You need explicit forgetting policies.

**TTL-based forgetting.** Every fact has a time-to-live. Stale facts expire automatically.

**Conflict-based pruning.** When new information contradicts old, the old gets demoted (or deleted, with a record).

**Importance-based decay.** Facts that aren't retrieved for N retrievals get demoted. Eventually deleted.

**Capacity-based eviction.** Fixed-size store; LRU-evict when full.

**Explicit deletion.** Users can request "forget that I said X" (GDPR right-to-be-forgotten; just good product design too).

```python
class ForgettingPolicy:
    async def expire_stale(self, store: SemanticMemory, max_age_days: int = 90):
        cutoff = now() - timedelta(days=max_age_days)
        await store.delete_where(ts__lt=cutoff)
    
    async def decay_unused(self, store, min_retrievals: int = 1, window_days: int = 30):
        cutoff = now() - timedelta(days=window_days)
        await store.delete_where(retrievals_in_window__lt=min_retrievals)
    
    async def resolve_conflicts(self, store, new_fact: Fact):
        existing = await store.find_contradicting(new_fact)
        for old in existing:
            if old.confidence < new_fact.confidence:
                await store.delete(old.id)
            else:
                # Keep both, mark new one as disputed
                ...
```

::: gotcha
Don't aggressively forget without an audit log. Memory that disappears without explanation is debugging hell. Every forgetting decision should be traceable: what was deleted, when, why, by what policy.
:::

---

## Section 9 — Wiring It All Together (≈3 pages)

The complete Module 7-8-9 stack:

```python
class Agent:  # Module 2 hardened agent, now with full context engineering
    def __init__(self, client, tools, system, model, budget,
                 memory: MemorySystem, compressor: Compressor, builder: ContextBuilder):
        ...

    async def run(self, query: str, user_id: str) -> AgentResult:
        task_id = uuid()
        working_memory = []
        
        # Initial turn
        working_memory.append({"role": "user", "content": query})
        
        while not self.budget.exceeded():
            # Module 7-8-9: build context
            messages = await self.builder.build(
                system=self.system,
                tools=self.tools.schemas(),
                working_memory=working_memory,
                user_id=user_id,
                task_id=task_id,
            )
            
            # Module 2: call model
            resp = await self.client.messages.create(
                model=self.model, messages=messages, tools=self.tools.schemas(),
            )
            
            # Update working memory; dispatch tools; etc. (Module 2 logic)
            ...
        
        # Session end: consolidate
        await self.memory.consolidate_session(user_id, task_id)
        
        return result
```

We run a long-horizon task end-to-end, watching:
- Working memory grow → compress → stay manageable
- Facts extracted into semantic memory
- Episodes recorded into episodic memory
- Retrieval surfacing relevant past memories on new queries
- Consolidation at session end producing reusable knowledge

::: code-exercise
**Exercise 9.2 — Run a multi-session benchmark.**

Run the agent through 5 sessions with the same user, separated by simulated days. Each session involves a different but related task. Measure:

- Cross-session continuity: does the agent remember user-stated preferences from session 1 in session 5?
- Memory hygiene: how many memories are stored after 5 sessions? Are they all useful?
- Retrieval quality: in session 5, when relevant memories from session 2 should surface, do they?

Compare with-memory vs without-memory agent on a held-out evaluation set.
:::

---

## Section 10 — What's Next (≈2 pages)

Module 10 covers **hallucinations** — including failure modes specific to retrieval (the agent confidently states a fact that came from a low-relevance retrieval; the agent invents a detail that "sounds like" what was retrieved). Memory is one of the biggest hallucination sources, and the defenses live in Module 10.

Module 11 covers **reflection** — explicitly using memory to learn from past failures. Reflective memory is a sibling of episodic memory, indexed by "lessons learned" rather than "what happened."

Module 12 covers **guardrails** — including memory-aware guardrails ("never reveal facts about other users").

By Module 14 (multi-agent), memory becomes a coordination problem: shared memory across agents vs isolated per-agent memory.

The capstone (Module 18) ships an agent with full layered memory.

---

## Recurring elements used in this module

- 1 chapter opener
- 1 hook
- 2 No Dumb Questions
- 2 Brain Power prompts
- 1 Production Postmortem (six-month knowledge graph for 2-point gain)
- 2 Watch It! / Gotcha blocks
- 2 Code Exercises (build memory, multi-session benchmark)
- 2 Pullquotes
- 1 architecture diagram
- 1 framework comparison table
- 1 Bullet Points recap

::: sam-arc
Sam ships the compressed agent from Module 8. The PM tries it across multiple sessions: "It works great until I come back tomorrow and it doesn't remember anything we discussed." Sam adds memory. First attempt: store every message. Retrieval surfaces irrelevant junk; quality drops; Sam reverts. Second attempt: extract only durable facts (Section 4), hybrid retrieval with re-ranking (Section 5), session-end consolidation (Section 6). By the end of Section 9, the agent has continuity across sessions, doesn't bloat with junk, and selectively recalls. Sam's arc this module: **storage isn't memory; engineering retrieval is**.
:::

::: page-budget
S1 (What memory means): 3p
S2 (Memory types): 4p
S3 (Backends): 5p
S4 (Capture): 4p
S5 (Retrieval): 5p
S6 (Consolidation): 4p
S7 (Frameworks): 4p
S8 (Forgetting): 3p
S9 (Wiring): 3p
S10 (Next): 2p
TOTAL: ~37 pages
:::

::: sources
**Must verify when drafting:**

- "Episodic Memory is the Missing Piece for Long-Term LLM Agents" (arXiv:2502.06975, Feb 2026) — primary source for consolidation argument
- "Memory in the Age of AI Agents" survey (arXiv:2512.13564, Dec 2025) — current taxonomy
- Mem0 v1.0.0 release notes — async default, procedural memory, conflict detection
- Mem0g (graph variant) benchmark — 68.4% vs 66.9% vector-only
- Letta documentation — three-tier (core/archival/recall), benchmark showing filesystem at 74%
- Zep / Graphiti — temporal knowledge graph specifics
- LangMem SDK — three memory types, agent-self-rewriting procedural memory
- Digital Applied 2026 client survey (vector + episodic + graph hybrid pattern)
- Cohere Rerank v3 / BGE-Reranker — current state of re-ranking models
- Anthropic Skills convention — for procedural memory format
- Tulving 1972 (the original memory taxonomy) — academic citation
- Hermes OS / Hermes Agent dual-layer 2026 architecture
- pgvector / Pinecone / Qdrant / Weaviate — current API and pricing for vector store comparison

**Stable knowledge:**
- Capture / store / index / retrieve / consolidate / forget pipeline
- Vector vs graph vs filesystem trade-offs
- Hybrid retrieval as the production pattern
- The forgetting policies (TTL, decay, eviction, deletion)

**Cross-references:**
- Module 7 introduced memory types; this module builds them
- Module 8 (compression) discards into memory; this is what catches it
- Module 10 covers retrieval-induced hallucination
- Module 11 (reflection) extends procedural and reflective memory
- Module 14 covers shared vs isolated memory in multi-agent
:::

::: bullet-points
### Module 9 in eight bullets

(filled at draft time)

- Memory is six operations: capture, store, index, retrieve, consolidate, forget — most teams build three
- Working / semantic / episodic / procedural — same taxonomy as Module 7, now with implementations
- Three storage backends: vector (similarity), graph (relations), filesystem (simple) — hybrid wins in production
- Capture is over-eager by default; extract durable claims, not interactions, with provenance
- Retrieval is the hard part: query construction, re-ranking, filtering, hybrid sources
- Consolidation turns raw episodes into reusable semantic knowledge — the missing piece for long-term agents
- Frameworks: Mem0 (defaults), Letta (tiered), Zep (graphs), LangMem (LangChain), or filesystem (surprisingly competitive)
- Forgetting needs explicit policies: TTL, decay, eviction, conflict resolution — and audit logs
:::
