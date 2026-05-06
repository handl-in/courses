# Module 18 — The Capstone

::: chapter-opener
<div class="module-num">MODULE 18</div>
<div class="module-title">The Capstone</div>
<div class="subtitle">Sixteen modules of disciplines, layered.<br>One system, designed deliberately, embodying all of them.</div>
<div class="pages">~50 pages · the synthesis · how the layers compose into a complete production agent system</div>
:::

::: hook
Monday morning, six months after the customer-success bot's M17 rollout. The CRO walked into Sam's office (rare; the CRO didn't usually do walk-ins).

*"I need a deal-research-and-brief system. End-to-end. Sales has been asking for it for a year. Whatever you built for customer-success — I want that level of polish for sales. Six months. Real budget. Real headcount if you need it. What do you need from me to make it happen?"*

Sam thought for a moment. *"I need three things. First, the time to do this deliberately — I want to design from scratch using everything we've learned, not bolt features onto an existing system. Second, sign-off that we'll build it the way the platform team builds things — saga discipline, observability, the full operational stack. Third, an explicit decision from you about which compromises you're willing to accept. There are real trade-offs in this kind of system. I want them surfaced, not hidden."*

The CRO nodded. *"Get me the design. We'll work through the trade-offs together."*

Sam pulled up a blank document and stared at the cursor. The customer-success bot had grown layer by layer over fifteen months — first the loop, then workflows, then tools, then memory, then guardrails, then sagas, then observability. Each layer had been added because the previous version had failed in some specific way. The system was good now, but the path to it had been reactive.

This was different. Sam had a clean slate and seventeen modules of disciplines. The question wasn't *"how do I make this work?"* The question was *"how do I design this so the disciplines compose deliberately, so the trade-offs are visible from the start, so the system is ready for production from day one rather than after months of incidents?"*

Sam wrote at the top of the document: *"Deal Research and Brief Generation System — Design Doc v0.1."* And then, below it: *"This is the capstone. Every layer from the book applies. The question is which ones earn their place here, in what configuration, with what trade-offs."*

This module is that design doc, written long.
:::

---

## What this module is

This is the synthesis. No new patterns. No new frameworks. Every discipline in this module has appeared in M2 through M17. The capstone's job is to show how they *compose* into a complete production system — how the layers interact, where trade-offs surface, how to make the right decisions about which patterns to apply at what altitude.

The system Sam is designing: a deal-research-and-brief generation tool for the sales team. AEs (account executives) request research on target companies before customer calls; the system produces a structured brief covering the company's profile, financials, leadership, recent news, competitive position, and conversation hooks specific to the AE's deal stage.

Why this case fits the capstone:

- **It's complex enough to exercise most layers.** Single-agent versus multi-agent decisions; tool design; memory; reflection; guardrails; sagas; observability. All land here.
- **It's high-stakes enough to require operational discipline.** Briefs feed real customer conversations on $50K-$500K deals. Wrong information produces wrong sales positioning. Reliability matters.
- **It's bounded enough to be tractable.** Not "build a complete sales platform." A specific feature with specific inputs, outputs, and quality requirements.
- **It's similar to but distinct from prior systems.** Sam's already-built customer-success bot, deal-research workflow, and regulatory-research system inform the design, but this is its own product with its own constraints.

The structure of this module:

- **Section 1** — system requirements: what's being built and why
- **Section 2** — the architecture decision: workflow, single-agent, or multi-agent? (M3, M13, M14)
- **Section 3** — the agent loop, hardened (M2)
- **Section 4** — tool design (M4, M5)
- **Section 5** — model selection and routing (M6)
- **Section 6** — context engineering and memory (M7, M8, M9)
- **Section 7** — reliability layer (M10, M11, M12)
- **Section 8** — operational layer (M16, M17)
- **Section 9** — putting it together: the system end to end
- **Section 10** — the readiness checklist

The module is longer than typical. It's the synthesis; longer is appropriate.

::: pullquote
Most agent failures in production are architecture failures, not model failures. The disciplines from this book aren't gates to pass; they're a vocabulary for *making the architecture decisions deliberately*. The capstone shows what that looks like when applied from the start, not retrofitted after incidents.
:::

---

## Section 1 — System Requirements

Before writing any code or sketching any diagrams, the requirements. The first discipline Sam learned in M3 was that the agent loop is the most expensive primitive — reach for it deliberately, not by default. The same applies here at the system level: clarify what the system needs to do before deciding how it does it.

### What the AEs actually want

Sam interviewed eight AEs. The pattern across the conversations:

- **A pre-call brief that takes 5 minutes to read** — not a 30-page research dump
- **Structured sections AEs can scan quickly** — company profile, financials snapshot, leadership context, recent news, competitive position, conversation hooks specific to the deal
- **Specificity to the deal stage** — a brief for a discovery call differs from a brief for a renewal negotiation
- **Confidence about what's known vs uncertain** — AEs hate confident-sounding briefs that turn out wrong on the call
- **Updated recently** — months-old data is worse than no data
- **Handoff-ready** — the brief should be skimmable by an SE joining the call cold

What AEs explicitly do not want:

- A full strategic analysis (their VP does that level)
- Auto-drafted email templates (they have their own voice)
- Generic platitudes about "the company's growth potential"
- Long research summaries they have to summarize themselves

### The constraints

Beyond AE requirements, the constraints from the business:

- **Latency:** AEs queue up briefs in batches. P50 latency target: 5 minutes per brief. P99: 15 minutes. They can do other work while waiting; they can't wait an hour.
- **Cost:** Sales operations has approved a budget of ~$8 per brief. At 50-100 briefs per day across the team, that's a manageable run-rate.
- **Quality:** Every brief should be defensible. If an AE walks into a customer call and says something based on the brief, the brief should support what was said.
- **Reliability:** AEs preparing for calls in 30 minutes can't wait for a system outage. Target: 99% successful brief generation; structured failure mode for the 1%.
- **Compliance:** No competitor pricing, no insider information, no claims that can't be sourced. Same M12 guardrails Sam built for customer-success.

### What this means for the design

The requirements pre-determine several decisions:

- **The output is structured and bounded** — not free-form research. That tells us the synthesis problem is well-bounded; the agent isn't writing a strategic essay.
- **The work is parallelizable** — different brief sections can be researched independently. M14's orchestrator-worker fits the structure.
- **High-stakes outcome justifies M14's cost** — briefs gate customer conversations. The 15× token cost (vs single-agent) is justified by the quality and parallelism gains.
- **AE expectations match what good agent systems can do** — structured outputs, confidence reports, clear sourcing. Not "write me a 30-page treatise" (which agents do badly) or "tell me what to say to close the deal" (which agents shouldn't do at all).

The discipline from M13: *the architecture decision should follow from the work's structure, not from architectural ambition.* Sam runs through the M13 framework explicitly in Section 2.

::: nodumbq
**Q: Aren't requirements gathering and constraints just standard product work? Why is this part of the capstone?**

Because most agent failures trace back to skipped requirements work. Teams build agents to "do research" or "automate support" without specifying what the output should look like, what success means, or what the business actually values. The agent is then optimized against an unclear target, and surprises everyone when production reveals the gap. Spending time on requirements isn't optional discipline — it's the prerequisite to applying any of the M2-M17 patterns correctly.
:::

---

## Section 2 — The Architecture Decision

Now the M13 framework, run explicitly:

**Q1: Is the task parallelizable?** **Yes (strong).** Each company brief is independent of the others. Within a brief, the sections (profile, financials, leadership, news, competitive position) are largely independent — researched in parallel, synthesized into the final brief.

**Q2: Does the value of the outcome justify 15× token cost?** **Yes.** Each brief feeds a sales call worth $50K-$500K in deal value. At ~$8 per brief, the cost is rounding error against the deal size. The quality gain from parallel research justifies the multiplier.

**Q3: Can you specify clear subagent boundaries?** **Yes.** Each section has a clear mandate. The "financials worker" investigates financial signals. The "leadership worker" investigates leadership context. Boundaries are well-defined, outputs are structured.

**Q4: Can your team operate the resulting system?** **Yes.** Sam's team has built and operated multi-agent systems (the regulatory-research from M14, the research-on-demand from M14's closing). The operational infrastructure exists.

**Q5: Have you tried single-agent first?** **Yes** — Sam's earlier deal-research workflow was single-agent. It worked but had a hard ceiling on parallelism; AEs preparing back-to-back calls hit the latency wall.

The framework verdict: **multi-agent earns its place.** Now M14's framework: which architecture?

**Section 6 of M14:**
- Q1 work structure: Independent subtasks with synthesis at end → orchestrator-worker
- Q2 synthesis problem: Combining structured outputs into a coherent brief → orchestrator-worker
- Q3 operational tolerance: Match team's existing capability → orchestrator-worker

Verdict: **orchestrator-worker.** Not hierarchical (no domain-deep tree of sub-investigations), not peer collaboration (no contested-approach problem), not swarm (5-7 sections isn't massive parallelism).

### The architectural sketch

```
                    ┌──────────────────────────┐
                    │   Brief Orchestrator     │
                    │  (Opus 4.7; M14 Section 2)│
                    └────────────┬─────────────┘
                                 │ plans, dispatches
              ┌──────────┬───────┼───────┬──────────┐
              │          │       │       │          │
          ┌───┴───┐  ┌──┴───┐  ┌─┴────┐  ┴───┐  ┌──┴───┐
          │ Profile│  │ Fin │  │ Lead │  │News│  │Comp. │
          │ Worker │  │Wkr  │  │ Wkr  │  │Wkr │  │ Wkr  │
          │(Sonnet)│  │(Son)│  │(Son) │  │(Son│  │(Son) │
          └───┬───┘  └──┬───┘  └─┬────┘  └─┬──┘  └──┬───┘
              │         │        │         │        │
              └─────────┴────────┼─────────┴────────┘
                                 │ structured returns
                    ┌────────────┴─────────────┐
                    │   Brief Synthesizer       │
                    │  (Opus 4.7; M15 Section 2)│
                    │  + critic pass (M11)      │
                    │  + guardrails (M12)       │
                    └────────────┬─────────────┘
                                 │ final brief
                    ┌────────────┴─────────────┐
                    │   AE delivery            │
                    └──────────────────────────┘

    Cross-cutting layers:
    - Saga orchestration (M16) wraps multi-step actions
    - Observability (M17) traces every span
    - Memory (M9) reads existing briefs, writes new ones
    - Tools via MCP (M5) for external services
```

Five workers, one orchestrator, one synthesizer with critic pass. Each agent fully M2-M12 disciplined. Cross-cutting saga and observability layers.

This is the load-bearing architecture. Everything else in this module fills it in.

::: brain
Why one orchestrator + one synthesizer rather than letting the orchestrator also synthesize?

(Two reasons. First, separation of concerns: the orchestrator's reasoning is about *what to research*; the synthesizer's reasoning is about *how to combine findings into a brief*. These are different cognitive tasks; separate prompts, separate context, better results. Second, the synthesizer + critic pass (M11) benefits from a fresh context. The orchestrator's full deliberation about decomposition would be noise to the synthesizer; starting fresh with structured worker results lets the synthesizer focus on the synthesis problem. The cost is one extra LLM call; the gain is cleaner separation and better outputs.)
:::

---

## Section 3 — The Agent Loop, Hardened (M2)

Every agent in this system runs M2's hardened loop. This is the foundation; the patterns are unchanged from Module 2 but worth restating in the capstone context to show how they apply per role.

### The orchestrator's loop

```python
@dataclass
class AgentBudget:
    max_turns: int
    max_tokens_total: int
    max_tool_calls: int
    max_wall_clock_seconds: int


ORCHESTRATOR_BUDGET = AgentBudget(
    max_turns=15,
    max_tokens_total=200_000,
    max_tool_calls=8,
    max_wall_clock_seconds=180,
)


async def run_orchestrator(
    request: BriefRequest,
    *,
    budget: AgentBudget = ORCHESTRATOR_BUDGET,
) -> OrchestrationPlan:
    """
    Orchestrator runs M2's hardened loop.
    Goal: produce a structured plan for worker dispatch.
    """
    return await run_agent_loop(
        system=ORCHESTRATOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _format_brief_request(request)}],
        tools=ORCHESTRATOR_TOOLS,  # plan tools only, no research tools
        model="claude-opus-4-7",
        budget=budget,
        on_budget_exhausted=_handle_orchestrator_budget_exhausted,
    )
```

The orchestrator's tools are minimal: plan, dispatch, read worker results. It doesn't do its own research; that's the workers' job. Tool isolation prevents the orchestrator from drifting into its own investigations.

### The worker's loop

```python
WORKER_BUDGET = AgentBudget(
    max_turns=20,
    max_tokens_total=120_000,
    max_tool_calls=15,
    max_wall_clock_seconds=120,
)


async def run_worker(
    handoff: HandoffBrief,
    context: HandoffContext,
    *,
    budget: AgentBudget = WORKER_BUDGET,
    role: str,
) -> HandoffReturn:
    """Worker runs M2's hardened loop scoped to handoff."""
    tools = WORKER_TOOLS_BY_ROLE[role]  # role-specific tool subset
    return await run_agent_loop(
        system=_get_worker_system_prompt(role),
        messages=[{"role": "user", "content": _format_handoff(handoff, context)}],
        tools=tools,
        model="claude-sonnet-4-6",
        budget=budget,
        on_budget_exhausted=_partial_handoff_return,
    )
```

The worker's `on_budget_exhausted` returns a partial `HandoffReturn` with explicit obstacles rather than failing the whole orchestration. The synthesizer (Section 9) handles partial returns gracefully.

### The synthesizer's loop

The synthesizer is single-turn (no tool use). Reads structured worker results; produces the brief. M2's loop is overkill for single-turn; a direct API call works. But the *budget discipline* still applies — bounded output tokens, explicit error handling.

### The critic's loop

After synthesis, an M11-style critic reviews the brief. Single-turn; produces structured findings (issues found, severity, suggested edits). The synthesizer reads the critic's output and revises if needed. Two-pass cycle, capped at one revision.

### What this section established

Every agent in the system has explicit budget caps. Every agent's failure mode is explicit (partial handoff for workers, plan failure for orchestrator, full retry for synthesizer if critic flags issues). No agent can exhaust system resources independently. The M2 discipline scaled across roles.

---

## Section 4 — Tool Design (M4, M5)

Each worker has a focused, role-specific tool set. The M4 discipline: tools should be cheap to learn correctly, expensive to use incorrectly, and have descriptions that read like API docs.

### Tools by role

**Profile Worker tools:**
- `lookup_company_basics` — name, industry, size, funding, public/private
- `fetch_company_website_summary` — current marketing positioning
- `search_company_news` — recent announcements, leadership changes
- `get_existing_brief` — prior briefs in the system (M9 memory)

**Financials Worker tools:**
- `lookup_financial_basics` — revenue, growth rate, public filings if available
- `lookup_funding_history` — investment rounds, investors, valuations
- `lookup_employee_count` — company size signals
- `query_news_for_financial_signals` — earnings reports, layoffs, hiring trends

**Leadership Worker tools:**
- `lookup_executive_team` — current C-suite and key VPs
- `lookup_leadership_changes` — recent hires/departures
- `search_leader_public_statements` — recent quotes, podcasts, posts (within attribution policy)
- `lookup_board_composition` — board members and affiliations

**News Worker tools:**
- `search_recent_news` — date-filtered news search
- `lookup_press_releases` — official announcements
- `search_industry_news` — sector-level events affecting the company
- `query_funding_news` — investment, M&A activity

**Competitive Position Worker tools:**
- `lookup_competitors` — direct and indirect competitors
- `compare_offerings` — product/service comparisons (without competitor pricing per M12)
- `lookup_market_position` — analyst reports if available
- `query_customer_signals` — public customer wins/losses

### Why tools, not skills

Notice these are *tools* (M4), not *skills* (the Claude Skills pattern). Tools are deterministic capabilities the agent invokes; skills are reusable workflow recipes the agent follows. For deal research, the *capabilities* (look up financials, search news) are stable; the *workflow* of investigation is the agent's reasoning. Tools fit the capability shape; skills would be overkill.

### Tool descriptions matter

Each tool has a description that reads like an API doc (M4 Section 3):

```python
@tool
async def lookup_company_basics(
    company_name: str,
    *,
    use_cache: bool = True,
) -> CompanyBasics:
    """
    Look up basic information about a company: industry, size, funding,
    public/private status, headquarters, founded year.

    Returns a CompanyBasics object with these fields:
    - name: official company name
    - industry: primary industry classification (NAICS or similar)
    - size_band: employee count band ("1-10", "11-50", "51-200", "201-1000", "1000+")
    - funding_status: "bootstrapped", "venture-backed", "public", "private-equity-backed"
    - founded_year: year founded (int or None)
    - hq_location: "City, Country" if known (str or None)
    - last_updated: date the data was last refreshed (datetime)

    Note: data may be stale if last_updated > 30 days. Check the timestamp.
    Use use_cache=False to force a fresh lookup (slower; ~2s vs ~50ms).

    Returns None if the company cannot be uniquely identified.
    """
```

The discipline: agents read tool descriptions as their learning material. Good descriptions produce correct usage; vague descriptions produce hallucinated parameters.

### MCP for cross-team tool access

Three of the tool servers (Financials, Leadership, News) are owned by other teams within the company. They expose their tools as MCP servers. The brief system invokes them via the MCP gateway pattern from M5 Section 7. This means:

- The brief system doesn't have to maintain or update those tools
- Tool descriptions stay current (the owning teams update their MCP servers)
- Authentication is centralized through the gateway
- Schema validation happens at the gateway boundary

The M5 discipline: integration is cheap, trust is expensive. Each MCP server has explicit allow/block lists for which tools the brief system can invoke; output schemas are validated; rate limits are enforced at the gateway.

::: gotcha
A common failure when reusing tools across systems via MCP: the brief system reads a tool's schema, the owning team updates the schema, the brief system breaks silently. The gateway should detect schema changes and either: (a) fail fast on schema mismatches with a clear error to the on-call team, (b) warn loudly in observability with a deprecation timeline, or (c) auto-rotate to a new schema version with a compatibility shim. Silent breakage is the worst failure mode.
:::

---

## Section 5 — Model Selection and Routing (M6)

The system uses three models, each chosen deliberately:

**Opus 4.7 — orchestrator and synthesizer.** High-stakes coordination; the orchestrator's plans gate everything downstream; the synthesizer's brief is the customer-facing output. Use the strongest model where reasoning quality matters most.

**Sonnet 4.6 — workers.** The right cost-quality balance for focused research subtasks. Each worker has a bounded scope and clear output format; Sonnet handles the work without burning Opus budget.

**Haiku 4.5 — guardrails and classifiers.** Input topic classification, output policy checks, content safety screening. Cheap, fast, sufficient for classification work that runs on every request.

### The routing table

```python
MODEL_BY_ROLE = {
    "orchestrator": "claude-opus-4-7",
    "synthesizer": "claude-opus-4-7",
    "worker.profile": "claude-sonnet-4-6",
    "worker.financials": "claude-sonnet-4-6",
    "worker.leadership": "claude-sonnet-4-6",
    "worker.news": "claude-sonnet-4-6",
    "worker.competitive": "claude-sonnet-4-6",
    "critic": "claude-opus-4-7",  # critic catches what synthesizer misses
    "guardrail.input_classifier": "claude-haiku-4-5",
    "guardrail.output_policy_check": "claude-haiku-4-5",
    "guardrail.content_safety": "claude-haiku-4-5",
}
```

### The cost calculation

Per brief, expected token usage:

```
Orchestrator: ~15K input + ~2K output = small
5 workers × 30K input + 8K output = ~190K
Synthesizer: ~50K input + 6K output = ~56K
Critic: ~60K input + 2K output = ~62K
Guardrails (input + output + each major action): ~15K total

Total: ~340K tokens per brief

Cost at current rates:
  Opus 4.7 (orchestrator + synthesizer + critic): ~80K input + ~10K output
    = $0.40 + $0.25 = $0.65
  Sonnet 4.6 (workers): ~150K input + ~40K output
    = $0.45 + $0.60 = $1.05
  Haiku 4.5 (guardrails): ~15K input + ~3K output
    = $0.015 + $0.015 = $0.03

Total per brief: ~$1.73
```

The team budgeted $8 per brief; actual cost projects to ~$2 with the multi-model routing. The savings from using Sonnet for workers and Haiku for guardrails (vs Opus everywhere) is substantial — about 4× cheaper than running everything on Opus.

### When the routing breaks

What if a worker hits a case Sonnet can't handle? The fallback: the worker reports `confidence < 0.5` in its handoff return. The orchestrator detects low-confidence returns and either re-runs that worker on Opus or escalates to a human reviewer. The route is dynamic; the default is cheap; the escalation is selective.

This is M6 Section 5's "model routing as architecture" principle: the system isn't tied to one model; it routes per-step, with explicit fallback when the cheap model fails.

::: pullquote
The right model selection isn't "use the best model for everything" or "use the cheapest model for everything." It's "use the right model per role, with fallback paths when the route fails." Multi-model routing is a first-class architectural decision, not an optimization.
:::

---

## Section 6 — Context Engineering and Memory (M7, M8, M9)

### Context engineering for each role

**Orchestrator's context:**
- System prompt: ~1.5K tokens with the orchestration discipline
- User request: ~500 tokens (the brief request)
- Tool descriptions: ~2K tokens for the orchestrator-only tools
- Memory: relevant prior briefs for the same company (M9) — capped at 3 most recent
- Total: ~6-8K input tokens, well within budget

**Worker's context (per worker):**
- System prompt: ~1K tokens with worker discipline + role specialization
- Handoff brief and context: ~2K tokens (the structured task)
- Tool descriptions: ~3K tokens for role-specific tools
- Worker-specific reference materials: ~5K tokens (e.g., "what good profile sections look like")
- During execution: tool results accumulate, capped by M8 compaction at 80K
- Total: ~10K initial + tool results, M8 keeps it bounded

**Synthesizer's context:**
- System prompt: ~1K tokens
- Original brief request: ~500 tokens
- 5 worker structured returns: ~30K-50K tokens
- Brief template / format spec: ~3K tokens
- Total: ~35K-55K input tokens

**Critic's context:**
- System prompt: ~1K tokens
- Brief draft from synthesizer: ~5K tokens
- Brief quality criteria: ~2K tokens
- Original request and worker findings: ~50K tokens
- Total: ~58K input tokens

The discipline from M7: every role has a *bounded* context budget, and the budget is split among system prompt, user content, tools, references, and runtime accumulation. None of the roles approach the 200K context limit; the discipline isn't fitting in the limit, it's spending the attention budget well within it.

### Compaction (M8)

The workers are most vulnerable to context bloat because they accumulate tool results over many turns. Each worker's loop applies M8 compaction at 80K tokens — older tool results get summarized, key findings preserved.

The synthesizer doesn't need compaction (single-pass; structured input). The orchestrator doesn't need compaction (small context, few turns).

### Memory (M9)

Three memory patterns apply:

**1. Cross-run memory: prior briefs.** When a brief request comes in for a company that's been researched before, the orchestrator reads the prior brief's metadata: when was it generated, what did it cover, what's stale. The orchestrator may decide to refresh only certain sections rather than research everything from scratch.

```python
@tool
async def get_existing_briefs(
    company_name: str,
    max_results: int = 3,
) -> list[BriefMetadata]:
    """
    Returns prior briefs for this company, most recent first.
    Each BriefMetadata includes: brief_id, created_at, sections, AE who requested.
    """
```

**2. Cross-run memory: AE preferences.** Each AE has implicit preferences (focus areas, level of detail, tone). The system records preferences over time; future briefs apply them.

```python
@tool
async def get_ae_preferences(ae_id: str) -> AEPreferences:
    """
    Returns what we've learned about this AE's preferences:
    - Preferred sections (some AEs care more about leadership, others about news)
    - Preferred detail level
    - Tone preferences
    """
```

**3. Within-run memory: shared state across workers.** This was tempting but Sam resisted it. The workers don't need to share state during execution; each has its own bounded scope. The synthesizer reads everyone's outputs at the end. M15's structured handoffs handle the integration without shared mutable state.

The discipline from M9: memory earns its place when it improves outputs without unbounded growth. Both used patterns (prior briefs, AE preferences) have natural caps and decay. The unused pattern (worker shared state) was rejected because the structured-handoff alternative is simpler.

::: brain
The orchestrator decides to refresh only the financials section of an existing brief because the rest is recent (less than 14 days old). What happens in the saga (Section 8) when the financials worker fails partway through?

(The saga's compensation only applies to the financials worker's effects. The other sections are unchanged from the prior brief. The synthesizer regenerates the brief with the prior sections + an explicit note that financials couldn't be refreshed. The system delivers a partial-update brief rather than a complete failure. This is the saga's value at the system level: failures are scoped to what actually failed, not magnified into full-system failures.)
:::

---

## Section 7 — The Reliability Layer (M10, M11, M12)

This is where most production agent systems either succeed or fail. The layers from M10-M12 transform "the agent works on happy paths" into "the agent works on production traffic, including the unhappy paths."

### Hallucination prevention (M10)

Every worker's structured return includes `references_used`. Every claim in a worker's `findings` traces to a citation. The synthesizer can't include a claim in the brief that isn't backed by at least one worker's reference.

```python
class WorkerFindings(BaseModel):
    findings: dict  # the structured findings
    references_used: list[Citation]  # provenance for every claim
    confidence: float
    nuances: list[str]


class Citation(BaseModel):
    source: str  # URL or system reference
    accessed_at: datetime
    relevant_excerpt: str  # the specific text supporting the claim
```

The synthesizer's prompt explicitly forbids unattributed claims:

```
Every factual claim in the brief MUST trace to at least one citation in the
worker findings. If you cannot find a citation supporting a claim, do NOT
include the claim. Better to have a shorter brief with verified content than
a longer brief with hallucinated specifics.
```

The critic (M11) verifies this. Any claim without a corresponding citation gets flagged. Synthesizer regenerates with the flagged claims removed or grounded.

### Reflection (M11)

The critic pass is a simple peer collaboration of one — but functionally identical to M11's pattern:

```python
async def critic_pass(brief: BriefDraft, worker_findings: list[WorkerFindings]) -> CriticReport:
    """Single-pass critic. Reviews brief against findings; flags issues."""
    response = await client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        system=CRITIC_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": _format_for_critic(brief, worker_findings),
        }],
        tools=[{
            "name": "submit_critique",
            "description": "Submit critique findings.",
            "input_schema": CriticReport.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "submit_critique"},
    )
    return CriticReport(**response.content[0].input)
```

Critic checks:
- Every brief claim is grounded in worker findings
- Worker confidence levels are reflected in brief language ("definitely" vs "likely" vs "appears to")
- No competitor pricing or proprietary information leaked through
- Brief is appropriately scoped to the AE's deal stage
- Length is within target (briefs longer than 5 minutes to read get flagged)

If the critic flags >0 issues, the synthesizer regenerates with the critic's findings as additional context. One revision cycle; if issues persist, escalate.

### Guardrails (M12)

Three guardrail layers, each from M12:

**1. Input filter on the brief request.** Confirms the request is for legitimate company research (not "research my ex" or other off-policy queries). Cheap classification on Haiku.

**2. Output filter on the brief draft.** Checks the synthesizer's output against the policy: no competitor pricing, no claims about non-public financial data beyond what's properly cited, no commitments on behalf of the company.

**3. Action screening on tool calls.** Each MCP tool call is evaluated against the worker's mandate. A leadership worker calling a financials API is flagged; the routing was wrong somewhere.

Together, the three layers catch what the model's training and the prompts don't. Each is cheap; combined, they're the difference between "produces fine briefs in dev" and "produces compliant briefs in production traffic."

### What makes the reliability layer work

The layers compose. M10 prevents hallucinations at the worker level. M11's critic catches what slipped through. M12 catches what's structurally outside policy. No single layer is sufficient; combined, they cover the threat surface.

Each layer is a small cost on each request. M10's citation discipline costs ~5% additional tokens per worker. M11's critic costs one extra Opus call (~$0.40 per brief). M12's guardrails cost ~$0.10 per brief in Haiku calls. Total reliability layer cost: ~$0.50 per brief on top of the ~$1.73 base. Still well under the $8 budget.

::: pullquote
The reliability layer is what production agent systems either build or get sued for not building. M10 grounds the claims; M11 catches the mistakes; M12 enforces the policies. Together they're the difference between "agent that produces fine output most of the time" and "agent you can defend in a customer escalation."
:::

---

## Section 8 — The Operational Layer (M16, M17)

Sagas (M16) and observability (M17) are the operational disciplines. They don't change what the agent does; they make the system survivable and diagnosable.

### Sagas at the system level

Most of the brief generation is *read-only* — looking up company info, querying news, etc. Read-only operations don't need saga discipline because there's nothing to compensate.

But two operations *do* take real-world actions:

**1. Writing the brief to memory.** When a brief is generated, it's stored for future reference (M9 cross-run memory). If the storage step fails after the brief is delivered to the AE, the system loses the memory but the AE got the brief. Recoverable via re-storing on next access.

**2. Sending the brief notification.** When the brief completes, the system notifies the AE via Slack. The notification is irreversible (you can't unsend a Slack message). Compensation = sending a correction if the brief content was later flagged invalid.

The saga shape:

```python
@workflow.defn
class BriefDeliverySaga:
    @workflow.run
    async def run(self, params: BriefDeliveryParams) -> SagaResult:
        # Step 1: Store the brief in the brief archive
        brief_id = await workflow.execute_activity(
            store_brief_activity,
            params.brief_content,
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Step 2: Update AE's brief queue
        await workflow.execute_activity(
            update_ae_queue_activity,
            params.ae_id, brief_id,
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Step 3: Send Slack notification (irreversible)
        try:
            await workflow.execute_activity(
                send_slack_notification_activity,
                params.ae_id, brief_id,
                retry_policy=RetryPolicy(maximum_attempts=5),
            )
        except Exception as e:
            # Compensate: revert queue update and brief
            # (Slack send was last; if it failed, no Slack to retract)
            await workflow.execute_activity(
                revert_ae_queue_activity, params.ae_id, brief_id,
            )
            await workflow.execute_activity(
                delete_brief_activity, brief_id,
            )
            return SagaResult(status="compensated", error=str(e))

        return SagaResult(status="success", brief_id=brief_id)
```

The saga is bounded. Three steps. Compensations on the first two. The Slack notification is irreversible and placed last, per M16 Section 5's discipline.

Most of the brief generation runs *outside* the saga — the actual agent work, the multi-agent orchestration, the synthesis. Those are the read-only operations that don't need saga semantics. The saga wraps only the cross-system writes.

### Observability layer

OpenTelemetry instrumentation throughout. Every span has the agent role, model, brief request ID, and AE ID as attributes. The trace tree:

```
brief.request (root)
├── orchestrator.plan (Opus)
├── workers (parallel)
│   ├── worker.profile (Sonnet) → tools → return
│   ├── worker.financials (Sonnet) → tools → return
│   ├── worker.leadership (Sonnet) → tools → return
│   ├── worker.news (Sonnet) → tools → return
│   └── worker.competitive (Sonnet) → tools → return
├── synthesizer.draft (Opus)
├── critic.review (Opus)
├── (optional) synthesizer.revise (Opus, if critic flagged issues)
├── guardrails.output_policy_check (Haiku)
├── saga.brief_delivery
│   ├── saga.step.store_brief
│   ├── saga.step.update_queue
│   └── saga.step.send_slack
└── brief.delivered
```

Every span is queryable. The trace tree reconstructs the full trajectory.

Metrics dimensioned by:
- `agent.role` (orchestrator, worker.X, synthesizer, critic, guardrail)
- `gen_ai.request.model` (opus-4-7, sonnet-4-6, haiku-4-5)
- `tool.name` (per tool call)
- `saga.type` (brief_delivery)
- `outcome.status` (success, partial, escalated)

Tier 1 alerts (page on-call):
- Brief request error rate > 5% over 15 minutes
- Saga escalation rate > 3 per hour
- p99 latency > 30 minutes (well past customer SLO)

Tier 2 alerts (notify on-call channel):
- Token-per-brief 50%+ above baseline
- Worker confidence distribution shift (more low-confidence returns)
- Critic flag rate above baseline (more issues detected)
- Tool error rate elevation for any specific tool

Tier 3 (dashboard for periodic review):
- Per-AE brief request patterns
- Per-company research patterns
- Memory hit rate (how often prior briefs are reused)
- Cost per brief trends

The observability layer doesn't prevent failures; it makes them visible and diagnosable. Per M17's data: the team can diagnose any escalation in under 10 minutes by pulling the relevant trace.

::: postmortem
**The First Production Incident — Two Weeks After Launch**

Sam's deal-research-and-brief system had been in production for two weeks. AEs were generating ~80 briefs per day. P50 latency was 4.2 minutes; P99 was 9.8 minutes. Things were working.

Then the financials MCP server started returning 500 errors at random for about 6% of requests. The financials worker would retry (M2 hardened loop), often succeed on retry, but occasionally exhaust retries and return a partial result with explicit obstacles ("financials API is degraded; got partial data on 3 of 5 financial signals").

Tier 2 alert fired on tool error rate. The on-call engineer pulled the dashboard. Saw the financials API errors. Pulled a trace. Confirmed the workers were degrading gracefully — partial returns, lower confidence, briefs still generated with explicit caveats in the financials section.

Engineer messaged the financials team. They were aware; recent deploy had introduced a regression; rollback in progress. Engineer didn't wake anyone up. The system absorbed the degradation; AEs got slightly degraded briefs ("Note: financial signals are limited due to data source degradation; full financials available in 24 hours") but the service didn't go down.

Total impact: 14 briefs out of ~1,100 had degraded financials sections that day. Zero incidents from the AE side. The financials team rolled back; the next morning's briefs were back to baseline quality.

**Lesson:** the disciplines from M2-M17 turned what could have been a production incident into a graceful degradation. Worker retries (M2). Partial handoff returns (M15). Critic verification that the brief still met quality bar despite incomplete sections (M11). Guardrails that didn't block on the partial data (M12). Saga that didn't escalate because the brief still delivered (M16). Observability that surfaced the issue at Tier 2 without paging (M17). Each layer's discipline contributed to the outcome.
:::

---

## Section 9 — The System End to End

Pulling together all the sections. The full request flow:

```
1. AE submits brief request via Slack slash command
2. Request enters input filter (M12 input guardrail; Haiku classifier)
3. Orchestrator (Opus) reads request + AE preferences (M9) + existing briefs (M9)
4. Orchestrator produces OrchestrationPlan with 5 worker tasks
5. Workers run in parallel (asyncio.gather, M14 orchestrator-worker)
   - Each worker (Sonnet) loads its system prompt + role tools + handoff
   - Each worker runs M2 hardened loop with budget caps
   - Tools called via MCP gateway (M5) with action screening (M12 layer 3)
   - Worker outputs include findings, citations (M10), confidence, nuances (M15)
6. Synthesizer (Opus) reads structured worker returns
   - Detects contradictions (M15 Section 6)
   - Resolves via tool-grounded lookup if possible
   - Produces brief draft with M10 grounding discipline
7. Critic (Opus) reviews draft against worker findings + criteria (M11)
   - Flags ungrounded claims, miscalibrated confidence, scope violations
   - If issues found, synthesizer regenerates (one revision cycle)
8. Output filter (M12 output guardrail; Haiku) checks brief against policy
9. Saga (M16) delivers brief: store + queue + Slack notification
10. AE receives brief in Slack with link to view in detail
11. Observability traces (M17) capture every step throughout
```

End-to-end latency budget:
- Step 1-2: <1 second
- Step 3-4 (orchestrator plan): ~10 seconds
- Step 5 (workers in parallel, slowest worker): ~120 seconds
- Step 6-7 (synthesizer + critic): ~30 seconds
- Step 8 (output filter): ~2 seconds
- Step 9 (saga): ~5 seconds
- Total: ~3 minutes median; ~5 minutes P95

End-to-end cost (re-stating from Section 5):
- ~$1.73 in inference
- ~$0.10 in MCP tool calls (paid services)
- ~$0.05 in observability infrastructure
- ~$0.02 in saga infrastructure
- ~$0.10 in Managed Agents runtime (per the M16 estimate of $0.08/hour at ~5 minutes per brief)
- Total: ~$2.00 per brief

End-to-end reliability characteristics:
- Single worker failure → partial handoff, brief generated with explicit gaps
- Multiple worker failures → orchestrator escalates rather than producing low-quality brief
- Synthesizer failure → automatic retry with simplified prompt
- Critic flagging unfixable issues → escalate to human reviewer
- Saga delivery failure → compensation runs; AE notified of generation failure with retry option
- Cascading downstream failures → observability alerts at Tier 2; on-call diagnoses in <10 min

### What gets escalated to humans

Per M12 and M16, the system escalates to humans in specific cases:
- **Compliance violations the guardrails couldn't auto-correct.** Manual review of the request.
- **Critic flags persistent issues even after revision.** Manual brief generation by the team.
- **Saga compensation failures.** Structured escalation per M16 Section 7.
- **Confidence below threshold across multiple workers.** Insufficient quality; human researcher takes over.

These cases are rare (target: <2% of requests) but the escalation path is structured. The AE gets a clear message: "Your brief request requires human review; expect delivery within 4 hours" rather than a confusing failure.

### What the AE sees

The AE submits `/brief AcmeCorp deal-stage:discovery` and receives, ~3-5 minutes later, a Slack message:

```
🎯 Brief ready for AcmeCorp (discovery call)
- 5 sections, all freshly researched
- Confidence: high (4 sections), medium (1 section)
- Generated 4m 12s ago
[View full brief →]
[Share with SE →]
```

If something degraded:

```
🎯 Brief ready for AcmeCorp (discovery call) — partial
- 4 sections complete, 1 section degraded (financials data source unavailable)
- Confidence: high (4 sections), N/A (1 section)
- Generated 5m 4s ago
- Note: financials section uses cached data from 18 days ago
[View full brief →]
[Request refresh in 4 hours →]
```

If escalated:

```
🎯 Brief request for AcmeCorp queued for human review
- Issue: critic flagged claims that couldn't be sourced from available data
- Senior researcher will deliver full brief within 4 hours
- You'll receive a follow-up Slack message when complete
```

The AE always has a clear understanding of what they got and what to expect. The system is honest about its limits.

::: code-exercise
**Exercise 18.1 — Design your capstone system.**

Pick an agent system you're considering or building. Run through the M18 framework:

1. **Section 1: Requirements.** What does the user actually want? What are the constraints?
2. **Section 2: Architecture.** Run M13 framework. If multi-agent, run M14 framework.
3. **Section 3: Loop.** Define budget per role.
4. **Section 4: Tools.** What's the focused tool set per role? Are they MCP-shareable?
5. **Section 5: Models.** Per-role model selection. Cost projection.
6. **Section 6: Context and memory.** Per-role context budget. What memory patterns apply?
7. **Section 7: Reliability.** Where does M10 grounding apply? M11 reflection? M12 guardrails?
8. **Section 8: Operations.** What sagas wrap which actions? What observability instrumentation?
9. **Section 9: End to end.** Sketch the request flow with all layers integrated.
10. **Section 10: Readiness.** Run the readiness checklist (next section).

The exercise is the synthesis. Most systems benefit from being designed deliberately rather than evolved reactively. The frameworks compose; the layers compose; the production system is the composition.
:::

---

## Section 10 — The Readiness Checklist

Before any agent system goes to production, the readiness checklist. Each item is from a prior module; failing any one is a signal the system isn't ready.

### Foundations (M2-M9)

- [ ] **M2:** Every agent has explicit `max_turns`, `max_tokens_total`, `max_tool_calls`, `max_wall_clock_seconds` budget caps
- [ ] **M3:** Architecture decision is explicit: workflow vs single-agent vs multi-agent, with documented reasoning
- [ ] **M4:** Every tool has API-doc-quality description; parameter schemas are precise; error returns are structured
- [ ] **M5:** External tools accessed via MCP gateway with auth, schema validation, rate limits
- [ ] **M6:** Model selection per role is explicit; fallback paths defined for cheap-model failures
- [ ] **M7:** Each role has bounded context budget; context spending is deliberate
- [ ] **M8:** Long-running agents have explicit compaction triggers
- [ ] **M9:** Memory patterns are explicit; what's remembered, what's not, what's the decay/cap

### Reliability (M10-M12)

- [ ] **M10:** Every factual claim in agent outputs traces to a verifiable source
- [ ] **M11:** High-stakes outputs go through critic review before delivery
- [ ] **M12:** Three-layer guardrails (input, output, action) implemented per the specific threat surface

### Multi-Agent (M13-M15) [if applicable]

- [ ] **M13:** Multi-agent earns its complexity per the framework; documented evidence
- [ ] **M14:** Specific architecture chosen per the framework; one pattern, not several
- [ ] **M15:** Structured handoff schemas with required nuance, confidence, obstacles, references_used; contradiction detection before synthesis; tool-grounded resolution where applicable

### Operations (M16-M17)

- [ ] **M16:** Multi-step actions are sagas; idempotency keys derived from saga ID; compensations defined per action; irreversible actions placed last
- [ ] **M17:** OpenTelemetry tracing covers entire request flow; metrics dimensioned by agent role and model; tier-based alerting; correlation IDs propagated; tail sampling configured

### Pre-launch verification

- [ ] **Compensation paths tested.** Force failures at each saga step; verify compensation works.
- [ ] **Guardrails tested.** Adversarial inputs; policy-violating outputs. Each layer verified.
- [ ] **Observability tested.** Pull a recent trace; verify on-call engineer can diagnose without external context.
- [ ] **Cost projections validated.** Run 100 production-like requests; measure actual cost; compare to projection.
- [ ] **Latency budgets validated.** Same. P50, P99 measured against targets.
- [ ] **Escalation paths tested.** Simulate cases that escalate; verify queue receives structured context.
- [ ] **Documentation reviewed.** Architecture, dependencies, runbooks. New on-call engineer should be able to operate the system from documentation alone.

### What "ready" means

Production-ready isn't "works on happy path." It's "the failure modes are explicit, recoverable, and diagnosable." The checklist surfaces gaps; passing all items means the system has the disciplines from the book applied deliberately.

The honest framing: most teams skip 30-50% of this checklist on first launch and discover the gaps in production. The checklist's value is *forcing the conversations* about which gaps you're accepting and why. *"We're skipping M11 reflection because the budget is too tight"* is a defensible decision; *"We didn't think about M11"* is the failure mode.

::: gotcha
The most dangerous checklist failure: assuming a layer "doesn't apply" without examining it. Teams sometimes argue *"we don't need M16 sagas because our agent doesn't take actions"* — and then discover the brief storage step is an action that needs saga discipline. Or *"we don't need M9 memory because each request is stateless"* — and then discover users want continuity across conversations. Examine each layer; document why it does or doesn't apply; don't assume.
:::

---

## Recap: Module 18 in eight bullets

::: bullet-points
- The capstone is synthesis: no new patterns, but the disciplines from M2-M17 composed into a complete production system. Most agent failures are architecture failures, not model failures. Designing deliberately from the start beats retrofitting after incidents.
- Sam's deal-research-and-brief system is the case study because it's complex enough to exercise most layers, high-stakes enough to require operational discipline, bounded enough to be tractable, and similar to but distinct from prior systems in the book.
- Architecture decisions follow from work structure, not architectural ambition. M13's framework picks multi-agent (parallelizable, high-stakes, bounded subagents). M14's framework picks orchestrator-worker (independent subtasks with synthesis at end). The simplest pattern that handles the structure is the right one.
- Each agent in the system is fully M2-M12 disciplined. Hardened loops with budget caps. Focused role-specific tools. Multi-model routing (Opus for orchestrator and synthesizer, Sonnet for workers, Haiku for guardrails). Bounded context budgets per role. Memory patterns explicit. Hallucination prevention via citations. Critic pass for reflection. Three-layer guardrails.
- The reliability layer is what production agent systems either build or get sued for not building. M10's grounding discipline + M11's critic + M12's guardrails compose into the difference between "agent that works in dev" and "agent you can defend in customer escalations." Each costs a small fraction of total compute; the combination justifies itself.
- The operational layer (M16 sagas + M17 observability) makes the system survivable and diagnosable. Sagas wrap multi-step actions with proper compensation; observability traces the full request flow with bounded sampling; tier-based alerting catches degradation before customers notice.
- The end-to-end system: ~3 minute median latency, ~$2 per brief cost, structured failure modes when things degrade, observability that supports <10 minute diagnosis. The disciplines compose; the system is the composition.
- Production readiness is a checklist of explicit decisions, not a feeling. Each layer either applies or is documented as not-applicable with reasoning. Most teams skip 30-50% on first launch and discover gaps in production. The checklist's value is forcing the conversations about which gaps you're accepting and why.
:::

---

::: sam-arc
**Sam, six months later.**

The deal-research-and-brief system had been running for four months. AEs were generating ~120 briefs per day across the team. P50 latency was holding at 3.4 minutes. Cost per brief had stabilized at $1.92. Sales had reported a measurable improvement in deal velocity for AEs who were using the briefs heavily — not because the briefs were magic, but because AEs were prepping more deals more thoroughly than before.

The CRO came back. *"What's next?"*

Sam thought about it. The platform team was building the next agent system — a customer-onboarding workflow — and was using the design doc Sam had written for this system as the template. The disciplines weren't customer-success-specific or sales-specific; they were *agent-system-specific*. Every new agent system inherited the M2-M17 stack as the starting point, not the destination.

The book had taken Sam from M1 (workflows in a trench coat, the embarrassing realization) to M17 (operational discipline as the actual deliverable). Each module had added a layer of attention to one part of the system. The customer-success bot Sam had hardened over fifteen months had accumulated the layers reactively, in response to specific incidents. The deal-research system had inherited the layers deliberately, in advance.

Both systems worked. The deal-research system worked *better*, and worked *faster to ship*, because the disciplines had been applied deliberately rather than discovered through incidents.

That was the actual lesson of the book. Not specific patterns, though those mattered. Not specific models or frameworks, though those mattered too. The lesson was: **agent systems are systems. They benefit from architectural discipline. They reward operational rigor. They punish reactivity. The disciplines that apply to one agent system apply to the next, and the next, and the next.**

Sam wrote the closing line of the design doc: *"This is the capstone. Every layer from the book applies. The next system will inherit the layers from day one. The system after that, the same. We're not building one agent system; we're building a platform for agent systems. The disciplines from this book are the platform."*

The CRO read the doc. *"What's the next agent we should build?"*

Sam smiled. *"Whatever sales needs most. Now that the platform exists, the next system takes weeks, not quarters."*

That, in the end, was what the book had built toward. Not a single agent system. Not a single discipline. The accumulated layers of attention that turn AI agent development from an art into an engineering practice — one where the patterns compose, the operations are sustainable, the failure modes are designed for, and the next system inherits the discipline of the last.

Sam closed the laptop. The book had been written, layer by layer, over many months. The system Sam was operating now embodied all of it. The next one would, too. That was the deliverable.
:::

---

## What's next

The book has covered M2 through M18. The arc has moved from foundational disciplines through multi-agent architecture into operational maturity into deliberate system design.

What this module didn't cover, on purpose:

- **Specific frameworks** — LangGraph, CrewAI, AutoGen, and their successors. Frameworks are tooling; the patterns in this book apply across all of them. Choose the framework that fits your team; apply the patterns deliberately.
- **Specific models** — beyond verifying that the right model goes to the right role, the patterns are model-agnostic. As models change (and they will), the disciplines stay the same.
- **Specific use cases** — Sam's customer-success bot, deal-research workflow, regulatory-research system, and brief-generation system are illustrations. Your use cases will differ; the disciplines apply.

The closing thought, from the production AI agents in 2026 piece that informed this module's research:

> *"Most agent failures are architecture failures, not model failures."*

The disciplines in this book are the architecture. They compose into systems that work in production. The next agent you build will be better than the last because the patterns are now muscle memory.

Build deliberately. Layer the disciplines. Make the trade-offs explicit. Ship systems that survive Tuesday.

That's the book.

---

::: bullet-points
**A reading guide, in case you ever need to come back:**

- **For a new project:** start at M1, then M2-M3 (foundations), then jump to M13 (architecture decision) to determine which multi-agent modules apply.
- **For a struggling project:** start with M11-M12 (reliability) and M16-M17 (operations). These are the layers most often missing in struggling systems.
- **For a multi-agent design:** M13 first (does it earn it?), then M14 (which architecture?), then M15 (how do they communicate?).
- **For operational issues:** M16 (failures recoverable?) and M17 (failures diagnosable?). Most operational gaps are in these two modules.
- **For tool design problems:** M4 then M5. Most tool problems are description problems.
- **For context bloat or cost issues:** M7 then M8. Most cost issues are context engineering problems.
- **For hallucination issues:** M10 first; M11 if grounding alone isn't enough.
- **For "the agent does the wrong thing" issues:** M12 (guardrails) before assuming a model problem.
- **For "the agent gets stuck or fails weirdly" issues:** M2 (loop discipline) and M17 (observability to see what happened).
- **For new agent systems built on a mature platform:** M18 readiness checklist. Apply the disciplines deliberately.
:::
