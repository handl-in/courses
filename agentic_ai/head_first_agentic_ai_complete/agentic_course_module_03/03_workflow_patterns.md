# Module 3 — Workflow Patterns You Should Reach for First

::: chapter-opener
<div class="module-num">MODULE 3</div>
<div class="module-title">Workflow Patterns You Should Reach for First</div>
<div class="subtitle">Five patterns. Code for each. The same task three ways, with the bills attached.<br>This is what you'll build most weeks, even when you're "building agents."</div>
<div class="pages">~42 pages · the patterns that earn their keep</div>
:::

::: hook
Sam's deal-research project — the one their CEO finally green-lit at the end of Module 1 — kicked off on Monday with a planning meeting. Two engineers, one product manager, a designer, and Sam in a conference room with a whiteboard.

The PM had questions. *"How many agents are we building? Three? Five? The Slack thread mentioned seven."*

Sam drew a single box on the whiteboard and labeled it `agent`. Then a different box labeled `workflow`. Then drew a line between them.

*"Most of this is the workflow side. The actual agent — autonomous loop, model decides what to do — is one piece. The pieces around it are workflows. Sequence the data gathering. Route by deal type. Run the company-research, market-research, and competitor-research in parallel. Evaluate the draft brief and revise. The agent loop is in the middle of all that, doing the open-ended work. Everything else is structured."*

The PM stared at the diagram for a moment. *"So we're building... mostly not an agent."*

*"We're building one agent surrounded by workflows. That's what most production agent systems are. The vocabulary makes it confusing."*

Sam circled the workflow box. *"Module 3 of the book I'm reading — these patterns. Five of them. Most of what we're going to build is one of those five. We'll come back to the agent box on Wednesday."*
:::

---

## What this module is

Module 1 reframed the field — most "agents" are workflows in trench coats, and that's fine. Module 2 hardened the agent loop for the cases where you really do need an agent. This module is about the patterns you'll actually compose for everything else, and the cases inside agent systems where workflow shapes show up as components.

The five canonical patterns from Anthropic's December 2024 "Building Effective Agents" essay — prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer — have held up for eighteen months of production use. The field has converged on them as the vocabulary. Every major framework (LangGraph, CrewAI, OpenAI Agents SDK, the Vercel AI SDK, even Azure Logic Apps) uses these same five names with minor renaming.

This module:

- Builds each pattern in code, with full implementations against the Anthropic SDK
- Shows the same realistic task implemented in three different patterns to make the trade-offs concrete (cost, latency, quality)
- Calls out the pitfalls that show up in production for each pattern
- Ends with composition: how the patterns combine, and what Sam's deal-research system looks like in workflow vocabulary

By the end you'll be able to reach for the right pattern for a problem the way you'd reach for the right collection type — instinctively, with the trade-offs internalized.

---

## Section 1 — Why Workflows First

A short justification before the patterns.

The hardened agent loop from Module 2 is the right tool when you don't know what the agent will do next. That's a real and important class of problems. It's also a *minority* of the problems you'll be asked to solve.

Most "AI features" decompose into known steps. Generate this, then summarize that. Classify the input, then route to the right handler. Process these ten documents, then aggregate the findings. Draft a response, then check it for compliance. None of these need an autonomous loop. All of them benefit enormously from the predictability, debuggability, and cost-determinism of a workflow.

A workflow is, in one sentence: **structured orchestration of LLM calls along a code path you wrote.** You decide what runs, when, in what order. The model produces text or structured output at each step; your code routes it to the next step. There's no `while` loop where the model decides to keep going.

This sounds restrictive. In practice it's liberating. A workflow:

- Has a fixed maximum cost per invocation (sum the per-step max_tokens times the rates)
- Has a fixed structure that traces cleanly (each step is one span)
- Has clear failure points (this prompt failed, this gate failed, this aggregator failed)
- Can be tested step-by-step without orchestrating the whole thing
- Scales horizontally without coordination overhead

Compare to an agent loop's cost surface (variable, hard to bound below the budget cap), trace surface (variable depth, branching), failure surface (any step can loop, any tool can hang), and test surface (you have to mock the model's autonomous decisions). The agent loop wins on flexibility and loses on operability.

Reach for the workflow first.

::: pullquote
A workflow is structured orchestration of LLM calls along a code path you wrote. The model produces; you route. That's the whole pattern, and it's most of what production needs.
:::

::: nodumbq
**Q: Are workflows just hand-rolled chains of API calls? Why is this even a "pattern"?**

The patterns matter because they each have known properties — when they work, when they fail, what they cost. When you say "this is a routing workflow," the listener immediately knows: there's a classifier upfront, downstream specialists, classifier accuracy is the key risk, fallback behavior matters. The vocabulary compresses a lot of design context. Without it, every system is bespoke; with it, you can have productive design conversations in minutes instead of hours.

**Q: Are these patterns just for cases where I'm not using an agent at all?**

No — and this is important. The patterns show up *inside* agent systems all the time. Multi-agent supervisors are an orchestrator-workers workflow with agent-shaped workers (Module 14). Reflection inside an agent is an evaluator-optimizer workflow inside the loop (Module 11). Topic guardrails on an agent's input are a routing workflow that decides whether to invoke the agent at all (Module 12). Knowing the patterns is foundational even when the final system is an agent.
:::

---

## Section 2 — Pattern 1: Prompt Chaining

The simplest and most common pattern. The output of one LLM call is the input to the next. Optional gates between steps verify intermediate state.

```
[Input] → [LLM Call 1] → [Gate?] → [LLM Call 2] → [Gate?] → [LLM Call 3] → [Output]
```

**When to use:** the task decomposes cleanly into ordered steps. Each step is easier in isolation than the whole task is in one prompt. Trading some latency for better accuracy is acceptable.

**When NOT to use:** the steps don't actually depend on each other (use parallelization), the output of step 1 doesn't really shape step 2 (you might just need a single better prompt), or the chain is a workaround for an under-specified prompt (fix the prompt instead).

### A worked example: research brief generation

Sam's deal-research project will eventually produce a brief on a target company. Even with rich context, asking a single model call to "write a 2000-word brief covering financials, leadership, market position, recent news, and risks" produces uneven output. Some sections are thorough; others are sparse. The model rushes to fit it all in one response.

Decomposed into a chain:

```
[Company name] →
  [Step 1: Research and structure findings as bullet points] →
  [Gate: Are all required sections populated?] →
  [Step 2: Write the brief from the structured findings] →
  [Step 3: Polish prose and fix transitions] →
  [Final brief]
```

Each step does one thing and does it well. The gate after step 1 catches the case where the research came back sparse — easier to retry the cheap research step than to discover the gap in the polished output.

### Implementation

```python
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

client = AsyncAnthropic()


class CompanyFindings(BaseModel):
    """Structured output of step 1; consumed by step 2."""
    financials: list[str] = Field(description="Revenue, growth, margins, runway")
    leadership: list[str] = Field(description="Key people and recent changes")
    market_position: list[str] = Field(description="Where they sit in the market")
    recent_news: list[str] = Field(description="Material events in last 90 days")
    risks: list[str] = Field(description="Concerns a buyer should know")


async def chain_step_1_research(company: str) -> CompanyFindings:
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=(
            "You research companies for a deal team. Produce structured findings, "
            "not prose. Each list item should be a single fact with a source if available."
        ),
        messages=[{"role": "user", "content": (
            f"Research {company}. Populate every category. If you don't have "
            f"information for a category, return an empty list — do not invent items."
        )}],
        # Use the SDK's tool-call mechanism to get structured output
        tools=[{
            "name": "submit_findings",
            "description": "Submit structured findings about the company.",
            "input_schema": CompanyFindings.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "submit_findings"},
    )
    findings_block = next(b for b in response.content if b.type == "tool_use")
    return CompanyFindings(**findings_block.input)


def gate_findings_complete(f: CompanyFindings) -> tuple[bool, str]:
    """Programmatic check: did step 1 produce something usable?"""
    required = [f.financials, f.leadership, f.market_position]
    if not all(len(section) >= 2 for section in required):
        return False, (
            f"Sparse findings: financials={len(f.financials)}, "
            f"leadership={len(f.leadership)}, market_position={len(f.market_position)}"
        )
    return True, ""


async def chain_step_2_draft(company: str, findings: CompanyFindings) -> str:
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        system="You write deal-team briefs. Be concise. Use the findings provided; do not invent.",
        messages=[{"role": "user", "content": (
            f"Write a 1500-word brief on {company} using these findings:\n\n"
            f"{findings.model_dump_json(indent=2)}\n\n"
            "Structure: Executive Summary, Financial Profile, Leadership, "
            "Market Position, Recent Developments, Key Risks."
        )}],
    )
    return response.content[0].text


async def chain_step_3_polish(draft: str) -> str:
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3500,
        system="You are an editor. Polish prose, fix transitions, tighten language.",
        messages=[{"role": "user", "content": (
            f"Polish this brief. Keep all facts; improve flow and clarity.\n\n{draft}"
        )}],
    )
    return response.content[0].text


async def research_brief_chain(company: str) -> str:
    # Step 1: Research
    findings = await chain_step_1_research(company)

    # Gate
    ok, reason = gate_findings_complete(findings)
    if not ok:
        # Decision: retry research, or fail with diagnostic. Here, retry once.
        findings = await chain_step_1_research(company)
        ok, reason = gate_findings_complete(findings)
        if not ok:
            raise ChainGateFailure(f"Research insufficient after retry: {reason}")

    # Step 2: Draft
    draft = await chain_step_2_draft(company, findings)

    # Step 3: Polish
    polished = await chain_step_3_polish(draft)
    return polished


class ChainGateFailure(Exception):
    pass
```

### What this pattern buys you

**Clearer prompts, each one focused.** The research prompt has nothing to say about prose style. The polishing prompt has nothing to say about facts. Each is shorter, less ambiguous, and easier to debug than a single prompt would be.

**Validatable intermediate state.** The gate after step 1 is a programmatic check — Python code, not another LLM call. Cheap, deterministic, catches "the research came back sparse" before you spend tokens drafting on top of nothing.

**Cheaper retries on failure.** If polishing produces something awkward, you can re-run polish without redoing research. If research is sparse, you can re-run research without throwing away a draft.

**Predictable cost.** Three calls at known max_tokens. The chain's worst-case cost is the sum of the three. Compare to an agent loop where the worst case is the budget cap.

### Pitfalls

**Failure cascading without gates.** A chain without programmatic checks between steps will happily polish nonsense into nice-sounding nonsense. The gate is what makes chaining safer than a single big prompt — without it, errors compound through the chain. Always have at least one structural gate.

**Over-decomposition.** Some teams chain everything into 8-step pipelines because three steps "felt like too few." Each step adds latency, cost, and a potential failure point. Decompose to the smallest number of steps that each do one job well, then stop.

**Implicit context loss between steps.** The polish step doesn't see the original findings; only the draft. If the draft loses a fact, polish can't recover it. For chains where steps need different views of the same source data, pass the source explicitly to every step that needs it.

::: gotcha
The most common chain bug: step 2's prompt assumes information that step 1 sometimes doesn't produce. Step 2 then hallucinates. The gate after step 1 must check for the presence of the information step 2 will assume — not just "did step 1 produce something." Be specific in your gates.
:::

---

## Section 3 — Pattern 2: Routing

A classifier (or LLM, or both) examines the input and dispatches it to a specialized handler. Used for separation of concerns and for cost optimization.

```
[Input] → [Router] → ┬─→ [Specialist A]
                    ├─→ [Specialist B]
                    └─→ [Specialist C]
```

**When to use:** distinct categories of input genuinely benefit from different handling. Different prompts, different models, different toolsets, different downstream pipelines. The categories are stable enough that classification can be reliable.

**When NOT to use:** the categories blend (most inputs touch multiple categories), the specialist prompts would be 90% identical (just write one good prompt), or the router itself is the hard problem (you're solving the wrong layer).

### A worked example: support ticket router

A SaaS company gets thousands of support tickets a day. The right model and prompt for "my password reset email isn't arriving" is very different from "your API is returning 503s on the /v2/orders endpoint." The first wants Haiku 4.5 with a friendly customer-success prompt; the second wants Sonnet 4.6 with API documentation in context.

A routing workflow decides which specialist runs. Cost optimization is real: 70% of tickets can be handled by Haiku at $1/$5 per million tokens; 25% need Sonnet at $3/$15; the remaining 5% need Opus 4.7 with extended thinking.

### Implementation

```python
from enum import Enum


class TicketCategory(str, Enum):
    ACCOUNT = "account"  # password resets, login, profile
    BILLING = "billing"  # subscription, invoices, payments
    TECHNICAL = "technical"  # API, integrations, errors
    GENERAL = "general"  # everything else


class RouteDecision(BaseModel):
    category: TicketCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


async def route_ticket(ticket: str) -> RouteDecision:
    """Use Haiku as the router. It's fast, cheap, and accurate enough for classification."""
    response = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        system=(
            "You categorize support tickets. Pick the most specific category. "
            "Return your confidence (0.0-1.0) and a one-sentence reason."
        ),
        messages=[{"role": "user", "content": ticket}],
        tools=[{
            "name": "submit_routing",
            "description": "Submit the routing decision.",
            "input_schema": RouteDecision.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "submit_routing"},
    )
    decision_block = next(b for b in response.content if b.type == "tool_use")
    return RouteDecision(**decision_block.input)


# Specialist handlers — different prompts, different models per the routing target.
SPECIALIST_CONFIG = {
    TicketCategory.ACCOUNT: {
        "model": "claude-haiku-4-5",
        "system": (
            "You handle account questions: password reset, login, profile updates, "
            "email changes. Be concise and friendly. If you can't fully answer, "
            "tell the user what info you'd need to escalate."
        ),
    },
    TicketCategory.BILLING: {
        "model": "claude-sonnet-4-6",
        "system": (
            "You handle billing questions: subscription tiers, invoices, payment "
            "issues, refunds. Reference the user's plan when known. Never promise "
            "refunds outside policy: pro-rated for cancellation, none for downgrades."
        ),
    },
    TicketCategory.TECHNICAL: {
        "model": "claude-sonnet-4-6",
        "system": (
            "You handle technical questions: API errors, integrations, SDK issues. "
            "Always ask for: error code, request payload (redacted), timestamp. "
            "Reference our docs at docs.example.com for non-trivial issues."
        ),
    },
    TicketCategory.GENERAL: {
        "model": "claude-sonnet-4-6",
        "system": (
            "You handle general inquiries that don't fit other categories. "
            "Be helpful; if the question is out of scope, redirect politely."
        ),
    },
}


async def handle_specialist(category: TicketCategory, ticket: str) -> str:
    config = SPECIALIST_CONFIG[category]
    response = await client.messages.create(
        model=config["model"],
        max_tokens=1000,
        system=config["system"],
        messages=[{"role": "user", "content": ticket}],
    )
    return response.content[0].text


CONFIDENCE_FALLBACK_THRESHOLD = 0.6


async def route_and_handle(ticket: str) -> tuple[str, RouteDecision]:
    decision = await route_ticket(ticket)

    # Low confidence? Fall back to GENERAL specialist (which is broader).
    # Or escalate to human; depends on your business.
    if decision.confidence < CONFIDENCE_FALLBACK_THRESHOLD:
        decision = RouteDecision(
            category=TicketCategory.GENERAL,
            confidence=decision.confidence,
            reasoning=f"Low confidence on initial routing ({decision.reasoning}); using general handler.",
        )

    response = await handle_specialist(decision.category, ticket)
    return response, decision
```

### What this pattern buys you

**Cost optimization with quality preservation.** You're not running every ticket through Sonnet just because *some* tickets need it. The router (Haiku, fast and cheap) decides which specialist runs. The aggregate cost on a realistic mix of tickets drops substantially without sacrificing quality on the ones that need a stronger model.

**Specialist prompts can be focused.** The billing specialist has detailed billing context loaded; the account specialist has account context. Each is shorter and more reliable than a god-prompt that knows everything.

**Per-category tooling.** Different specialists can have different tool access. The technical specialist might have access to a docs-search tool; the billing specialist might have access to a subscription-lookup tool; the account specialist might have neither.

**Failure isolation.** If the billing specialist's prompt is broken, only billing tickets misbehave. Account and technical tickets are unaffected.

### Pitfalls

**Router becomes the bottleneck.** If the router is wrong 10% of the time, your specialists handle 10% of tickets they're not optimized for. Routing accuracy is the most important metric to track. Use a confidence threshold and route low-confidence to a generalist or to a human queue.

**Categories drift.** New ticket types appear that don't fit existing categories. The router stuffs them into the closest fit, badly. Periodically sample low-confidence routings (say, weekly) and see if a new category is needed.

**Two-class classifiers in disguise.** Sometimes "we have four categories" really means "we have one important category and three rare ones." If 95% of traffic is one category, you might just want a single specialist plus an escalation path; the routing infrastructure is overkill.

**Confidence inflation.** LLM-based routers report confidence based on what looks-like-confidence in their training data, not calibrated probability. Don't trust the absolute number; calibrate against your own labeled data.

::: brain
Your router classifies billing tickets correctly 96% of the time. The 4% that misroute go to the technical specialist, which gives wrong answers about payment plans. Where do you fix this?

(Three options. (1) Improve the router — better prompt, more examples, fine-tune a small classifier. (2) Add a confidence floor that routes uncertain billing-or-technical tickets to a human queue. (3) Make the technical specialist's prompt include "if this is actually a billing question, redirect to billing rather than answering." Option 3 is often the cheapest fix and most robust — it makes specialists self-aware of mis-routing. Real production systems usually combine all three.)
:::

::: nodumbq
**Q: Should the router be an LLM or a small classifier (BERT-class)?**

For most production systems with reasonable volume, an LLM router (Haiku-tier) is the right call. The marginal cost is small, the accuracy is usually high enough out of the box, and you don't have to maintain a training pipeline. A small dedicated classifier wins on extreme-volume cases where Haiku-level cost-per-call adds up, or where the categories are very stable and you have lots of labeled data. Most teams over-engineer this; start with the LLM router and migrate only if you outgrow it.
:::

---

## Section 4 — Pattern 3: Parallelization

A task is run by multiple LLM calls simultaneously, with their outputs aggregated. Two flavors that share the structure but differ in intent:

- **Sectioning:** the task is split into independent subtasks that run in parallel and combine into a whole.
- **Voting:** the same task runs multiple times with different prompts or configurations, and the outputs are aggregated for confidence.

```
                  ┌─→ [LLM Call A] ─┐
[Input] ──────────┼─→ [LLM Call B] ─┼─→ [Aggregator] → [Output]
                  └─→ [LLM Call C] ─┘
```

**When to use sectioning:** subtasks are genuinely independent. Reviewing code with separate prompts for security, performance, and style. Translating a long document chapter-by-chapter. Running content moderation in parallel with response generation so safety doesn't block latency.

**When to use voting:** confidence matters more than speed. The task is borderline or ambiguous. You want to catch outlier outputs. Multiple votes from a single model, with different temperatures or prompts, can detect the cases where the model is uncertain.

**When NOT to use:** the subtasks aren't actually independent (chaining is right). The task isn't borderline (one good prompt is right). You're using parallelization for "more is better" without an aggregation strategy that benefits from parallelism.

### A worked example (sectioning): document review

A legal team reviews contracts. Each contract gets reviewed by:

- A clauses-specialist (looks for unusual terms)
- A jurisdiction-specialist (checks governing law and venue)
- A risk-specialist (flags liabilities and indemnification issues)

Sequential review of a long contract is slow. Parallel review with three specialists, each focused, finishes in roughly the time of the slowest review. The aggregator combines findings into a single report.

```python
import asyncio


class ReviewFinding(BaseModel):
    severity: str = Field(description="info | warn | error")
    location: str = Field(description="section/clause reference")
    issue: str
    suggestion: str | None = None


class SpecialistReport(BaseModel):
    specialist: str
    findings: list[ReviewFinding]


SPECIALIST_PROMPTS = {
    "clauses": (
        "You are a contract clauses specialist. Identify unusual terms, "
        "non-standard language, and clauses that deviate from market norms. "
        "Cite the specific section. Severity: info (notable), warn (worth discussion), "
        "error (deal-breaker without redline)."
    ),
    "jurisdiction": (
        "You are a jurisdiction specialist. Check governing law, venue, "
        "dispute resolution mechanism, and any cross-border issues. "
        "Cite the specific section. Severity: info | warn | error."
    ),
    "risk": (
        "You are a risk specialist. Identify liability caps, indemnification scope, "
        "warranties, and termination triggers. Flag asymmetric risk allocation. "
        "Cite the specific section. Severity: info | warn | error."
    ),
}


async def specialist_review(specialist: str, contract: str) -> SpecialistReport:
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SPECIALIST_PROMPTS[specialist],
        messages=[{"role": "user", "content": (
            f"Review this contract and report findings.\n\n{contract}"
        )}],
        tools=[{
            "name": "submit_findings",
            "description": "Submit your specialist findings.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": ReviewFinding.model_json_schema(),
                    },
                },
                "required": ["findings"],
            },
        }],
        tool_choice={"type": "tool", "name": "submit_findings"},
    )
    block = next(b for b in response.content if b.type == "tool_use")
    return SpecialistReport(
        specialist=specialist,
        findings=[ReviewFinding(**f) for f in block.input["findings"]],
    )


async def parallel_contract_review(contract: str) -> dict:
    # Fan out: all three specialists run concurrently.
    reports = await asyncio.gather(*[
        specialist_review(s, contract) for s in SPECIALIST_PROMPTS
    ], return_exceptions=True)

    # Handle individual failures without losing the others.
    successes: list[SpecialistReport] = []
    failures = []
    for s, r in zip(SPECIALIST_PROMPTS, reports):
        if isinstance(r, Exception):
            failures.append((s, str(r)))
        else:
            successes.append(r)

    # Aggregate
    all_findings = [
        {**f.model_dump(), "specialist": rep.specialist}
        for rep in successes
        for f in rep.findings
    ]
    by_severity = {
        "error": [f for f in all_findings if f["severity"] == "error"],
        "warn":  [f for f in all_findings if f["severity"] == "warn"],
        "info":  [f for f in all_findings if f["severity"] == "info"],
    }
    return {
        "by_severity": by_severity,
        "specialists_succeeded": [r.specialist for r in successes],
        "specialists_failed": failures,
        "total_findings": len(all_findings),
    }
```

Three things worth noticing:

**`return_exceptions=True` on `asyncio.gather`.** If the jurisdiction specialist times out, you still get the clauses and risk reports. The aggregate output flags which specialists succeeded, so the user (or the next step) knows the review is partial.

**Aggregation by severity.** Cross-specialist organization helps the reader. A single contract might get 3 errors, 7 warnings, 12 info — grouped by severity is more useful than three separate reports stacked vertically.

**No retry of failed specialists in this version.** If the jurisdiction specialist fails, this code reports the partial result and moves on. Whether to retry is a business call; don't bake retry into the pattern, expose the failure and let the caller decide.

### A worked example (voting): high-stakes classification

A safety classifier needs to flag potentially harmful content. False negatives are bad (harmful content reaches users); false positives are bad (legitimate content gets blocked). Single-shot classification is too brittle.

The voting variant: run the classification five times with slightly different prompts, take majority vote, surface disagreements for human review.

```python
async def vote_classify(content: str, n: int = 5) -> dict:
    """
    Run the same classification N times. Aggregate by majority.
    If votes split, escalate to human review.
    """
    base_prompts = [
        "Classify whether this content violates community guidelines.",
        "Does this content contain prohibited material? Be strict.",
        "Is this safe content for general audiences? Be lenient on edge cases.",
        "Flag this content if you'd be uncomfortable with it on a public platform.",
        "Classify as safe or unsafe based on a reasonable-person standard.",
    ][:n]

    async def one_vote(prompt: str) -> bool:
        response = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            system=prompt + " Respond with just 'safe' or 'unsafe'.",
            messages=[{"role": "user", "content": content}],
        )
        text = response.content[0].text.lower().strip()
        return "unsafe" in text  # True = unsafe

    votes = await asyncio.gather(*[one_vote(p) for p in base_prompts])
    unsafe_count = sum(votes)
    safe_count = n - unsafe_count

    if unsafe_count >= n - 1:
        return {"verdict": "unsafe", "confidence": "high",
                "votes": (unsafe_count, safe_count)}
    if safe_count >= n - 1:
        return {"verdict": "safe", "confidence": "high",
                "votes": (unsafe_count, safe_count)}
    return {"verdict": "uncertain", "confidence": "low", "votes": (unsafe_count, safe_count),
            "action": "human_review"}
```

The voting pattern's value isn't catching every bad case — it's catching *uncertainty*. When five voters disagree, that's a signal. The system surfaces it for human review rather than picking one verdict and shipping. For high-stakes classification, escalating uncertainty is the whole game.

### What this pattern buys you

**Wall-clock speedup.** Three sequential calls of 8 seconds each is 24 seconds. Three parallel calls is ~8 seconds (the slowest). For latency-sensitive flows, parallelization is the cheapest improvement.

**Specialist focus.** Each parallel call has a tight prompt. No multi-aspect god-prompt that tries to do everything.

**Confidence signal (voting).** When votes agree, you have higher confidence. When they disagree, you have a signal to escalate.

**Resilience to single failures.** With `return_exceptions=True`, one failed branch doesn't kill the whole task.

### Pitfalls

**Cost multiplies linearly.** Three parallel specialists is three times the cost of one. The latency win is real; the cost is too. Make sure the work justifies the multiplier.

**Aggregator complexity.** Combining outputs from independent specialists isn't always trivial. If the specialists produce overlapping findings, you need de-duplication. If they produce conflicting findings, you need a resolution strategy. Sometimes the aggregator becomes the hardest part of the system.

**Voting with no diversity.** If you run the same prompt five times with the same model and same temperature, you get five very similar answers. The "voting" gives false confidence. Diversity comes from prompt variation, model variation, or temperature. Without it, voting is theater.

**Premature parallelization.** Some teams parallelize tasks that aren't independent ("draft" running in parallel with "summarize the draft"). The "summary" gets garbage. Confirm independence before fanning out.

::: gotcha
The most common parallelization bug: running asyncio.gather without `return_exceptions=True`. One specialist throws; gather raises; everyone else's results are lost in the propagated exception. The default is "all-or-nothing." For fan-out workflows where partial results are still valuable, you almost always want `return_exceptions=True` and explicit handling of the per-branch failures.
:::

---

## Section 5 — Pattern 4: Orchestrator-Workers

A central LLM dynamically decomposes the task and delegates to worker LLMs, then synthesizes their results. Looks similar to parallelization, with one critical difference: the *structure of the work isn't known in advance.* The orchestrator looks at the input and decides how to split it.

```
[Input] → [Orchestrator: decompose into subtasks] →
          ┌─→ [Worker 1] ─┐
          ├─→ [Worker 2] ─┼─→ [Synthesizer] → [Output]
          └─→ [Worker N] ─┘
```

**When to use:** the decomposition depends on the input. A code change might affect 3 files or 30; you can't pre-define the worker fan-out. A research question might decompose into 4 sub-questions or 12; the orchestrator decides.

**When NOT to use:** the decomposition is static (parallelization is right). The decomposition needs to be revised mid-flight based on intermediate findings (you're approaching agent territory; consider the agent loop).

This pattern is the closest workflow pattern to "agentic." Module 13 will reveal that Anthropic's multi-agent research system is essentially a scaled-up orchestrator-workers shape with agent-shaped workers — but in workflow form, the orchestrator and workers each make a single LLM call and the structure is deterministic from the orchestrator's plan.

### A worked example: codebase change planner

A coding agent gets a feature request: "add OAuth login with Google and GitHub." The work decomposes into "edit auth module," "edit user model," "add OAuth callback routes," "update frontend login UI," "add tests." The number and identity of these subtasks depends on the codebase and the request — you can't pre-compute the fan-out.

The orchestrator looks at the request and the codebase summary, plans the subtasks, dispatches to workers (each focused on one file or area), and the synthesizer assembles the changes into a coherent diff.

```python
class SubTask(BaseModel):
    description: str
    files: list[str]
    estimated_complexity: str = Field(description="low | medium | high")


class TaskPlan(BaseModel):
    subtasks: list[SubTask]
    overall_approach: str


class WorkerOutput(BaseModel):
    subtask_index: int
    summary: str
    proposed_changes: list[dict]  # {file, before_excerpt, after_excerpt, rationale}


async def orchestrator_plan(feature_request: str, codebase_summary: str) -> TaskPlan:
    response = await client.messages.create(
        model="claude-opus-4-7",  # Orchestrator uses the strongest model
        max_tokens=3000,
        system=(
            "You are a senior engineer planning a feature change. Decompose the "
            "request into focused subtasks, each touching a coherent set of files. "
            "Aim for 3-7 subtasks; each should be independently implementable."
        ),
        messages=[{"role": "user", "content": (
            f"FEATURE REQUEST:\n{feature_request}\n\n"
            f"CODEBASE SUMMARY:\n{codebase_summary}\n\n"
            "Produce a task plan."
        )}],
        tools=[{
            "name": "submit_plan",
            "description": "Submit the decomposed task plan.",
            "input_schema": TaskPlan.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "submit_plan"},
    )
    block = next(b for b in response.content if b.type == "tool_use")
    return TaskPlan(**block.input)


async def worker_implement(subtask: SubTask, idx: int, file_contents: dict) -> WorkerOutput:
    relevant = {f: file_contents.get(f, "<file not found>") for f in subtask.files}
    response = await client.messages.create(
        model="claude-sonnet-4-6",  # Workers use a cheaper model; orchestrator gates quality
        max_tokens=4000,
        system=(
            "You implement a focused subtask within a larger feature. "
            "Produce minimal, correct changes. Cite the lines you would change."
        ),
        messages=[{"role": "user", "content": (
            f"SUBTASK: {subtask.description}\n"
            f"FILES TO MODIFY: {subtask.files}\n\n"
            f"CURRENT CONTENTS:\n"
            + "\n\n".join(f"--- {f} ---\n{c}" for f, c in relevant.items())
        )}],
        tools=[{
            "name": "submit_changes",
            "description": "Submit the proposed changes for this subtask.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "proposed_changes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {"type": "string"},
                                "before_excerpt": {"type": "string"},
                                "after_excerpt": {"type": "string"},
                                "rationale": {"type": "string"},
                            },
                            "required": ["file", "rationale"],
                        },
                    },
                },
                "required": ["summary", "proposed_changes"],
            },
        }],
        tool_choice={"type": "tool", "name": "submit_changes"},
    )
    block = next(b for b in response.content if b.type == "tool_use")
    return WorkerOutput(subtask_index=idx, **block.input)


async def synthesizer_combine(
    feature_request: str, plan: TaskPlan, outputs: list[WorkerOutput]
) -> str:
    response = await client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4000,
        system=(
            "You synthesize a coherent change plan from worker outputs. "
            "Identify any conflicts between subtasks, surface them, and produce "
            "a unified summary suitable for code review."
        ),
        messages=[{"role": "user", "content": (
            f"ORIGINAL REQUEST: {feature_request}\n\n"
            f"PLAN: {plan.model_dump_json(indent=2)}\n\n"
            f"WORKER OUTPUTS:\n{[o.model_dump() for o in outputs]}"
        )}],
    )
    return response.content[0].text


async def orchestrator_workers_workflow(
    feature_request: str, codebase_summary: str, file_contents: dict
) -> str:
    # 1. Plan
    plan = await orchestrator_plan(feature_request, codebase_summary)

    # 2. Dispatch workers in parallel
    outputs = await asyncio.gather(*[
        worker_implement(st, i, file_contents) for i, st in enumerate(plan.subtasks)
    ], return_exceptions=True)

    # 3. Filter failures (we cover deeper failure handling in M14)
    successes = [o for o in outputs if isinstance(o, WorkerOutput)]
    if len(successes) < len(plan.subtasks) * 0.5:
        raise OrchestrationFailed(
            f"Too many worker failures: {len(successes)}/{len(plan.subtasks)} succeeded"
        )

    # 4. Synthesize
    return await synthesizer_combine(feature_request, plan, successes)


class OrchestrationFailed(Exception):
    pass
```

### What this pattern buys you

**Dynamic decomposition.** The pattern handles inputs of varying shape. A 1-line typo fix becomes a single-worker plan; a feature touching ten files becomes a ten-worker plan. The same workflow code handles both.

**Different models for different roles.** The orchestrator uses Opus 4.7 because plan errors propagate. Workers use Sonnet 4.6 because they're focused and there are many of them. The synthesizer uses Opus again because it's reading all the worker outputs and producing a coherent summary. Cost is concentrated where quality matters most.

**Parallel execution after planning.** Once the orchestrator decides on N subtasks, they fan out and run concurrently. Wall-clock time is dominated by the orchestrator + the slowest worker + the synthesizer.

### Pitfalls

**Plan errors propagate.** If the orchestrator decomposes badly — misses a subtask, conflates two — the workers can't fix it. Investing in the orchestrator's prompt and giving it a strong model is non-optional.

**Synthesis loses detail.** When the synthesizer compresses N worker outputs into a summary, important details get dropped. For tasks where the worker outputs are themselves the deliverable (here, code changes), the synthesizer should preserve them, not summarize them. Synthesize the *narrative*; pass the *artifacts* through.

**Worker outputs conflict.** Two workers modify the same file with conflicting changes. The synthesizer should detect and surface this; the workflow shouldn't blindly merge. We deepen this in Module 14 when we cover multi-agent failure handling.

**Looks-like-an-agent confusion.** Teams sometimes implement orchestrator-workers and call it "multi-agent." It's not. The orchestrator and workers each make a single LLM call; there's no autonomous loop. This is a workflow with structured fan-out. Calling it multi-agent inflates the perceived complexity and obscures what's actually deterministic.

::: brain
Look at the orchestrator_plan function. The orchestrator's prompt says "aim for 3-7 subtasks." What if the right answer is 12? What if it's 1?

(The fixed range is a heuristic that biases the orchestrator toward "reasonable" plans. The cost: you'll occasionally get suboptimal decompositions. The benefit: you'll never get a runaway 200-subtask plan or a degenerate 1-subtask plan that doesn't decompose at all. For most production systems, a soft constraint in the prompt is the right call. For systems where the right number genuinely varies wildly, drop the constraint and add a downstream check on plan reasonableness.)
:::

---

## Section 6 — Pattern 5: Evaluator-Optimizer

One LLM produces an output; another LLM evaluates it and provides feedback; the producer revises. Loop until the evaluator approves or a cap is hit.

```
[Input] → [Producer] → [Output]
              ↑           ↓
              └─[Revise]─[Evaluator]─[approves? → done]
```

**When to use:** quality criteria are clear, iterative refinement provides measurable value, and the producer's first attempt is variable enough that evaluation catches real issues. Translation with a separate evaluator for nuance. Code generation with a separate critic for correctness. Long-form writing with a separate editor.

**When NOT to use:** the criteria are vague (the evaluator's feedback will be vague). The producer has hit its quality ceiling (more iterations don't help). The producer and evaluator are the same model with the same prompt (it's confirmation bias dressed as evaluation).

### A worked example: marketing-copy refinement

A marketing team needs product descriptions for an e-commerce catalog. The descriptions need to:

- Hit a target tone (warm, confident, not breathless)
- Be 80-120 words
- Mention three specific product attributes (each product has different attributes)
- Pass brand-voice guidelines (no jargon, no superlatives like "revolutionary")

A single prompt that tries to do all this often misses one criterion. The evaluator-optimizer pattern is exactly right: producer drafts, evaluator scores against the explicit criteria, producer revises with specific feedback.

```python
class CopyEvaluation(BaseModel):
    tone_score: int = Field(ge=0, le=10, description="Warmth and confidence, 0-10")
    word_count: int
    word_count_in_range: bool
    attributes_mentioned: list[str]
    all_attributes_mentioned: bool
    brand_voice_violations: list[str]
    overall_pass: bool
    feedback: str = Field(description="Specific suggestions if not passing")


async def producer_draft(
    product: dict, attributes: list[str], previous_attempt: str | None = None,
    feedback: str | None = None,
) -> str:
    system = (
        "You write product descriptions for an e-commerce catalog. "
        "Tone: warm, confident, not breathless. "
        "Length: 80-120 words. "
        "Brand voice: no jargon, no superlatives like 'revolutionary' or 'best-in-class'."
    )
    user_content = f"PRODUCT: {product['name']}\nATTRIBUTES TO MENTION: {attributes}"
    if previous_attempt and feedback:
        user_content += (
            f"\n\nPREVIOUS ATTEMPT:\n{previous_attempt}\n\n"
            f"REVISE based on this feedback:\n{feedback}"
        )

    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text


async def evaluator_score(copy: str, attributes: list[str]) -> CopyEvaluation:
    response = await client.messages.create(
        model="claude-sonnet-4-6",  # Different prompt makes it functionally distinct
        max_tokens=1000,
        system=(
            "You evaluate product copy against strict criteria. Be honest; do not pass "
            "copy that fails any required check. Return specific, actionable feedback."
        ),
        messages=[{"role": "user", "content": (
            f"COPY TO EVALUATE:\n{copy}\n\n"
            f"REQUIRED ATTRIBUTES: {attributes}\n\n"
            "Score: tone (0-10), word count, attributes mentioned, brand voice violations. "
            "Mark overall_pass true only if: tone >= 7 AND word_count_in_range AND "
            "all_attributes_mentioned AND no brand_voice_violations."
        )}],
        tools=[{
            "name": "submit_evaluation",
            "description": "Submit the evaluation.",
            "input_schema": CopyEvaluation.model_json_schema(),
        }],
        tool_choice={"type": "tool", "name": "submit_evaluation"},
    )
    block = next(b for b in response.content if b.type == "tool_use")
    return CopyEvaluation(**block.input)


async def evaluator_optimizer_workflow(
    product: dict, attributes: list[str],
    max_iterations: int = 3,
) -> tuple[str, CopyEvaluation]:
    copy = await producer_draft(product, attributes)
    evaluation = await evaluator_score(copy, attributes)

    iteration = 1
    while not evaluation.overall_pass and iteration < max_iterations:
        copy = await producer_draft(
            product, attributes,
            previous_attempt=copy, feedback=evaluation.feedback,
        )
        evaluation = await evaluator_score(copy, attributes)
        iteration += 1

    # Return last attempt regardless; caller can check evaluation.overall_pass
    return copy, evaluation
```

### What this pattern buys you

**Higher final quality on tasks with clear criteria.** When the criteria are well-defined and the producer is variable, the iterative refinement actually moves quality. The evaluator catches missed attributes; the producer fixes them. After 2-3 rounds, output quality is meaningfully better than first-shot.

**Visible quality criteria.** The evaluator's structured output (`CopyEvaluation`) is auditable. You can review which criteria pass and fail in production traces. When something ships that shouldn't have, you can see the criterion that should have caught it.

**Calibrated iteration count.** Most tasks converge in 1-2 iterations. The cap (3 here) catches the rare task that's iterating without progress.

### Pitfalls

**Vague criteria → vague feedback → vague revision.** This is the dominant failure mode. "Make it better" produces "different but not better." The evaluator's prompt must encode specific, checkable criteria. If you can't write them down, you can't iterate on them.

**Confirmation bias when producer == evaluator.** Same model, same prompt = self-evaluation = passes everything. Use different prompts at minimum; use different model families when you can. Module 11 covers this in depth for reflection.

**Quality plateau.** The producer's ceiling is a real thing. After 2-3 iterations, additional rounds usually move quality 0%. Capping iterations at 2-3 is the right default; longer caps just waste tokens.

**Infinite loop on contradictory feedback.** Evaluator says "shorter"; producer shortens; evaluator now says "missing detail." Bound iteration count *unconditionally*. The cap is your defense.

::: gotcha
The single most expensive mistake in evaluator-optimizer workflows: not capping iterations. The model reaches a quality plateau but the evaluator still finds nits; the producer revises pointlessly; the loop continues until your budget burns out. Always cap at 2-3 unless you have specific reason to go higher. The marginal quality past iteration 3 rarely justifies the marginal cost.
:::

---

## Section 7 — The Same Task, Three Patterns

The patterns aren't always exclusive — many tasks could be approached multiple ways. Here we take one realistic task and implement it three ways to make the trade-offs concrete.

**The task:** generate a 1500-word competitive analysis of a target company against three competitors. Output should cover: market positioning, pricing, feature differentiation, customer reviews, and strategic vulnerabilities.

### Approach A: Prompt chain (sequential)

```
[Inputs] → [Step 1: research target] → [Step 2: research competitor 1] →
[Step 3: research competitor 2] → [Step 4: research competitor 3] →
[Step 5: synthesize comparative analysis] → [Output]
```

**Predicted profile.** Cost: 5 calls × ~3000 tokens average = ~15K input + ~8K output. ~$0.10 on Sonnet 4.6. Latency: ~5 × 6 sec = ~30 sec wall-clock. Quality: dependent on chain coherence; if step 1 finds a key fact, steps 2-4 might miss it because they don't see step 1's output.

### Approach B: Parallelization (sectioning)

```
[Inputs] → ┌─→ [Worker: target research] ─┐
           ├─→ [Worker: competitor 1] ────┼─→ [Synthesizer] → [Output]
           ├─→ [Worker: competitor 2] ────┤
           └─→ [Worker: competitor 3] ────┘
```

**Predicted profile.** Cost: 4 parallel calls + 1 synthesizer = 5 calls. Same total tokens as Approach A. Latency: ~max(parallel) + synthesizer = ~8 + 6 = ~14 sec. Quality: research is independent (good), but the synthesizer has to combine four reports without seeing them in flight.

### Approach C: Orchestrator-workers (dynamic)

```
[Inputs] → [Orchestrator: plan analysis sections] →
           ┌─→ [Worker: section 1] ─┐
           ├─→ [Worker: section 2] ─┼─→ [Synthesizer] → [Output]
           └─→ [Worker: section N] ─┘
```

**Predicted profile.** Cost: 1 orchestrator call (Opus) + N worker calls (Sonnet) + 1 synthesizer (Opus). For 5 sections: ~7 calls, biased toward more expensive model on the orchestrator and synthesizer. ~$0.30. Latency: orchestrator + max(workers) + synthesizer ~= 4 + 8 + 6 = ~18 sec. Quality: orchestrator can plan around the specific input — adapt section structure based on whether competitors are hardware vs SaaS, etc.

### Comparison

| Aspect | A: Chain | B: Parallel | C: Orchestrator-Workers |
|---|---|---|---|
| Cost | ~$0.10 | ~$0.10 | ~$0.30 |
| Latency | ~30s | ~14s | ~18s |
| Steps known upfront | Yes | Yes | No (orchestrator decides) |
| Adapts to input shape | No | No | Yes |
| Parallelism | None | High | Medium (after planning) |
| Synthesis complexity | Low | High | Medium |
| Failure isolation | Step-by-step | Per-branch | Per-branch + synthesis |

**Which to pick?**

Approach A (chain) wins when the structure is genuinely sequential — step N depends on N-1's output. For competitive analysis, the steps are largely independent, so chain is the wrong shape; you're paying latency for sequencing you don't need.

Approach B (parallel) wins when the structure is fixed and known. For "target + 3 competitors" — exactly four research arms — parallelization is the right shape. You get the wall-clock speedup, you keep cost low (no orchestrator/synthesizer using the strongest model), and the synthesizer's job is straightforward (compile four parallel reports).

Approach C (orchestrator-workers) wins when the structure varies. If "competitive analysis" sometimes means three competitors and sometimes nine, sometimes hardware-focused and sometimes SaaS-focused, the orchestrator earns its cost. For a fixed-shape task, it's overkill.

For Sam's specific deal-research feature, Approach B is the right choice: deals always include the target plus a known list of comparison companies. The wall-clock matters (AEs are watching the spinner). The cost matters (this runs on every deal). Orchestrator-workers becomes interesting later, when the team adds variable analysis types — for the v1, parallelization wins.

::: pullquote
The right pattern is the one that matches the *shape of your task*, not the one that sounds most sophisticated. Parallel beats chain when steps are independent. Orchestrator-workers beats parallel when the decomposition varies. Each pattern has the workload it earns.
:::

::: code-exercise
**Exercise 3.1 — Three patterns, one task.**

Pick a task you actually need to build (or one you've seen). Sketch it three ways: prompt chain, parallelization, orchestrator-workers. For each, estimate:

1. Number of LLM calls
2. Token usage (rough; multiply max_tokens by call count)
3. Wall-clock latency (sum of sequential steps; max of parallel)
4. Cost in USD using current model pricing
5. Failure isolation (what happens if one step fails?)

Pick the winner. Then implement it. Compare your prediction to the actual numbers when you run.

This is the calibration that turns "I know the patterns" into "I can pick the right pattern." Most engineers have to do this exercise three or four times before the choice becomes instinctive.
:::

---

## Section 8 — Composing Patterns

Real production workflows rarely live in one pattern. They compose. Sam's deal-research workflow, sketched on the conference room whiteboard, ends up looking like this:

```
[Deal info] →
  [Routing: deal type → standard or M&A research path] →
    [Standard path:]
      [Parallelization: target / competitors / market research run concurrently] →
      [Orchestrator-workers: synthesizer plans the brief sections, workers write each] →
      [Evaluator-optimizer: critic reviews brief, writer revises]
    [M&A path:]
      [Prompt chain: due-diligence checklist → red flag scan → exec summary]
  →
  [Final brief, possibly with agent layer for follow-up Q&A — see Module 14]
```

Five patterns, layered. The routing decision picks the path. The path itself is a chain of patterns. The output of one pattern is the input to the next.

This is what production looks like. Not "we picked the orchestrator-workers pattern and used it" — but "we picked routing for the entry point, parallelization for the bulk research, orchestrator-workers for the synthesis, and evaluator-optimizer for the polish."

Three principles for composing patterns well:

**1. Each pattern handles a distinct concern.** The router decides type. The parallelization gathers facts. The orchestrator structures output. The evaluator polishes. None of them try to do another's job. When you find a pattern doing two jobs, split it into two patterns.

**2. The interfaces between patterns are structured.** Pattern outputs are typed (Pydantic models), not free-form text. The router emits a `RouteDecision`. The parallel workers emit `SpecialistReport`s. The orchestrator emits a `TaskPlan`. The composition is reliable because each pattern's output has a known shape that the next pattern's input expects.

**3. Failures at one layer are visible to the next.** If parallelization had partial failures, the orchestrator sees the partial set and decides whether to proceed or escalate. Failures don't get hidden by the abstraction; they propagate as data.

The hardened agent loop from Module 2 fits into this composition where it earns its place — typically at the end of the pipeline, doing the open-ended Q&A or follow-up that the workflow couldn't anticipate. The agent isn't replacing the workflow; it's slotted into the workflow at the point where the workflow's predictability ends.

::: brain
Sam's deal-research system has five patterns. If a system has thirteen patterns, what's probably going on?

(Either: (a) the system is genuinely complex and the team has rigorously decomposed it — possible but rare; or (b) the team has over-decomposed and many of those patterns are doing the same job under different names. The smell test: can each pattern's purpose be stated in a sentence without overlapping any other pattern's sentence? If two patterns' descriptions overlap, you have one pattern with two implementations. Merge them.)
:::

---

## Section 9 — When NOT to Use a Workflow at All

The contrarian flip side. Workflows are great. They're not always right.

**Use a single augmented LLM call instead of a workflow when:**

- The task fits in a single well-crafted prompt
- The latency budget can't accommodate sequential steps
- The cost budget is so tight that even one extra LLM call breaks economics
- You're prototyping and the workflow structure isn't yet justified

**Use an agent loop instead of a workflow when:**

- The number of steps depends on what's discovered along the way (Module 2)
- The model needs to react to tool results in unpredictable ways
- The task is genuinely open-ended Q&A or research

**Use a multi-agent system instead of a workflow when:**

- Module 13's threshold test is satisfied (and not before)

**Use code (no LLM) instead of a workflow when:**

- The decisions are deterministic
- The transformations are well-specified
- The cost of an LLM is wasted on a problem that's not LLM-shaped

The last one is the most underrated. Plenty of "AI workflows" have steps that are just data transformation — parse JSON, lookup in a table, format output. Those steps don't need an LLM. The model is being used as a fancy expression evaluator, at LLM cost. Use code; save the LLM calls for the parts that genuinely need language understanding.

::: nodumbq
**Q: How do I know if a step "needs language understanding"?**

A useful test: could a Python function with deterministic logic do this step if I had clear inputs? If yes, that's a function, not an LLM call. Examples that *don't* need an LLM: parsing structured input, looking up values in a known table, validating a schema, formatting output to a template, summing numbers, picking a max. Examples that *do*: classifying free-text intent, generating natural prose, judging fuzzy criteria, synthesizing across heterogeneous sources. Most workflows have both kinds of steps; identify and code the deterministic ones, save LLM calls for the language ones.
:::

---

## Section 10 — Putting It Together

A short framework for picking and composing patterns, based on the questions you should ask yourself for each step of any system you're designing.

**Question 1: Does this step need an LLM at all?**

If no: write a function. Don't add an LLM where deterministic code suffices.

If yes: continue.

**Question 2: Is this step's structure known in advance?**

If yes: pick from chain, parallelization, evaluator-optimizer based on whether steps are sequential, independent, or iterative.

If no: orchestrator-workers (workflow) or agent loop (Module 2).

**Question 3: Does this step take inputs of varying type/shape?**

If yes: routing pattern picks the right downstream handler.

**Question 4: Does this step have clear quality criteria worth iterating on?**

If yes: evaluator-optimizer adds value.

If no: don't iterate. One pass.

**Question 5: Are the sub-results combinable into a clean output?**

If yes: parallelization or orchestrator-workers, depending on whether structure is fixed.

If no: chain, where each step's output is an intermediate the next step transforms.

These questions cascade. Most production workflows answer them differently at different levels (composition). Once you've answered them for each step, you have your composed workflow.

The patterns aren't a checklist to climb. They're a vocabulary for naming the shapes you'll need. Most weeks, you'll find yourself reaching for them not because you set out to "use workflow patterns" but because you have a problem and the patterns name what you'd build anyway. That's the right way to use them.

---

## Recap: Module 3 in eight bullets

::: bullet-points
- Five canonical workflow patterns: prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer. They cover the majority of production AI features.
- Workflows are predictable, debuggable, and cost-bounded in ways agent loops aren't. Reach for workflows first; reach for the agent loop only when the workflow shape genuinely doesn't fit.
- Prompt chaining: sequential steps with optional gates. Each step does one job well. Always have at least one structural gate to prevent error compounding.
- Routing: classifier dispatches to specialists. Optimize for cost (cheap router, specialized specialists) and quality (focused prompts, model variation per category). Confidence threshold + fallback for low-confidence cases.
- Parallelization: fan out independent subtasks (sectioning) or repeat for confidence (voting). Always use `return_exceptions=True` so partial failures don't kill the whole task.
- Orchestrator-workers: dynamic decomposition. The orchestrator picks the structure based on the input. Use Opus-class for orchestrator and synthesizer, Sonnet-class for workers. Plan errors propagate, so don't economize on the orchestrator.
- Evaluator-optimizer: producer drafts, evaluator scores, producer revises. Specific criteria, not vibes. Cap iterations at 2-3; gains plateau fast.
- Real systems compose patterns. Routing into parallelization into orchestrator-workers into evaluator-optimizer is a normal shape. The agent loop slots in where the workflow's predictability ends.
:::

---

::: sam-arc
**Sam, after Wednesday's planning meeting.**

Sam left the conference room with a whiteboard photo on their phone and a clear architecture in their head. The deal-research system, in workflow vocabulary:

- **Routing layer:** deal type (standard / M&A) determines path
- **Parallel research:** target, competitors, market — three concurrent worker calls
- **Orchestrator-synthesizer:** plans the brief structure based on what the research returned, then drafts each section
- **Evaluator-optimizer:** brief critic reviews and revises (2 rounds max)
- **Agent loop (Module 14 territory):** for the AE's follow-up questions on a specific brief — open-ended, tool-using, genuinely agent-shaped

When the PM asked again, Friday, "so... how many agents are we building?" Sam said one. One agent, surrounded by four workflow layers, each doing what it does well.

The PM liked the answer better when it was specific. The cost estimate was specific. The latency estimate was specific. The failure modes were specific. None of those specifics had been reachable when the language was "we're building an agent system."

Sam's arc this module: **restraint. The competence from Module 2 says you can build the agent. The restraint from Module 3 says you should put it where it belongs — surrounded by workflows that do the structured work.**
:::

---

## What's next

Module 4 is the first of the building-blocks chapters: tools. The agent loop and the workflows we've built so far have leaned heavily on tools (web search, structured output, custom handlers) — but we haven't designed them carefully. Tool design is where most agent failures actually originate. We'll cover what makes a good tool description, how to structure schemas the model can use, the confused-deputy problem, idempotency, dry-run patterns, and the "tool result poisoning" failure mode that crashes more agents than any other single cause.

After Module 4, you'll know what to put inside your hardened loop and your workflow steps. The patterns are how you compose; the tools are what the model actually calls.

Sam wrote the v1 of the deal-research workflow on Thursday. It worked. They'll be back in Module 4 to figure out why one of the tools — a "company search" wrapper they slapped together — keeps producing results that the synthesizer treats with too much confidence. Tools are next.
