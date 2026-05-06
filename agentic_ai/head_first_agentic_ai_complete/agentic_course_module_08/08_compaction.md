# Module 8 — Compaction and Summarization

::: chapter-opener
<div class="module-num">MODULE 8</div>
<div class="module-title">Compaction and Summarization</div>
<div class="subtitle">When the trajectory grows past the model's working memory, careful upstream pruning isn't enough.<br>Three techniques for keeping long-running agents coherent — and the modern primitives that automate the boring parts.</div>
<div class="pages">~38 pages · what to do when context fills up anyway</div>
:::

::: hook
The deal-research synthesizer was sharp now — Sam had spent Thursday rebuilding it with Module 7's discipline. The brief generation flow was running cleanly.

But there was a second agent in the system. The "deal Q&A" agent — the one AEs talked to *after* a brief had been generated, asking follow-up questions during prep for a customer call. That one was a real agent: open-ended loop, tools for fetching more research, no fixed shape. And on Friday morning, an AE pinged Sam: *"Q&A agent is acting weird on the Mercer prep. It keeps repeating itself, and I asked about leadership and it answered something about supply chain."*

Sam pulled the trace. Forty-seven turns into the session. The context was 220,000 tokens. The agent had pulled three full briefs, six market research reports, and twelve company lookups. Every tool result was still in context. The system prompt was at the top; the AE's actual current question — "what changed in their leadership last quarter?" — was buried at message 89, surrounded by stale tool outputs from twenty minutes ago.

The model wasn't broken. The trajectory had outgrown its working memory.

Sam looked at the trace and noticed something else, almost worse. Around turn 30, the agent had begun *summarizing prematurely* — it had started wrapping up answers it shouldn't have wrapped up, behaving as if it were running out of room. Anthropic's term for this had a name Sam recognized from the docs: "context anxiety." The model had sensed the impending limit and was rushing.

Module 7 had taught Sam how to make context lean. Module 8 was about what to do when even lean context wasn't enough — when the trajectory itself was the thing exceeding capacity.
:::

---

## What this module is

Module 7 was about what to put *in* context. This module is about what to do when the trajectory grows past what your upstream discipline can keep small. The principles from M7 still apply — every token still costs attention budget — but for long-running agents, you also need active management *during* the trajectory.

Three techniques, in order of operational simplicity:

1. **Tool result clearing.** Drop the bodies of stale tool results that the agent has already processed, replacing them with markers. The lightest-touch form of context management. Anthropic's Developer Platform now offers this as a built-in `clear_tool_uses_20250919` strategy.

2. **Compaction.** When the trajectory approaches the limit, summarize the message history into a compact form and continue with the summary plus recent activity. The most common technique for long agentic sessions; available as a server-side beta on the platform.

3. **Structured note-taking.** The agent writes notes to a persistent surface (memory tool, file, scratch document) outside the context window. Reads them back when needed. The most explicit and the most powerful technique — and the bridge to Module 9's memory-across-runs.

A fourth approach — multi-agent architectures with sub-agent context isolation — gets a brief mention here and the deep treatment in Module 13.

By the end of this module:

- You'll know when each technique is the right one and how they compose
- You'll have working code for all three, plus the production beta primitives
- You'll understand the "art of compaction" — what to keep vs discard, how to test, how to avoid losing critical context invisibly
- You'll know the failure modes that show up when these techniques are misapplied (because they have specific, recognizable failure modes)

We're following Anthropic's own playbook closely. The September 2025 essay on context engineering called out these three techniques specifically; the platform features that automate them have shipped through 2025 and 2026. We'll use the platform features where they're available and show the from-scratch implementations where they aren't, so you understand what's happening underneath.

::: pullquote
The agent loop in Module 2 cared about what happens within one turn. Module 8 cares about what happens across fifty.<br>The challenge isn't the loop — it's the trajectory.
:::

---

## Section 1 — When You Need This

A short triage section. The techniques in this module add complexity. Not every agent needs them.

**You probably need active context management when:**

- Trajectories routinely exceed 30+ turns
- Tool results frequently exceed 5,000 tokens each
- The agent's task spans tens of minutes or longer
- Sessions are stateful — same conversation continues across many user messages
- You've seen "context anxiety" symptoms (the agent wrapping up early, declaring tasks done that aren't, summarizing prematurely)

**You probably don't need it when:**

- Agent runs are short (< 10 turns) and one-shot
- Tool results are small (< 1K tokens each)
- Tasks complete in a single conversation turn
- You haven't yet hit the issues this module solves

The principle: **don't preempt these techniques.** They earn their complexity when the trajectory genuinely demands them. A 5-turn agent with tight tool results doesn't need compaction. Adding it anyway just adds failure modes (sub-summarization losing context, etc.) without solving a problem you have.

The diagnostic: trace your agent's longest real session. Plot tokens-in-context per turn. If the curve is flat or shallow (each turn adds ~1-3K tokens of meaningful new content), you're probably fine. If the curve climbs steeply (every turn adds 5-10K tokens, mostly tool results that won't be relevant later), you have a Module 8 problem.

::: brain
Sam's deal-research workflow runs ~7 turns per request and produces a single output (the brief). Sam's deal Q&A agent runs 30-50+ turns per session as the AE asks follow-ups. Which one needs Module 8 techniques?

(The Q&A agent. The workflow's trajectory is short and bounded; even with rich tool results, it's a single pass. The Q&A agent compounds: each user question adds context, each tool call adds more, each subsequent question reads everything from before. Forty turns in, the context is dominated by stale evidence from earlier questions. Module 8 is for the Q&A agent. The workflow keeps the Module 7 discipline; the agent gets the Module 8 layering on top.)
:::

---

## Section 2 — Tool Result Clearing: The Lightest Touch

Of the three techniques, tool result clearing is the cheapest to adopt and often the highest-impact. The premise is simple: once the agent has processed a tool result, the *raw* result is rarely needed in subsequent turns. The agent has already extracted what it cared about; the original 8K tokens of customer record is dead weight from turn 4 onward.

Anthropic's framing in their context engineering essay: *"once a tool has been called deep in the message history, why would the agent need to see the raw result again?"*

The platform now ships this as a built-in strategy.

### The Anthropic Developer Platform feature

The `clear_tool_uses_20250919` strategy clears tool results when the conversation context grows beyond a configurable threshold. The API replaces each cleared result with placeholder text so the model knows the result *was there* — preserving the conversational structure — but the body is gone.

```python
import anthropic

client = anthropic.Anthropic()

response = client.beta.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    messages=[...],
    tools=[...],
    betas=["context-management-2025-06-27"],
    context_management={
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                # Optional configuration:
                # - trigger: when to start clearing (default: ~80% of context limit)
                # - clear_oldest: how aggressively (default: oldest first)
                # - clear_tool_inputs: whether to clear tool call params too (default: false)
            }
        ],
    },
)
```

That's it. The platform handles the rest. When the context approaches the configured threshold, the oldest tool results get replaced with placeholder text. The model continues without losing the structural sense of "I called X tool earlier."

The key configuration choice is `clear_tool_inputs`. The default — clear results, keep the tool call parameters — preserves more semantic content (the agent can still see *what* it asked for) while reclaiming the bulk of the tokens (the result body). For most workloads, this default is right. Set `clear_tool_inputs: true` only when you're under heavy context pressure and the tool call parameters themselves are taking meaningful space.

On Anthropic's internal evaluations, this single feature reduced token consumption by 84% on a 100-turn web search benchmark while *enabling* the agent to complete workflows it would otherwise have failed on. That's the rare case where a feature is both cheaper and better.

### Why this is the right starting point

Three reasons tool result clearing is what you should adopt first if you're hitting context limits:

**It's nearly invisible to the agent.** The model sees "I called X tool, got a placeholder back" rather than "the universe has been edited around me." The trajectory's logical structure stays intact. Compare to compaction (Section 3), which replaces messages wholesale — much more invasive.

**It targets the highest-volume waste.** In most agent traces, tool results dominate context size. Modules 4 and 7 covered tool-level discipline (truncation, response_format, etc.); tool result clearing handles the bodies that survived that discipline and have since become stale.

**It composes with other techniques.** You can combine tool result clearing with compaction, with structured note-taking, or both. It's additive, not exclusive.

### Doing it from scratch

If you're not on the Anthropic Developer Platform — Bedrock or Vertex deployments, or for some reason you need client-side control — the same pattern is straightforward to implement:

```python
def clear_old_tool_results(
    messages: list[dict],
    *,
    keep_recent: int = 5,
    placeholder: str = "[tool result cleared from context]",
) -> list[dict]:
    """
    Replace the bodies of old tool_result blocks with a placeholder.
    Keep the most recent N tool results intact.

    Returns a new messages list; doesn't mutate the input.
    """
    # First pass: count tool_result blocks
    tool_result_indices: list[tuple[int, int]] = []  # (message_idx, block_idx)
    for mi, msg in enumerate(messages):
        if isinstance(msg.get("content"), list):
            for bi, block in enumerate(msg["content"]):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_result_indices.append((mi, bi))

    # Decide which to clear: all but the last N
    if len(tool_result_indices) <= keep_recent:
        return messages
    to_clear = tool_result_indices[: -keep_recent] if keep_recent > 0 else tool_result_indices

    # Second pass: rebuild messages with cleared bodies
    cleared = set(to_clear)
    new_messages = []
    for mi, msg in enumerate(messages):
        if isinstance(msg.get("content"), list):
            new_content = []
            for bi, block in enumerate(msg["content"]):
                if (mi, bi) in cleared:
                    new_content.append({
                        **block,
                        "content": placeholder,
                    })
                else:
                    new_content.append(block)
            new_messages.append({**msg, "content": new_content})
        else:
            new_messages.append(msg)
    return new_messages
```

Used in the agent loop:

```python
# In your hardened agent loop (Module 2):
async def run_agent_with_tool_clearing(
    user_query: str,
    tools: list[dict],
    tool_handlers: dict[str, callable],
    *,
    model: str = "claude-sonnet-4-6",
    keep_recent_tool_results: int = 5,
    clear_threshold_tokens: int = 100_000,
):
    messages = [{"role": "user", "content": user_query}]
    while True:
        # Estimate token usage; if above threshold, clear old tool results
        if estimate_tokens(messages) > clear_threshold_tokens:
            messages = clear_old_tool_results(
                messages,
                keep_recent=keep_recent_tool_results,
            )

        response = await client.messages.create(
            model=model, max_tokens=4096, tools=tools, messages=messages,
        )
        # ... rest of the loop unchanged
```

The platform feature is better when available — it knows exact token counts, integrates with caching, and handles edge cases. The from-scratch version covers the same idea when you can't use the platform.

::: gotcha
A subtle bug: clearing the body of a tool_result also affects prompt caching. If your cache prefix includes any tool results, modifying them invalidates the cache. The Anthropic feature handles this by clearing results *outside* the cached prefix where possible; if you're clearing manually, be aware that aggressive clearing trades cache savings for context savings. For workloads with strong cache reuse, set `keep_recent` higher to preserve cache validity.
:::

---

## Section 3 — Compaction: Summarize and Continue

When tool result clearing isn't enough — the trajectory's been long enough that even the message history (without tool result bodies) is large, or the structure of the agent's reasoning is itself contributing significantly to the context — compaction is the next lever.

The premise: take the conversation nearing the context window limit, summarize its contents, and reinitialize a new context window with the summary. The agent continues, working from the summary plus its most recent activity.

Anthropic's framing: *"compaction typically serves as the first lever in context engineering to drive better long-term coherence. At its core, compaction distills the contents of a context window in a high-fidelity manner, enabling the agent to continue with minimal performance degradation."*

Two implementations matter: the platform's server-side compaction beta, and the from-scratch version when you need control over what gets preserved.

### Server-side compaction

The platform's compaction beta handles this automatically. You declare a trigger (e.g., when input_tokens crosses a threshold) and the API runs a summarization pass internally, replacing the older history with the summary, before continuing.

```python
response = client.beta.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    messages=[...],
    tools=[...],
    betas=["compact-2026-01-12"],
    context_management={
        "edits": [
            {
                "type": "compact_20260112",
                "trigger": {
                    "type": "input_tokens",
                    "value": 180_000,
                },
            }
        ],
    },
)
```

When input crosses 180K tokens, the API summarizes the older portion of the history and continues. From the agent's perspective, the trajectory is now shorter. From the *user's* perspective, the conversation is unchanged — they don't see the compaction event. From the model's perspective, the older messages are gone, replaced by a high-fidelity summary.

What gets preserved (per Anthropic's own description of how Claude Code does this):

- Architectural decisions
- Unresolved bugs or open questions
- Implementation details and constraints the agent has been holding
- Identity of the user and core task

What gets discarded:

- Redundant tool outputs (already processed)
- Failed approaches the agent has since abandoned
- Conversational filler (acknowledgments, transitions)
- Stale intermediate state

Plus, in Claude Code's specific implementation, the *five most recently accessed files* are preserved as their own block — the recent context the agent is actively working with stays intact even as older context is summarized.

### Client-side compaction

When server-side isn't available or you need fine-grained control, the same pattern is straightforward. The art is in the compaction prompt itself.

```python
COMPACTION_SYSTEM_PROMPT = """
You are a compaction system for an AI agent's conversation history. Your job is
to take a long conversation and produce a high-fidelity compressed summary that
the agent can use to continue working without losing critical context.

PRESERVE:
- The user's original task and any clarifications added since
- Decisions made by the agent and the reasoning behind them
- Open questions, constraints, or assumptions still in play
- Specific facts the agent has gathered (numbers, names, identifiers, dates)
- Tool call outcomes that affect later decisions (e.g., "discovered customer X
  has plan Y", "test suite Z is currently failing on test_w")

DISCARD:
- Verbatim tool outputs that have already been distilled into facts above
- Failed approaches the agent abandoned (mention the abandonment without details)
- Conversational filler (greetings, acknowledgments, transitions)
- Restatements of things already preserved

OUTPUT FORMAT:
A markdown document with these sections:
1. ## Original Task
2. ## Decisions and Reasoning
3. ## Facts Gathered
4. ## Open Questions
5. ## What's Currently in Progress

Be specific. The agent will rely on this summary to continue working; vagueness
becomes failure later. If you're unsure whether something matters, include it.
""".strip()


async def compact_messages(
    messages: list[dict],
    *,
    keep_recent: int = 4,
    summary_model: str = "claude-sonnet-4-6",
) -> list[dict]:
    """
    Summarize all but the last N messages. Return new messages list:
    [system, user(summary), ...last N messages].
    """
    if len(messages) <= keep_recent + 1:
        return messages

    history_to_compact = messages[: -keep_recent]
    recent = messages[-keep_recent:]

    # Render the history into a single string the summarizer can read
    history_text = _render_messages_for_compaction(history_to_compact)

    summary_response = await client.messages.create(
        model=summary_model,
        max_tokens=4000,
        system=COMPACTION_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": (
                f"Compact this agent conversation:\n\n{history_text}"
            )},
        ],
    )
    summary = summary_response.content[0].text

    # Construct the new messages list
    new_first_message = {
        "role": "user",
        "content": (
            f"<compacted_history>\n{summary}\n</compacted_history>\n\n"
            "The above is a summary of our conversation so far. Continue from the "
            "current state described above. The most recent messages follow."
        ),
    }
    return [new_first_message, *recent]


def _render_messages_for_compaction(messages: list[dict]) -> str:
    """Render messages as a readable string for the compaction model."""
    lines = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            lines.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        lines.append(f"[{role}]: {block['text']}")
                    elif block.get("type") == "tool_use":
                        lines.append(
                            f"[{role}/tool_use]: {block.get('name', '?')}"
                            f"({block.get('input', {})})"
                        )
                    elif block.get("type") == "tool_result":
                        result_text = block.get('content', '')
                        if isinstance(result_text, list):
                            result_text = "".join(
                                b.get('text', '') for b in result_text
                                if isinstance(b, dict)
                            )
                        # Truncate tool results in compaction input — the summarizer
                        # doesn't need full bodies, just signal of what was returned
                        if len(result_text) > 1000:
                            result_text = result_text[:1000] + "...[truncated]"
                        lines.append(f"[{role}/tool_result]: {result_text}")
    return "\n".join(lines)
```

Two design choices in this implementation worth flagging:

**The compaction prompt is itself an artifact you tune.** Anthropic's recommendation is "start by maximizing recall, then iterate to improve precision." Initial versions should err toward keeping too much; iterations should remove what turns out to be unnecessary. You'll know your prompt is right when post-compaction quality on long trajectories matches pre-compaction quality on equivalent short ones.

**Recent messages stay verbatim.** The agent's most recent context is the most live — the message it's about to respond to is in there. Compacting that loses the immediate task structure. Keep the last 3-5 messages intact; let compaction handle only the history that's already been processed.

::: pullquote
The art of compaction lies in the selection of what to keep versus what to discard. Overly aggressive compaction loses subtle but critical context whose importance only becomes apparent later.<br>— Anthropic, "Effective Context Engineering for AI Agents"
:::

### Compaction in the agent loop

Wired into the hardened loop from Module 2:

```python
async def run_agent_with_compaction(
    user_query: str,
    tools: list[dict],
    tool_handlers: dict[str, callable],
    *,
    model: str = "claude-sonnet-4-6",
    compaction_threshold_tokens: int = 150_000,
    keep_recent_messages: int = 4,
):
    messages = [{"role": "user", "content": user_query}]
    while True:
        # Compact if approaching threshold
        if estimate_tokens(messages) > compaction_threshold_tokens:
            messages = await compact_messages(
                messages, keep_recent=keep_recent_messages,
            )

        response = await client.messages.create(
            model=model, max_tokens=4096, tools=tools, messages=messages,
        )
        # ... rest of the loop unchanged
```

The threshold matters. Anthropic recommends compacting *before* you hit a hard limit — typically at 75-85% of context window — so the model isn't already in context-anxiety territory when the compaction runs. For Sonnet 4.6's 1M context, 750-850K is the safe trigger; for Haiku 4.5's 200K, 150-170K.

### Pitfalls

**Compaction loses what you didn't tell it to preserve.** The summarizer follows your prompt. If your prompt doesn't mention "preserve experimental constraints," and the agent had been holding an important constraint in dialogue context, compaction may discard it silently. Test on real long traces before relying on compaction in production.

**Aggressive compaction creates surprise.** Going from a 100-message history to a 1500-token summary in one step is a big shift; the agent may behave differently after compaction than before. For trust-sensitive applications, stage compaction (multiple smaller compactions) or use structured note-taking (Section 4) instead.

**Compaction can run too eagerly.** A trigger set too low fires frequently; each compaction call is itself an LLM call (Sonnet-tier, typically). For high-volume systems, eager compaction can dominate cost. Trigger only when actually needed, not preemptively.

::: postmortem
**The Compaction That Forgot the Constraint**

A team built a long-running coding agent that periodically compacted its history. The agent worked across many turns, accumulating constraints from the user ("don't touch the auth module," "preserve the existing test naming convention," "the database schema can't change without DBA review").

After a compaction event around turn 30, the agent decided to "improve the auth module" — directly violating a constraint the user had stated at turn 6. The compaction summary had captured "user wants better security" but lost the specific "don't touch the auth module" prohibition. The agent's next round of edits broke production.

The compaction prompt had said "preserve decisions and constraints" but the summarizer had judged "don't touch auth" as a *concern about scope* rather than a *prohibition*, and bundled it into a general "user wants security improvements" line. The original constraint, stated once 25 turns ago, had been silently softened.

Two fixes shipped together:

1. The compaction prompt grew an explicit "Prohibitions and constraints" section, separate from "decisions" — and the prompt instructed the summarizer to **never** rephrase items in this section, only verbatim copies.

2. The team adopted structured note-taking (Section 4) for any user-stated constraint. The agent learned to call `add_constraint(text=...)` whenever the user stated one; the constraint went to memory; compaction couldn't lose what compaction never touched.

The lesson: compaction is a summarization, and summarizations lose nuance. For things that *cannot* be lost, don't put them in the path of compaction. Move them to memory.
:::

---

## Section 4 — Structured Note-Taking (Agentic Memory)

The third technique. The agent writes notes — outside the context window — to a persistent surface, and reads them back when needed. The context window stays manageable; the important state lives somewhere durable.

This is what humans do. You don't keep your shopping list in your working memory; you write it on a post-it. You don't memorize your team's deployment runbook; you keep it in a document and reference it. The Anthropic essay's framing: *"Like Claude Code creating a to-do list, or your custom agent maintaining a NOTES.md file, this simple pattern allows the agent to track progress across complex tasks, maintaining critical context and dependencies that would otherwise be lost across dozens of tool calls."*

The point of structured note-taking, vs raw memory, is that the *structure* matters. A `notes.md` flat file works for some agents. For richer state, structured types (a todo list with statuses, a constraint list, a fact ledger) help the agent both write good notes and read them with the right framing.

### The Anthropic memory tool

The platform exposes a built-in `memory` tool that gives agents file-based persistence outside the context window. Released alongside Sonnet 4.5 and now broadly available.

```python
response = client.beta.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    messages=[...],
    tools=[
        {"type": "memory_20250818", "name": "memory"},
        # ... your other tools
    ],
    betas=["context-management-2025-06-27"],
)
```

The memory tool exposes file operations (read, write, list, append) on a directory of memory files. Where those files actually live is up to you — the tool calls reach into your client-side handler, which can store them locally, in S3, in a database, anywhere. The model sees a stable file-system-like surface; you provide the storage.

The full memory + context editing combo on Anthropic's internal evaluations: 39% improvement over baseline on agentic search, with the memory tool *plus* tool result clearing. The memory tool alone is significant; the combination is multiplicative because they target different problems (compaction loss vs context bloat).

### A from-scratch note-taking pattern

If you don't want the full memory tool — or want to expose specific note-taking primitives shaped to your task — you can build the pattern out of regular tools. This is often the right call for production agents because it gives you control over the schema.

```python
from typing import Literal

# A simple in-memory note store; in production, persist to your storage.
_notes_store: dict[str, list[dict]] = {}


@tool
def add_note(
    session_id: str,
    category: Literal["fact", "decision", "constraint", "open_question", "todo"],
    content: str,
):
    """
    Record an important note that should persist across long agent trajectories.
    Use this for information that you'll need later but don't want to lose to
    compaction or context limits.

    category:
      - "fact": a piece of information gathered (e.g., "user is on Plan B, $99/month")
      - "decision": a choice made and why (e.g., "decided to use approach X because Y")
      - "constraint": a hard rule the user stated or the system enforces (e.g.,
        "do not modify the auth module"). NEVER paraphrase constraints when adding.
      - "open_question": something unresolved (e.g., "still need to verify Z")
      - "todo": an action you intend to do later (e.g., "follow up on X after Y")

    Notes persist outside the context window. Use list_notes() to read them back
    when relevant.
    """
    if session_id not in _notes_store:
        _notes_store[session_id] = []
    _notes_store[session_id].append({
        "category": category,
        "content": content,
        "added_at_turn": _current_turn(),
    })
    return {"status": "noted", "total_notes": len(_notes_store[session_id])}


@tool
def list_notes(
    session_id: str,
    category: Literal["fact", "decision", "constraint", "open_question", "todo"] | None = None,
):
    """
    Read back notes from the current session. Filter by category if you only need
    a specific type. Use this WHEN you need to remember something stated earlier
    in the conversation that may have been compacted out of context.

    Common patterns:
    - At the start of complex reasoning: list_notes(category="constraint") to
      review what you can and can't do.
    - When the user references something you don't see in immediate context:
      list_notes() to check whether they're referring to an earlier-established fact.
    - Periodically as a sanity check: list_notes(category="open_question") to
      verify you haven't lost track of unresolved items.
    """
    notes = _notes_store.get(session_id, [])
    if category is not None:
        notes = [n for n in notes if n["category"] == category]
    if not notes:
        return {"notes": [], "message": "No notes in this category."}
    return {"notes": notes, "count": len(notes)}
```

The schema is doing important work. The `category` field forces the agent to think about *what kind* of thing it's noting. Constraints get stored as constraints — never paraphrased, never bundled into "concerns." Facts get stored as facts. This structure makes retrieval better: when the agent needs to verify what's in scope, it asks for constraints specifically, not "everything I noted."

The discipline encoded in the prompt:

- Note specific things, not vague summaries
- Categorize correctly so retrieval works
- Never paraphrase constraints (the postmortem case)
- Read notes back at the right moments (start of complex reasoning, when references are unclear)

System prompt guidance the agent needs:

```python
NOTE_TAKING_GUIDANCE = """
For long conversations, you have access to add_note() and list_notes() tools
that persist information outside the context window.

USE add_note() WHEN:
- The user states a constraint, preference, or hard rule
- You make a decision based on multiple factors and the reasoning matters later
- You discover a fact that affects future actions (configuration, account state, etc.)
- You commit to a follow-up action you can't do yet

USE list_notes() WHEN:
- Starting any reasoning that depends on prior context
- The user references something you don't immediately recall
- Before any high-stakes action, to verify you haven't lost track of constraints

DON'T USE add_note() FOR:
- Acknowledgments or conversational filler
- Things that are clearly visible in the most recent few messages
- Information from a single tool call you've already acted on

The notes are your durable memory. Treat them seriously.
"""
```

### Composing structured notes with compaction

Notes and compaction are complementary, not redundant. Compaction summarizes what's *in* the context window into a denser form. Notes preserve what *cannot* be lost regardless of compaction. The combination:

1. The agent calls `add_note()` for any item that would be catastrophic to lose
2. Compaction handles bulk reduction of messages
3. After compaction, the agent's notes are still there — accessible via `list_notes()` whenever needed

The Anthropic Sonnet 4.5 launch results — 39% improvement over baseline — came from this combination, not either alone. The two address different failure modes; using them together is the production pattern.

::: nodumbq
**Q: Doesn't this just push the problem? The notes themselves can grow without bound.**

Yes — and you handle that the same way you'd handle any growing data store. Periodic pruning (notes older than X turns and not referenced recently can be archived), categorization-based limits (you might cap "facts" at 50 per session but allow unlimited "constraints"), or a separate compaction pass on the notes themselves. For short agent trajectories, this rarely becomes a problem; for long-running systems (Module 9's territory), the note-store hygiene is its own engineering concern.

**Q: Should I always use the memory tool, or sometimes use my own custom note-taking tools?**

Both are valid. Use the memory tool when you want a generic file-system-like surface and the agent needs flexibility about how to organize notes. Use custom note-taking tools when you have specific structure that matters (categories, schemas, validation rules) and you'd rather encode that structure in the tool definitions than have the agent infer it. For most production systems, custom tools win — the structure is doing the work, not the storage layer.
:::

---

## Section 5 — When to Pick Which (and How They Compose)

A short field guide. The three techniques aren't competing; they're layered. Most production agents that need this module's content end up using all three.

**Tool result clearing first.** Cheap, automated by the platform, near-invisible to the agent. The default for any agent that hits context pressure. Catches the highest-volume waste.

**Compaction next.** When the message history itself (not just tool results) is the source of bloat. Use server-side compaction when on the platform; client-side when you need control over what's preserved. Most agents that run more than 30 turns need compaction.

**Structured note-taking for things that must not be lost.** Compaction always loses some nuance. For constraints, decisions, and facts where loss would be catastrophic, write them to notes. The agent's reasoning over compaction-survived content is supplemented by notes that are guaranteed to persist.

The composition pattern looks like:

```python
response = client.beta.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    messages=[...],
    tools=[
        {"type": "memory_20250818", "name": "memory"},
        # ... your other tools, including custom note-taking if used
    ],
    betas=["context-management-2025-06-27", "compact-2026-01-12"],
    context_management={
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                # Standard tool result clearing
            },
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 750_000},
                # Compact when we hit ~75% of Sonnet's 1M context
            },
        ],
    },
)
```

Tool results get cleared as the conversation grows. Compaction kicks in at 750K tokens. Memory persists across both. The agent has all three layers active simultaneously.

When *not* to use all three:

- Short trajectories don't need any of this. Module 7's discipline alone handles them.
- For agents on platforms that don't support all three (some Bedrock or Vertex configurations), use what's available; the from-scratch versions in this module fill the gaps.
- For latency-sensitive interactive agents, compaction adds a turn (the summarization call). If 5-10 second pauses are unacceptable, lean harder on tool result clearing and notes; defer compaction or skip it entirely.

::: brain
You have an agent running 50+ turn coding sessions. Tool results dominate context (file reads, test output, search results). The agent occasionally loses track of architectural decisions made earlier. Which techniques?

(All three. Tool result clearing handles the file-read and test-output bulk — that's where the volume is. Compaction handles the slower drift of the message history itself. Notes preserve the architectural decisions specifically — those are the things compaction sometimes softens, and where loss is most damaging. The combination is exactly the production setup for long-running coding agents.)
:::

---

## Section 6 — A Brief Note on Sub-Agent Architectures

The fourth technique Anthropic's essay names: **sub-agent architectures.** Rather than one agent attempting to maintain state across an entire project, specialized sub-agents handle focused tasks with clean context windows. The main agent coordinates with a high-level plan; sub-agents do deep work and return distilled summaries.

This is genuinely the right pattern for some long-horizon work — Anthropic's own research system, for instance, uses sub-agents for focused exploration. Each sub-agent might use tens of thousands of tokens; only a 1-2K-token summary returns to the main agent.

We're not deep-diving sub-agents here because Module 13 does. The brief preview: when your trajectory is long because of *parallelizable* sub-tasks (each requiring its own deep exploration), splitting into sub-agents wins on both context efficiency and quality. When your trajectory is long because of *sequential* dependencies on a single thread, compaction and notes win.

The decision criterion for Module 8 vs Module 13:

- Long single-thread reasoning → Module 8 (compaction + notes)
- Long parallelizable exploration → Module 13 (sub-agents with isolated contexts)
- Long *and* parallelizable *and* needing coherent final synthesis → both, layered

Module 13 will cover the sub-agent patterns in full, with the multi-agent research system as the worked example.

---

## Section 7 — Context Anxiety and the Harness Layer

A specific failure mode worth naming, because it shows up exactly when this module's techniques are most needed.

**Context anxiety** is what Anthropic's engineering team called the phenomenon where a model — Sonnet 4.5 was the documented case — would wrap up tasks prematurely as it sensed its context limit approaching. The agent would start saying "I think we have enough here" or "let me summarize what we have" when the user's task wasn't actually complete. The model was trying to be helpful by not getting cut off mid-thought; the result was incomplete work.

The Sam scenario in the cold open had this. Around turn 30, the Q&A agent had started wrapping up answers prematurely. Not a model failure — a model behavior, in response to context pressure.

The fix happens at the harness layer (the Module 2 loop) rather than via prompt engineering:

**Apply context management *before* the model senses the squeeze.** Tool result clearing at 50% of the context window, compaction at 75%. The model never sees the full 200K tokens; it sees a managed version that doesn't trigger context-anxiety behavior.

**Don't tell the model how much context it has.** Some teams insert "you have N tokens remaining" status messages. This often *triggers* anxiety rather than calibrating; the model hyper-aware of capacity starts conserving artificially. Let the harness handle context management; let the model focus on the task.

**Configure compaction triggers below the natural anxiety threshold.** Anthropic's modeling found 80%+ as the rough threshold where Sonnet 4.5 started behaving anxiously. Modern models are calibrated more carefully, but the principle stands: don't run with fingers crossed against the limit.

```python
# A safe default for Sonnet 4.6's 1M context
SAFE_COMPACTION_THRESHOLDS = {
    "claude-sonnet-4-6": 750_000,    # 75% of 1M
    "claude-opus-4-7": 750_000,      # 75% of 1M
    "claude-haiku-4-5": 150_000,     # 75% of 200K
}

SAFE_TOOL_CLEAR_THRESHOLDS = {
    "claude-sonnet-4-6": 500_000,    # 50% of 1M; clear tool results earlier
    "claude-opus-4-7": 500_000,
    "claude-haiku-4-5": 100_000,
}
```

The split — tool clearing earlier, compaction later — is intentional. Tool clearing is cheap and minimally invasive; doing it preemptively at 50% means the more expensive compaction step might not be needed at all. By the time compaction would fire, you've already shed the bulk of the easy waste.

::: gotcha
A common production mistake: setting compaction trigger to 95%+ of context window because "we want to use all of it." This puts the agent into context-anxiety territory before compaction runs, and the compaction itself happens under pressure (the summarizer is also working with a near-full context). The agent's behavior degrades at exactly the moment you're trying to fix the problem. Trigger at 75%, not 95%. Pay the small overhead of compacting earlier; reap the larger gain of the agent never running near the limit.
:::

---

## Section 8 — Putting It Together: The Long-Running Q&A Agent

Tying everything to Sam's specific case from the cold open. The deal Q&A agent — the one running the 47-turn session that broke — gets the full Module 8 treatment.

The architecture, post-fix:

```python
QA_SYSTEM_PROMPT = """
You are a deal Q&A assistant. AEs ask you follow-up questions during prep
for customer calls. You have tools for fetching deal details, company
research, and past briefs. You also have notes (add_note, list_notes) for
recording things that should persist across the long session.

For each session:
1. Start by calling list_notes() to see if there's prior context.
2. When a constraint or important fact is established, add_note() it
   immediately. Especially for things stated by the AE that affect later answers.
3. Before high-stakes answers, list_notes(category="constraint") to verify
   you're staying within the AE's stated boundaries.
""".strip()


async def run_qa_agent(
    session_id: str,
    initial_question: str,
    *,
    model: str = "claude-sonnet-4-6",
):
    messages = [{"role": "user", "content": initial_question}]

    while True:
        # Estimate token usage
        tokens = estimate_tokens(messages)

        # Tool result clearing at 50% threshold (handled by platform)
        # Compaction at 75% threshold (handled by platform)
        # Notes are always available

        response = await client.beta.messages.create(
            model=model,
            max_tokens=4096,
            system=QA_SYSTEM_PROMPT,
            tools=DEAL_QA_TOOLS,  # includes add_note, list_notes, plus deal tools
            messages=messages,
            betas=["context-management-2025-06-27", "compact-2026-01-12"],
            context_management={
                "edits": [
                    {"type": "clear_tool_uses_20250919"},
                    {
                        "type": "compact_20260112",
                        "trigger": {"type": "input_tokens", "value": 750_000},
                    },
                ],
            },
        )
        messages.append({"role": "assistant", "content": response.content})
        # ... rest of the hardened loop, with note tools added to handlers
```

What changed from Friday morning's broken run:

- Tool results from earlier in the session no longer take up 80% of the context — they get cleared as the conversation grows
- The session can run 100+ turns without hitting compaction; compaction at 75% of 1M handles the rare longer cases
- The AE's stated constraints ("the customer is sensitive about pricing"; "leadership is the focal point of this call") get noted explicitly the moment they're stated, and remain accessible regardless of compaction
- The agent's behavior is no longer dictated by context pressure; it's dictated by the actual task

Sam ran the same Mercer-prep scenario from the cold open. Forty-seven turns again. Different outcome: the agent answered the leadership question accurately, hadn't repeated itself, hadn't drifted to supply chain. The trajectory was 90K tokens at peak; tool clearing had handled the bulk of waste. Compaction never fired (the trajectory wasn't long enough to hit 750K). Notes had captured three constraints the AE had stated; the agent referenced them correctly throughout.

::: code-exercise
**Exercise 8.1 — Audit a long trajectory.**

Pick a real production agent run (or a representative test case) that exceeded 30 turns. Walk through:

1. **Token trajectory.** Plot tokens-in-context per turn. Where does the curve climb steeply? Tool results, message accumulation, or both?
2. **Stale content.** Identify content from turn N that was no longer relevant by turn N+10. What fraction of late-trajectory tokens is content like this?
3. **Critical context.** Identify items that *should* have been preserved across the trajectory — constraints, decisions, facts. Did any of them get lost or softened?
4. **Decide.** Which Module 8 techniques would help most: tool result clearing, compaction, structured notes, or some combination?
5. **Implement and test.** Apply the chosen technique(s). Re-run the same trajectory. Compare quality and token cost.

The goal is calibration. Most teams over-apply (turning on every Module 8 feature for every agent) or under-apply (waiting until users complain). The audit gives you the data to calibrate per agent.
:::

---

## Section 9 — Putting It Together

A short framework for adding context management to a system that's hitting limits:

**1. Diagnose first.** Use the Section 1 triage: do you actually need this module's techniques? Don't preempt complexity for problems you don't have.

**2. Start with tool result clearing.** Cheapest, highest-impact, lowest-risk. Use the platform feature when available; from-scratch otherwise. Set the threshold at 50% of context window.

**3. Add compaction when message history itself bloats.** Server-side compaction is the recommendation when on the platform. Trigger at 75% of context window. Tune the compaction prompt on real long traces; iterate from "preserve everything" toward precision.

**4. Use structured note-taking for must-not-lose content.** Constraints, decisions, key facts. Custom note-taking tools with categories beat freeform memory for production agents. The schema is doing the work.

**5. Compose all three for long-running agents.** They target different problems and stack additively. Anthropic's own benchmarks show the combination outperforms any single technique by significant margins.

**6. Set thresholds below context-anxiety territory.** Don't run near limits with fingers crossed. The model behaves better when context management is preemptive rather than reactive.

**7. Test on real long trajectories.** Synthetic short tests don't catch the failure modes this module addresses. Use representative production-scale runs as your evaluation set.

**8. Treat note-store hygiene as its own concern.** Notes themselves can grow; categorization, retention policies, and periodic pruning keep them useful. Module 9 expands this when notes persist across runs.

The discipline this enforces: long-running agents become reliable rather than degrading toward failure as trajectories grow. The 47-turn session that broke for Sam becomes the 100-turn session that doesn't, because the system is actively managing what the model sees.

---

## Recap: Module 8 in eight bullets

::: bullet-points
- Three techniques for managing context across long trajectories: tool result clearing (drop stale results), compaction (summarize and continue), structured note-taking (write outside the context window). They compose; production agents use all three.
- Tool result clearing is the lightest touch and the highest-impact for most agents. Anthropic's `clear_tool_uses_20250919` strategy automates this; the from-scratch version is straightforward.
- Compaction summarizes message history into a high-fidelity compressed form. Server-side compaction (`compact_20260112`) is now the recommended strategy on the Anthropic Developer Platform; client-side is still useful when you need control over what's preserved.
- The art of compaction is in the prompt. Start by maximizing recall; iterate to improve precision. Test on real long traces. Constraints and other "cannot be lost" content shouldn't go through compaction at all.
- Structured note-taking moves critical state outside the context window. The Anthropic memory tool provides a generic file-system surface; custom note-taking tools with categories give better structure for production agents.
- Anthropic's own evaluations: memory tool + context editing = 39% improvement over baseline; context editing alone = 29%; in 100-turn web search, context editing reduced token consumption by 84% while enabling completion of workflows that would have failed.
- Context anxiety — the model wrapping up prematurely as it senses limits — happens when context management runs reactively. Set thresholds preemptively (50% for tool clearing, 75% for compaction); don't run near the limit.
- Diagnose before applying. Short trajectories don't need this module; long ones need different combinations. Audit a real long run; calibrate per agent.
:::

---

::: sam-arc
**Sam, after rebuilding the Q&A agent.**

Sam shipped the new Q&A agent on Monday morning. Tool result clearing turned on. Compaction configured. `add_note` and `list_notes` tools wired in with the categorical schema. The system prompt updated to teach the agent when to use each.

The same AE from Friday tried the same Mercer-prep session. Forty-seven turns again. Brief came back complete. Leadership question answered correctly. No repetition. The AE pinged Sam: *"It's working. Did you also make it faster? Feels faster."*

Sam looked at the trace. Yes, faster — by a lot. The peak context size was 90K instead of 220K. Each turn's input was a quarter the size. Latency had dropped roughly proportionally.

The Tuesday standup, the platform team lead asked Sam to walk the team through the changes. Sam pulled up the trajectory plots side by side. The Friday-morning broken run climbed steeply for 40 turns, then plateaued near the limit while quality degraded. The Monday run climbed gently, leveled off, never approached the limit.

Sam said: *"Module 7 was about being deliberate with what goes into context. Module 8 is about being deliberate with what stays there. Same discipline, different time horizon."*

The team lead added one more agent to Sam's audit list — the customer-success bot, which had been mysteriously losing track of customer constraints in long sessions. Sam already knew the diagnosis without looking.

Sam's arc this module: **trajectory thinking.** The hardened loop in Module 2 reasoned about a single turn. M3 reasoned about workflows. M5 reasoned about systems. M6 about cost. M7 about context per call. M8 about context across calls. Each module adds another time horizon. Production agents have to be right at all of them; the disciplines stack.
:::

---

## What's next

Module 9 takes the structured note-taking from this module and extends it across runs. So far, "memory" has meant "within a single trajectory." Module 9 covers persistence across separate sessions — a customer-success agent that remembers a customer between conversations, a research agent that builds up a knowledge base over weeks, a coding agent that retains project conventions across days.

The techniques: scoped memory (per user, per project, per topic), retrieval patterns from memory (when to load what), the lifecycle of a memory (create, update, archive, prune), and the privacy implications of agents that remember.

After Module 9, the M7-M8-M9 arc closes: what to put in (M7), what to take out (M8), what to remember (M9). Sam's deal-research and Q&A agents will have everything they need to handle long sessions, multiple sessions, and cross-session context — without the bloat that broke them at the start of M7.

For now: take the audit exercise from Section 8. Find a long trajectory you've shipped. Plot the token curve. Find the bloat. Decide which technique applies. Ship the fix. Watch the trajectory go from "degrades gracefully toward failure" to "stays sharp across hundreds of turns." That's the production capability this module unlocks.
