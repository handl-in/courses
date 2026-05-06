# Module 2 — The Agent Loop, Hardened

::: chapter-opener
<div class="module-num">MODULE 2</div>
<div class="module-title">The Agent Loop, Hardened</div>
<div class="subtitle">The six-line version is easy. Production is where it gets interesting.<br>This is the loop you'd actually ship.</div>
<div class="pages">~38 pages · the loop, with operational discipline</div>
:::

::: hook
Sam pushed the agent to staging on Tuesday. By Thursday morning, three things had happened:

One, an agent run had spent forty-three dollars trying to read a single PDF that 404'd on every fetch. The model never got bored. It just kept calling `read_url` against the same dead URL — fifty-seven times — varying the parameters slightly each pass, hoping a different incantation would make the document materialize.

Two, a different run had silently completed in eleven seconds with no output. No error in the logs. The model had returned `stop_reason: "max_tokens"` mid-thought, the loop had treated it as completion, and the response was an empty string. The user got `null`. Customer support filed a ticket.

Three, a run had succeeded — but had taken eight minutes instead of the expected forty seconds. Sam looked at the trace and found a tool that returned 80,000 tokens of XML when asked for a customer record. The model had dutifully read the whole thing, three times, before answering.

The loop in Module 1 was correct. It was also not what you'd ship.
:::

---

## What this module is

The minimal agent loop from Module 1 was about ten lines of code. It demonstrated the *shape*: call the model, check the stop reason, execute tool calls, feed results back, repeat. That shape is durable. Steve Kinney looked at six different agent frameworks in early 2026 — Claude Agent SDK, OpenAI Agents SDK, Cursor, the Vercel AI SDK, LangGraph, smolagents — and concluded "not similar. The same. A while loop that calls an LLM, checks if the response contains tool calls, executes them if it does, and stops if it doesn't. That's the whole thing."

He's right. And he's also right about what comes next: *"The 6-line version is easy. The production-hardened version — with context compaction, loop detection, cost budgets, and graceful termination — is where things get interesting."*

This module is the production-hardened version.

By the end you'll be able to:

- Implement an agent loop that handles every stop reason correctly (not just `tool_use` and `end_turn`)
- Add the four operational defenses that prevent runaway behavior: turn caps, token budgets, cost ceilings, loop detection
- Handle tool failures without poisoning context, and recover from API errors without losing state
- Distinguish the Anthropic Client SDK (where you implement the loop) from the Agent SDK (where Claude handles the loop), and know when each makes sense
- Read your own agent's traces well enough to debug them at 2am

We'll write the loop in stages — first the minimal version, then we'll add each defense one at a time, explaining what each one prevents. By the end of the module the agent is something you'd actually deploy.

::: pullquote
The loop is the easy part. Making it reliable is the whole job.<br>— Steve Kinney, "The Anatomy of an Agent Loop"
:::

---

## Section 1 — The Two SDKs (Brief)

Before we write any code, a quick orientation that affects the whole module.

Anthropic ships two Python SDKs that both call themselves "the SDK," and the distinction matters.

**The Anthropic Client SDK** (`anthropic` on PyPI). The basic one. You make API calls. When the model wants to use a tool, you receive a `tool_use` block, execute the tool yourself, and send the result back as a `tool_result`. *You implement the loop.* This is what we'll teach in this module.

**The Claude Agent SDK** (`claude-agent-sdk` on PyPI, formerly Claude Code SDK). The higher-level one. It runs on top of the Claude Code CLI binary. You hand it a prompt; it runs the agent loop for you, including built-in tools (filesystem, bash, web), MCP integration, hooks, and permission management. *Claude handles the loop.*

The Agent SDK is genuinely production-grade — it's the engine inside Claude Code. For many tasks, you should just use it and stop reading this module.

We're teaching the Client SDK path anyway, and not because it's better. We're teaching it because:

1. **You need to understand what's inside the loop to debug what's outside it.** Every framework — including the Agent SDK — runs roughly the loop we're about to write. When something goes wrong in production, the trace is going to show tool calls and stop reasons, and you need to recognize what each one means.
2. **Custom agents have custom loops.** The Agent SDK is opinionated about tools (filesystem, bash, web) and execution model (CLI subprocess). If you're building, say, a customer-support agent that should *only* call your API and *never* touch the filesystem, the Client SDK gives you exact control. The Agent SDK gives you most of what you want plus a bunch you don't.
3. **Multi-agent systems often roll their own loops.** Module 14 builds supervisor-worker patterns where each level has its own loop with its own constraints. You can do that with the Agent SDK, but you understand it better if you've built the loop from scratch first.

When the module ends, you'll be able to make an informed choice between the two for your next project. For now: Client SDK. Loop in your hands.

::: nodumbq
**Q: Wait, the field has a "Managed Agents" thing now too?**

Yes, as of April 2026 — Managed Agents is Anthropic's hosted agent runtime. They run the loop *and* the sandbox on their infrastructure; you make REST API calls to a session. Different again from both SDKs. We don't cover it in depth in this book because it's still in beta and the patterns shift quickly. Once you understand the Client SDK loop, the others are straightforward.

**Q: Is one of these going to win?**

The Client SDK and Agent SDK serve different needs and will likely coexist. The Agent SDK is the right default for most teams, especially for coding-adjacent agents. The Client SDK is the right choice when you need precise control or are building something the Agent SDK isn't shaped for. Treat them as complementary, not competing.
:::

---

## Section 2 — The Minimal Loop, Properly

Let's start with a slightly less minimal loop than Module 1 had — still small, but with names and types that we'll keep building on.

```python
from anthropic import Anthropic
from anthropic.types import Message, MessageParam

client = Anthropic()


def run_agent_minimal(
    user_query: str,
    tools: list[dict],
    tool_handlers: dict[str, callable],
    model: str = "claude-sonnet-4-6",
    max_turns: int = 20,
) -> str:
    messages: list[MessageParam] = [{"role": "user", "content": user_query}]

    for turn in range(max_turns):
        response: Message = client.messages.create(
            model=model,
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )

        # Append the assistant's response (including any tool_use blocks) to messages
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return _extract_text(response)

        if response.stop_reason == "tool_use":
            tool_results = _execute_tools(response, tool_handlers)
            messages.append({"role": "user", "content": tool_results})
            continue

        # Other stop_reason values handled in Section 3
        raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason}")

    raise RuntimeError(f"Agent exhausted {max_turns} turns")


def _extract_text(response: Message) -> str:
    return "".join(
        block.text for block in response.content if block.type == "text"
    )


def _execute_tools(response: Message, handlers: dict[str, callable]) -> list[dict]:
    results = []
    for block in response.content:
        if block.type == "tool_use":
            handler = handlers[block.name]
            result = handler(**block.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result),
            })
    return results
```

A few things to note about this version that the Module 1 sketch glossed:

**The whole assistant response gets appended to messages, not just text.** When the model emits a `tool_use` block, that block is part of its response. The next turn needs to see "I called this tool" to make sense of "here's the result." If you only append text and drop tool_use blocks, the conversation no longer parses and the next call will fail.

**Tool results are wrapped in a user message.** This is the API's protocol: the assistant emits tool_use blocks, the user (your code, on behalf of the user) emits matching tool_result blocks. Same role channel, alternating.

**Every tool_use block needs a matching tool_result.** If the model called three tools in one turn, you owe three tool_result blocks before the next call. Forget one and you'll get a 400.

**Parallel tool calls happen in one turn.** When the model emits multiple `tool_use` blocks in a single response, you can — and should — execute them concurrently. The function above does them in order; we'll fix this in Section 6 with `asyncio.gather`.

This is still a fragile loop. It explodes on any stop reason that isn't `end_turn` or `tool_use`. It has no budget. It can't recover from a tool that throws. We're going to fix all of that. But this is the skeleton; everything from here is layering on top.

::: gotcha
The single most common mistake in hand-rolled agent loops is not appending the assistant's full response (including tool_use blocks) to messages before the next turn. The symptom is a `400` from the API a few turns in, complaining about unexpected message structure. The cause is almost always that the dev appended only the text content. The full `response.content` list — including tool_use blocks — is what goes back into messages.
:::

---

## Section 3 — Stop Reasons: All of Them

`stop_reason` tells you why the model finished its turn. The minimal loop handled two values: `end_turn` (model is done) and `tool_use` (model wants to call tools). There are more, and you have to handle them or your agent will misbehave in subtle ways.

Here are all the values you'll see in production:

**`"end_turn"`** — the model finished naturally. It said its piece and stopped. This is the success case. Extract the text and return.

**`"tool_use"`** — the model called one or more tools and is waiting for results. Execute the tools, append `tool_result` blocks, continue the loop.

**`"max_tokens"`** — the model hit the `max_tokens` limit you set on the request and was cut off mid-generation. **This is not done.** The text it produced is incomplete. The remembered failure from the cold open — the agent that returned eleven seconds of nothing — was this case, mishandled. Treat as either an error or a "continue" depending on your design. Often the fix is to raise `max_tokens` and retry; sometimes it's to break the task into smaller chunks.

**`"stop_sequence"`** — the model hit a `stop_sequences` value you provided. Treat similarly to `end_turn` for most uses; the specific stop sequence tells you why it stopped if you care.

**`"refusal"`** — the model refused to respond, often for safety reasons. The output text will explain. Don't retry the same prompt; surface the refusal to the user or upstream caller.

**`"pause_turn"`** — the model is pausing within an extended thinking session, typically to let you handle something asynchronously and resume. Most agents won't see this; if you're using extended thinking with interleaved tool calls and async patterns, handle it like `tool_use` (process and continue).

**`null` (or anything else)** — protocol violation or new value Anthropic added. Don't pretend you handled it; raise loudly so it shows up in your traces.

Here's the loop with all stop reasons handled:

```python
def run_agent_with_stop_reasons(
    user_query: str,
    tools: list[dict],
    tool_handlers: dict[str, callable],
    model: str = "claude-sonnet-4-6",
    max_turns: int = 20,
    max_tokens_per_turn: int = 4096,
) -> str:
    messages: list[MessageParam] = [{"role": "user", "content": user_query}]

    for turn in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens_per_turn,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        match response.stop_reason:
            case "end_turn" | "stop_sequence":
                return _extract_text(response)

            case "tool_use":
                tool_results = _execute_tools(response, tool_handlers)
                messages.append({"role": "user", "content": tool_results})
                continue

            case "max_tokens":
                # Output truncated. Could:
                # (a) Return what we have with a flag (often wrong — output is incomplete)
                # (b) Increase max_tokens and retry the turn
                # (c) Raise an error and let the caller decide
                # We pick (c) for clarity; you choose for your application.
                partial = _extract_text(response)
                raise OutputTruncated(
                    f"Model output truncated at {max_tokens_per_turn} tokens",
                    partial_output=partial,
                    turn=turn,
                )

            case "refusal":
                refusal_text = _extract_text(response)
                raise ModelRefusal(refusal_text, turn=turn)

            case "pause_turn":
                # Extended-thinking pause; treat like tool_use continuation
                continue

            case _:
                raise UnexpectedStopReason(
                    f"Unhandled stop_reason: {response.stop_reason!r}",
                    response=response,
                    turn=turn,
                )

    raise TurnLimitExceeded(f"Agent exhausted {max_turns} turns")


class OutputTruncated(Exception):
    def __init__(self, message, partial_output, turn):
        super().__init__(message)
        self.partial_output = partial_output
        self.turn = turn


class ModelRefusal(Exception):
    def __init__(self, message, turn):
        super().__init__(message)
        self.turn = turn


class UnexpectedStopReason(Exception):
    def __init__(self, message, response, turn):
        super().__init__(message)
        self.response = response
        self.turn = turn


class TurnLimitExceeded(Exception):
    pass
```

Three things this version gets right that the minimal version didn't:

1. **`max_tokens` is no longer a silent success.** You get a typed exception with the partial output, so the caller can decide whether to retry with a larger budget or fail loudly.
2. **`refusal` is named and visible.** A refusal usually means your prompt is asking for something the model won't do; the right response is human review, not a retry loop.
3. **Unhandled stop reasons explode rather than hide.** When Anthropic adds a new stop reason — and they will — your agent fails fast with a clear error, not silently or weirdly.

::: brain
The cold open's "eleven seconds of nothing" agent failed because `max_tokens` was treated like `end_turn`. Trace through it: the model started generating, hit the budget, returned partial text with `stop_reason: "max_tokens"`, the loop's `if stop_reason == "end_turn": return text` happened to match because the dev had used `else: return text` as a default. What's the cheapest defense? (Answer: explicit handling of every known stop reason, and a default that raises. The cost is one `match` statement; the savings is every silent-truncation bug for the lifetime of the agent.)
:::

---

## Section 4 — Tool Failures and Context Hygiene

Tools fail. The API a tool wraps is down, returns an unexpected schema, returns way too much data, raises a Python exception. Your loop has to keep going — but it also has to be honest about what happened.

There are four ways a tool can fail, and they want different handling:

**Type 1: The tool raised a Python exception.** Network error, parse error, KeyError. The model never sees a result; instead, you decide what to send.

**Type 2: The tool returned an error object.** The API said `{"error": "user_not_found"}`. The tool didn't crash; the operation just didn't succeed.

**Type 3: The tool returned valid but enormous output.** The 80,000-token customer record from the cold open. Tool succeeded; result will eat the context.

**Type 4: The tool succeeded with usable but stale or wrong data.** Cache returned yesterday's price. Tool can't know it's wrong; you can't catch this here.

For Types 1 and 2, the right answer is roughly the same: surface the failure to the model in the tool_result, with `is_error: true`. The model can see the failure, reason about it, and try something else (or give up gracefully).

```python
def _execute_tools_safely(
    response: Message,
    handlers: dict[str, callable],
    max_result_chars: int = 20_000,
) -> list[dict]:
    results = []
    for block in response.content:
        if block.type != "tool_use":
            continue

        handler = handlers.get(block.name)
        if handler is None:
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "is_error": True,
                "content": f"Unknown tool: {block.name}",
            })
            continue

        try:
            raw_result = handler(**block.input)
            content = _serialize_result(raw_result)
        except Exception as exc:
            # Type 1: tool raised. Tell the model.
            content = f"Tool {block.name} raised {type(exc).__name__}: {exc}"
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "is_error": True,
                "content": content,
            })
            continue

        # Type 3: result too big. Truncate and warn.
        if len(content) > max_result_chars:
            truncated = content[: max_result_chars]
            content = (
                f"{truncated}\n\n"
                f"[result truncated: {len(content)} chars total, "
                f"showing first {max_result_chars}]"
            )

        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": content,
        })
    return results


def _serialize_result(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        import json
        return json.dumps(result, indent=2, default=str)
    return str(result)
```

The choices in this version, made deliberately:

**Tool exceptions become tool_result with `is_error: true`, not Python exceptions.** The model can see "the tool failed because X" and decide what to do. If you let the exception propagate, you've thrown away the agent's ability to recover. The model often *can* recover — it tries a different argument, or admits the failure and asks the user, or moves on to a different tool.

**Truncation is explicit, not silent.** If the result is too big, we show the model a prefix and tell it the rest was cut. Without the marker, the model might assume it has the complete data and synthesize confidently from a partial picture.

**Unknown tool names produce a tool_result, not an exception.** If your tool registry doesn't have the tool the model called for, that's still recoverable — the model can pick a different tool next turn. But if it raises, you've crashed the loop.

::: postmortem
**The Tool Result That Broke the Compactor**

A team's agent was working great for a week. Then on Friday afternoon it started timing out. Long traces. Tool calls returning, but the agent never finishing.

Investigation: a customer record tool was returning ~5KB normally. For one specific customer (with 200 line items in their order history) it returned 320KB of JSON. The agent dutifully read all of it. The next turn's input was now huge. The model's response was slow. The model called more tools. The next turn was huger. By the fifth turn, the agent's context had grown past 100K tokens and inference was crawling.

The fix was the truncation block above: cap tool results at 20K characters, with an explicit marker that the result was truncated. The model, seeing the marker, learned to make narrower queries (filter by date, paginate) instead of swallowing the whole record.

The lesson: **tool result size is part of your tool's contract**. A tool that can return arbitrary amounts of data is a tool that can lock up your agent. Cap aggressively at the loop level even when you're confident the tool is well-behaved — you'll eventually hit the case you weren't confident about.
:::

::: gotcha
A tempting design is to filter tool results based on what looks "relevant." Don't. The truncation marker is honest; selective filtering is silent context corruption. If the tool is returning too much, fix the tool's interface (add filters, paginate, summarize at the tool layer) — don't let the loop secretly hide data from the model.
:::

---

## Section 5 — Budgets: Turns, Tokens, and Dollars

Three budgets, layered. Each catches a different runaway pattern.

**Turn budget.** The simplest. A counter. Caps the number of times the loop runs. Catches infinite-loop bugs but doesn't catch a single very expensive turn.

**Token budget.** Tracks cumulative input and output tokens across the whole session. Catches the case where individual turns are reasonable but their sum isn't.

**Dollar budget.** Tracks cumulative cost in USD, computed from the session's token totals against current model pricing. Catches the case where token totals look fine but the model upgrades you ran on were expensive (Opus instead of Sonnet, etc.).

In practice you want all three. They're cheap to track. Each prevents a different incident.

```python
from dataclasses import dataclass, field


# Pricing as of May 2026; verify against Anthropic's current pricing page.
# Numbers are USD per million tokens, (input, output).
MODEL_PRICING = {
    "claude-opus-4-7":   (5.00, 25.00),
    "claude-opus-4-6":   (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5":  (1.00,  5.00),
}


@dataclass
class Budget:
    max_turns: int = 30
    max_input_tokens: int = 500_000
    max_output_tokens: int = 100_000
    max_cost_usd: float = 5.00

    turns_used: int = 0
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    cost_used_usd: float = 0.0

    def record_turn(self, response: Message) -> None:
        self.turns_used += 1
        usage = response.usage
        self.input_tokens_used += usage.input_tokens
        self.output_tokens_used += usage.output_tokens
        rates = MODEL_PRICING.get(response.model)
        if rates is None:
            # Unknown model: don't crash, but flag it.
            return
        in_rate, out_rate = rates
        self.cost_used_usd += (
            usage.input_tokens / 1_000_000 * in_rate
            + usage.output_tokens / 1_000_000 * out_rate
        )

    def check(self) -> None:
        if self.turns_used >= self.max_turns:
            raise BudgetExceeded("turns", self.turns_used, self.max_turns)
        if self.input_tokens_used >= self.max_input_tokens:
            raise BudgetExceeded("input_tokens", self.input_tokens_used, self.max_input_tokens)
        if self.output_tokens_used >= self.max_output_tokens:
            raise BudgetExceeded("output_tokens", self.output_tokens_used, self.max_output_tokens)
        if self.cost_used_usd >= self.max_cost_usd:
            raise BudgetExceeded("cost_usd", self.cost_used_usd, self.max_cost_usd)


class BudgetExceeded(Exception):
    def __init__(self, dimension: str, used, limit):
        super().__init__(f"Budget exceeded on {dimension}: {used} >= {limit}")
        self.dimension = dimension
        self.used = used
        self.limit = limit
```

The budget object is updated after every API response and checked before every API call. Three dimensions plus turns means four ways the agent can stop early — each with a clear name in your traces, so when you debug "why did the agent stop?" you can answer specifically: it ran out of turns, or burned through the cost ceiling, or generated too much output.

The choice of caps matters more than people realize. Your budget should reflect what *you*, the operator, are willing to pay for *one* user request. For a customer-support agent that's expected to resolve a ticket, $0.50 might be the right ceiling — past that, you're spending more on the LLM than the customer's question is worth. For a deep research agent that's expected to take five minutes and produce a 2000-word report, $5 is reasonable.

If you can't articulate the dollar value of one successful run, you can't size the budget. And if you can't size the budget, you don't know what "too expensive" means until you're paying it.

::: nodumbq
**Q: What if my budget is too tight and the agent stops mid-task?**

That's the right outcome — better than infinite cost. The agent should return *what it has so far* (partial results, or a clear "I couldn't complete this within the budget"), and your application should decide whether to escalate to a human, retry with a bigger budget, or surface the partial result to the user. We cover graceful degradation in Section 9.

**Q: How do I figure out a reasonable cost cap if I haven't shipped yet?**

Run your agent on 50-100 representative tasks with a generous cap (say, $10) and look at the cost distribution. Pick a cap at the 95th or 99th percentile; that's the cap that catches outliers but lets normal traffic through. Then watch production for a week — if you're hitting the cap on legitimate tasks, raise it; if you're never hitting it, lower it. Caps are tuneable; don't agonize over the first number.
:::

---

## Section 6 — Loop Detection

A budget catches *eventual* runaway. Loop detection catches *immediate* runaway — the agent calling the same tool with the same arguments over and over because it's stuck.

The cold open's $43 PDF read was this. Same URL, fifty-seven times. Each call cost a fraction of a cent in tokens, but they added up. A turn budget of 100 wouldn't have caught it in time. A cost cap of $5 would've, but only after burning $5.

A simple loop detector that runs on every tool call:

```python
import hashlib
from collections import Counter


@dataclass
class LoopDetector:
    # If the same (tool_name, args_hash) shows up this many times
    # in the recent history window, we declare a loop.
    repeat_threshold: int = 3
    history_window: int = 8
    history: list[tuple[str, str]] = field(default_factory=list)

    def record(self, tool_name: str, tool_input: dict) -> None:
        sig = (tool_name, _stable_hash(tool_input))
        self.history.append(sig)
        if len(self.history) > self.history_window:
            self.history = self.history[-self.history_window:]

    def in_loop(self) -> tuple[bool, tuple[str, str] | None]:
        if not self.history:
            return False, None
        counts = Counter(self.history)
        most_common, count = counts.most_common(1)[0]
        if count >= self.repeat_threshold:
            return True, most_common
        return False, None


def _stable_hash(data: dict) -> str:
    import json
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

Wired into the loop:

```python
def _execute_tools_with_loop_detection(
    response, handlers, detector: LoopDetector, max_result_chars=20_000,
):
    results = []
    for block in response.content:
        if block.type != "tool_use":
            continue
        detector.record(block.name, block.input)
        in_loop, signature = detector.in_loop()
        if in_loop:
            tool_name, _ = signature
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "is_error": True,
                "content": (
                    f"Loop detected: tool {tool_name} called with the same arguments "
                    f"{detector.repeat_threshold}+ times in {detector.history_window} turns. "
                    f"Try a different approach, or report inability to proceed."
                ),
            })
            continue
        # ... normal execution path ...
    return results
```

Notice what we do when a loop is detected: we don't crash the agent. We feed the loop-detection message back to the model as a tool_result with `is_error: true`. The model sees its own loop, named explicitly, and gets a chance to try something else.

This is more useful than crashing. Most loops happen because the model is fixated on a path that isn't working — wrong URL, wrong query, wrong argument. With explicit feedback, it usually changes course. The 4% of cases where it doesn't are caught by the budget instead.

What counts as "the same call" matters. Two calls with `{"url": "x"}` and `{"url": "x", "timeout": 30}` are not exact matches but probably count as the same loop. We hash the canonical-JSON representation, which catches dict-key-order differences but does treat extra fields as different. For most agents this is fine; for picky cases, you can canonicalize more aggressively (drop optional fields, normalize whitespace, etc.).

::: brain
A loop detector with `repeat_threshold=3` catches three identical calls in eight turns. What if your agent legitimately needs to call the same tool four times — say, polling an async job? (Answer: either tighten the loop detector's idea of "same call" so polling doesn't count as a loop — different call IDs, slightly different timestamps — or design the tool itself to handle polling internally so the agent doesn't see four calls. The second option is almost always better. Polling is a tool concern, not an agent concern.)
:::

---

## Section 7 — Concurrent Tool Calls and Async

The model can emit multiple tool_use blocks in a single turn. The minimal loop executes them in order. This wastes wall-clock time when the tools are independent — and most are, since the model wouldn't have called them in parallel if it thought they depended on each other.

The async version:

```python
import asyncio
from anthropic import AsyncAnthropic

async_client = AsyncAnthropic()


async def _execute_tools_concurrent(
    response: Message,
    handlers: dict[str, callable],
    max_result_chars: int = 20_000,
) -> list[dict]:
    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

    async def run_one(block):
        try:
            handler = handlers[block.name]
            if asyncio.iscoroutinefunction(handler):
                raw = await handler(**block.input)
            else:
                # Run sync handlers in a thread to avoid blocking the loop
                raw = await asyncio.to_thread(handler, **block.input)
            content = _serialize_result(raw)
            if len(content) > max_result_chars:
                content = content[:max_result_chars] + f"\n\n[truncated: {len(content)} total]"
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            }
        except KeyError:
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "is_error": True,
                "content": f"Unknown tool: {block.name}",
            }
        except Exception as exc:
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "is_error": True,
                "content": f"Tool {block.name} raised {type(exc).__name__}: {exc}",
            }

    return await asyncio.gather(*[run_one(b) for b in tool_use_blocks])
```

Two things worth being explicit about:

**`asyncio.to_thread` for sync handlers.** A lot of real tools are sync — they wrap a sync HTTP client, a sync database driver, a CPU-bound transformation. If you `await` them naively in an async loop, you block the event loop. `to_thread` lets you run sync tools concurrently without rewriting them.

**Per-tool exception handling, not per-batch.** If three tools run in parallel and one crashes, the other two should still return results. Wrapping each in its own try/except (and letting `asyncio.gather` collect them all) is the right pattern. The default `gather` behavior — where one exception aborts the whole gather — is what we *don't* want here.

The full async loop looks essentially the same as the sync one, with `await` sprinkled where you'd expect:

```python
async def run_agent_async(
    user_query: str,
    tools: list[dict],
    tool_handlers: dict[str, callable],
    *,
    model: str = "claude-sonnet-4-6",
    budget: Budget | None = None,
    detector: LoopDetector | None = None,
    max_tokens_per_turn: int = 4096,
) -> AgentResult:
    budget = budget or Budget()
    detector = detector or LoopDetector()
    messages = [{"role": "user", "content": user_query}]

    while True:
        budget.check()
        response = await async_client.messages.create(
            model=model,
            max_tokens=max_tokens_per_turn,
            tools=tools,
            messages=messages,
        )
        budget.record_turn(response)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason in ("end_turn", "stop_sequence"):
            return AgentResult(
                output=_extract_text(response),
                budget=budget,
                trace_messages=messages,
                terminated_by="end_turn",
            )

        if response.stop_reason == "tool_use":
            for block in response.content:
                if block.type == "tool_use":
                    detector.record(block.name, block.input)
            in_loop, sig = detector.in_loop()
            if in_loop:
                # Inject loop notice into next user turn instead of executing
                messages.append({
                    "role": "user",
                    "content": _loop_notice(response, sig),
                })
                continue
            tool_results = await _execute_tools_concurrent(response, tool_handlers)
            messages.append({"role": "user", "content": tool_results})
            continue

        if response.stop_reason == "max_tokens":
            return AgentResult(
                output=_extract_text(response),
                budget=budget,
                trace_messages=messages,
                terminated_by="max_tokens",
                partial=True,
            )

        if response.stop_reason == "refusal":
            return AgentResult(
                output=_extract_text(response),
                budget=budget,
                trace_messages=messages,
                terminated_by="refusal",
                partial=True,
            )

        raise UnexpectedStopReason(response.stop_reason, response=response)


@dataclass
class AgentResult:
    output: str
    budget: Budget
    trace_messages: list
    terminated_by: str  # "end_turn" | "max_tokens" | "refusal" | "budget" | "turns"
    partial: bool = False
```

Now we have a loop that:

- handles every stop reason
- caps turns, tokens, and cost
- detects loops and gives the model a chance to recover
- runs parallel tool calls concurrently
- returns a structured result with enough metadata to debug

That's most of what production needs. Three things left: API errors, observability, and graceful termination. Sections 8 through 10.

---

## Section 8 — API Errors and Retries

The Anthropic API can return errors. Some are retryable, some aren't. The Python SDK handles a lot of this for you (it retries 5xx and rate-limit responses with backoff), but you should know what's happening and override defaults when needed.

The errors you'll see in production:

**`anthropic.RateLimitError` (HTTP 429).** You're sending requests too fast. The SDK retries with exponential backoff by default. If your traffic is bursty, the default behavior is usually fine; if you're consistently hitting limits, you need higher tier limits or to throttle upstream.

**`anthropic.APIStatusError` with 5xx codes.** Server-side issue. Retry with backoff. The SDK does this by default (default retries: 2, configurable).

**`anthropic.APIStatusError` with 4xx codes (other than 429).** Your request is wrong. Don't retry — fix the request. Common causes: invalid tool schema, message structure broken (forgot a tool_result), context window exceeded.

**`anthropic.APIConnectionError`.** Network problem reaching the API. Retry with backoff.

**`anthropic.APITimeoutError`.** The request took too long and the SDK gave up. Default timeout is 10 minutes per request, which is generous; you can lower it.

The pragma: configure retries at SDK construction, then handle terminal errors at your loop level.

```python
import anthropic

# Two retries on retryable errors, ten-minute timeout per call.
async_client = AsyncAnthropic(
    max_retries=2,
    timeout=600.0,
)


async def call_with_diagnosis(client, **kwargs):
    """Wrap messages.create so terminal errors carry useful context."""
    try:
        return await client.messages.create(**kwargs)
    except anthropic.APIStatusError as e:
        # 4xx (non-429) reach here after the SDK gave up.
        # 5xx and 429 reach here only if retries exhausted.
        raise AgentAPIError(
            f"API error {e.status_code}: {e.message}",
            status_code=e.status_code,
            request_id=getattr(e, "request_id", None),
        ) from e
    except anthropic.APIConnectionError as e:
        raise AgentAPIError(
            f"Connection error after retries: {e}",
            status_code=None,
            request_id=None,
        ) from e


class AgentAPIError(Exception):
    def __init__(self, message, status_code, request_id):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
```

A subtle thing about retries inside an agent loop: if the SDK retries a request that already succeeded once on a tool that has side effects, you can get double execution. This is rare with `messages.create` (the API is idempotent on the model's side; you sent inputs and you get outputs), but the *tools* you call from within the loop can be non-idempotent. If your retry strategy upstream re-runs an entire agent call, you can re-send emails, re-charge cards, re-create resources.

The defense for this lives in tool design, not loop design — Module 4 covers idempotency keys and dry-run patterns. For now: be aware that retrying a whole agent run is *not safe by default*, and design your tools (or your retry policy) accordingly.

::: gotcha
The Anthropic SDK's default `max_retries=2` is good for most cases. Setting it higher (`max_retries=10`) sounds defensive but is dangerous — if the API is genuinely down and you're in an agent loop, you'll burn turns and budget on retries that won't succeed. Trust the SDK's default; if retries fail, fail upward fast.
:::

---

## Section 9 — Graceful Termination and Partial Results

When the loop terminates for *any* reason that isn't `end_turn` — budget exhausted, refusal, max-tokens, turn cap, API error — you have a partial result. The right behavior is to surface what you have, not throw it away.

The `AgentResult` dataclass we built handles this — every termination path produces an `AgentResult` with `terminated_by` set and `partial=True` when applicable. Three call sites that benefit:

**The user-facing UI.** Show the partial result with a label: "I couldn't finish this in the time/budget I had. Here's what I gathered so far:" The user can read it, judge usefulness, and ask a more focused follow-up.

**Upstream retry logic.** If the agent terminated by budget, retry with a higher budget *only if the partial result is empty or unhelpful*. If it produced something useful, ship it.

**Eval and observability.** Partial-result rate is a metric you'll want. If 10% of runs hit `terminated_by="budget"`, you have a sizing problem; if 10% hit `terminated_by="refusal"`, you have a prompt problem. Lumping all failures into "didn't complete" hides the diagnosis.

Here's the full loop, returning `AgentResult` from every exit path, including the budget-exhausted case:

```python
async def run_agent_complete(
    user_query: str,
    tools: list[dict],
    tool_handlers: dict[str, callable],
    *,
    model: str = "claude-sonnet-4-6",
    budget: Budget | None = None,
    detector: LoopDetector | None = None,
    max_tokens_per_turn: int = 4096,
) -> AgentResult:
    budget = budget or Budget()
    detector = detector or LoopDetector()
    messages = [{"role": "user", "content": user_query}]

    while True:
        try:
            budget.check()
        except BudgetExceeded as e:
            return AgentResult(
                output=_partial_output_so_far(messages),
                budget=budget,
                trace_messages=messages,
                terminated_by=f"budget:{e.dimension}",
                partial=True,
            )

        try:
            response = await call_with_diagnosis(
                async_client,
                model=model,
                max_tokens=max_tokens_per_turn,
                tools=tools,
                messages=messages,
            )
        except AgentAPIError as e:
            return AgentResult(
                output=_partial_output_so_far(messages),
                budget=budget,
                trace_messages=messages,
                terminated_by=f"api_error:{e.status_code}",
                partial=True,
            )

        budget.record_turn(response)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason in ("end_turn", "stop_sequence"):
            return AgentResult(
                output=_extract_text(response),
                budget=budget,
                trace_messages=messages,
                terminated_by="end_turn",
            )

        if response.stop_reason == "tool_use":
            for block in response.content:
                if block.type == "tool_use":
                    detector.record(block.name, block.input)
            in_loop, sig = detector.in_loop()
            if in_loop:
                messages.append({"role": "user", "content": _loop_notice(response, sig)})
                continue
            tool_results = await _execute_tools_concurrent(response, tool_handlers)
            messages.append({"role": "user", "content": tool_results})
            continue

        # max_tokens, refusal, anything else: terminal.
        return AgentResult(
            output=_extract_text(response),
            budget=budget,
            trace_messages=messages,
            terminated_by=f"stop_reason:{response.stop_reason}",
            partial=True,
        )


def _partial_output_so_far(messages: list) -> str:
    """Best-effort recovery of the agent's last text output."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg["content"]
        if isinstance(content, list):
            text_parts = [b.text for b in content if hasattr(b, "type") and b.type == "text"]
            if text_parts:
                return "\n".join(text_parts)
    return ""
```

This loop is almost done. One more layer: knowing what it actually did.

::: pullquote
A budget that crashes the agent silently is not a defense — it's a different kind of failure. Surface the partial result; let the caller decide whether it's worth shipping.
:::

---

## Section 10 — Observability: Just Enough for Now

We'll do production observability properly in Module 17 — OpenTelemetry semantic conventions, distributed tracing, the platform comparisons. Here, in Module 2, we install just enough that the agent isn't a black box from day one.

Two things matter at this stage:

1. **Structured logging of every turn.** Each turn produces one log line with: turn number, model, input/output tokens, cumulative cost, stop reason, tool calls (names only — not arguments — to avoid logging PII).

2. **A trace object** that holds the full message history and per-turn metadata, returnable from the loop for inspection in tests and replayable in dev.

```python
import logging
import time

logger = logging.getLogger("agent")


@dataclass
class TurnRecord:
    turn: int
    started_at: float
    duration_s: float
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    stop_reason: str
    tool_calls: list[str]  # names only, no args


@dataclass
class AgentTrace:
    turns: list[TurnRecord] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def add(self, record: TurnRecord) -> None:
        self.turns.append(record)
        logger.info(
            "agent.turn",
            extra={
                "turn": record.turn,
                "model": record.model,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "cost_usd": round(record.cost_usd, 4),
                "stop_reason": record.stop_reason,
                "tool_calls": record.tool_calls,
                "duration_s": round(record.duration_s, 2),
            },
        )

    def summary(self) -> dict:
        total_in = sum(t.input_tokens for t in self.turns)
        total_out = sum(t.output_tokens for t in self.turns)
        total_cost = sum(t.cost_usd for t in self.turns)
        return {
            "total_turns": len(self.turns),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_cost_usd": round(total_cost, 4),
            "wall_clock_s": round(time.time() - self.started_at, 2),
        }
```

Hooked into the loop, the `AgentResult` returns the trace along with the output. Now in tests, dev, and incident response, you can see exactly what the agent did:

```python
result = await run_agent_complete(...)
print(result.trace.summary())
# {'total_turns': 7, 'total_input_tokens': 18420, 'total_output_tokens': 1840,
#  'total_cost_usd': 0.0829, 'wall_clock_s': 12.4}

for turn in result.trace.turns:
    print(f"Turn {turn.turn}: {turn.stop_reason} ({turn.duration_s:.1f}s, "
          f"${turn.cost_usd:.4f}, tools={turn.tool_calls})")
```

This is enough observability to debug on day one. By Module 17 we'll add OpenTelemetry spans, distributed trace context propagation, and platform integrations (Langfuse, LangSmith). For now, structured logs plus a trace object you can introspect in tests is the right scope.

::: nodumbq
**Q: Should I log the full message bodies?**

In dev: yes, it's invaluable for debugging. In production: usually no, especially for user-facing agents. Message bodies often contain PII, customer data, or secrets that ended up in tool inputs. Log metadata (turn number, tool names, token counts, durations); persist full message bodies to a separate observability store with appropriate access controls (Module 17 covers this). The default for Module 2's loop is metadata-only.

**Q: Why not just use `print` while developing?**

Because by week three you'll be running the agent against ten different test cases and want to compare costs across them, and you'll wish you had structured records. Start with structured logging on day one; it's not more work than print statements once you've written it once.
:::

---

## Section 11 — Putting It Together: The Hardened Loop

Here's the full hardened loop, condensed. The pieces from Sections 2-10 assembled into one runnable artifact. In the book this is presented as a downloadable file; in the outline form, it's the consolidated version of everything above.

```python
async def run_hardened_agent(
    user_query: str,
    tools: list[dict],
    tool_handlers: dict[str, callable],
    *,
    model: str = "claude-sonnet-4-6",
    budget: Budget | None = None,
    detector: LoopDetector | None = None,
    trace: AgentTrace | None = None,
    max_tokens_per_turn: int = 4096,
    system_prompt: str | None = None,
) -> AgentResult:
    budget = budget or Budget()
    detector = detector or LoopDetector()
    trace = trace or AgentTrace()
    messages = [{"role": "user", "content": user_query}]

    while True:
        try:
            budget.check()
        except BudgetExceeded as e:
            return AgentResult(
                output=_partial_output_so_far(messages),
                budget=budget, trace=trace, trace_messages=messages,
                terminated_by=f"budget:{e.dimension}", partial=True,
            )

        turn_idx = len(trace.turns) + 1
        started = time.time()
        try:
            kwargs = dict(
                model=model, max_tokens=max_tokens_per_turn,
                tools=tools, messages=messages,
            )
            if system_prompt:
                kwargs["system"] = system_prompt
            response = await call_with_diagnosis(async_client, **kwargs)
        except AgentAPIError as e:
            return AgentResult(
                output=_partial_output_so_far(messages),
                budget=budget, trace=trace, trace_messages=messages,
                terminated_by=f"api_error:{e.status_code}", partial=True,
            )

        budget.record_turn(response)
        tool_call_names = [b.name for b in response.content if b.type == "tool_use"]
        trace.add(TurnRecord(
            turn=turn_idx,
            started_at=started,
            duration_s=time.time() - started,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=_cost_for(response),
            stop_reason=response.stop_reason or "unknown",
            tool_calls=tool_call_names,
        ))
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason in ("end_turn", "stop_sequence"):
            return AgentResult(
                output=_extract_text(response),
                budget=budget, trace=trace, trace_messages=messages,
                terminated_by="end_turn",
            )

        if response.stop_reason == "tool_use":
            for block in response.content:
                if block.type == "tool_use":
                    detector.record(block.name, block.input)
            in_loop, sig = detector.in_loop()
            if in_loop:
                messages.append({"role": "user", "content": _loop_notice(response, sig)})
                continue
            tool_results = await _execute_tools_concurrent(response, tool_handlers)
            messages.append({"role": "user", "content": tool_results})
            continue

        return AgentResult(
            output=_extract_text(response),
            budget=budget, trace=trace, trace_messages=messages,
            terminated_by=f"stop_reason:{response.stop_reason}", partial=True,
        )
```

It's about 80 lines including helpers. Steve Kinney's observation that a 100-line agent on this pattern can hit ~77% on SWE-bench Verified isn't an exaggeration — *this loop is the loop.* Everything past Module 2 is improvements to the things the loop *uses* (better tools, smarter context, more capable models, multi-agent coordination), not to the loop itself. The loop is fine.

The improvements are real, of course, and the next 16 modules will build them. But it's worth pausing on what we've shipped here:

- Correct handling of every stop reason
- Tool failures recoverable by the model
- Tool result truncation that doesn't lie about what was cut
- Three-dimensional budget (turns / tokens / dollars)
- Loop detection with model-visible feedback
- Concurrent tool execution
- API error handling with terminal-error distinction
- Structured trace and per-turn logging
- Graceful termination with partial results from every exit path

That's the production-hardened agent loop. The next modules layer onto it; they don't replace it.

::: code-exercise
**Exercise 2.1 — Stress-test your loop.**

Take the hardened loop above. Build it in a sandbox. Then deliberately break things:

1. **The repeating-fail tool.** Implement a tool that always raises a `ConnectionError`. Hand it to your agent with a prompt that requires using it. Confirm: the loop detector trips after 3 calls; budget never reaches its cap; agent returns partial result with `terminated_by="end_turn"` (because the model gave up gracefully on its own) or with the loop notice.

2. **The runaway tool.** Implement a tool that returns 200,000 tokens of fake JSON. Confirm: the truncation activates; the agent's context doesn't blow up; the cost ceiling holds.

3. **The tight budget.** Set `max_cost_usd=$0.01` and run a non-trivial query. Confirm: the agent terminates with `terminated_by="budget:cost_usd"` and returns a partial result; subsequent inspection of the trace shows where it stopped.

4. **The flaky API.** Mock `messages.create` to fail with `APIStatusError(503)` on the second call. Confirm: SDK retries; if retries succeed, loop continues; if retries fail, loop terminates with `terminated_by="api_error:503"` and a partial result.

By the end of this exercise, you've watched your loop survive every failure mode you'll see in production. That's the calibration you wanted.
:::

---

## Section 12 — When to Reach for the Agent SDK Instead

A short coda. The hardened loop above is great when you need control. The Claude Agent SDK is great when you don't.

Reach for the Agent SDK when:

- You're building a coding-adjacent agent (it has built-in `Read`, `Write`, `Edit`, `Bash`, `Grep`, `Glob` tools, and these are well-tuned)
- You want MCP servers wired up with minimal code
- You want hooks (PreToolUse, PostToolUse) for permission checks and telemetry
- The deployment shape "Python program shells out to a CLI subprocess" is acceptable for your stack

Reach for the Client SDK (the loop in this module) when:

- Your tools are entirely your own — internal APIs, custom databases, things the Agent SDK doesn't know about — and the built-in tools would be a distraction or a security risk
- You want exact control over the loop's behavior (custom termination logic, custom budget rules, embedded in a larger orchestrator)
- You're building multi-agent systems with bespoke coordination between agent loops (Module 14)
- You're targeting a deployment where adding a Node-based CLI subprocess to the runtime is a non-starter

For most readers of this book, the answer will land somewhere in the middle: use the Client SDK while you're learning, use the Agent SDK once you understand what it's hiding. Both are correct in different contexts. Knowing the loop is what lets you choose well.

::: pullquote
The framework you understand best is the one whose internals you've built once. Build the loop, then decide whether to keep using yours or switch.
:::

---

## Recap: Module 2 in eight bullets

::: bullet-points
- Every major agent framework runs the same core loop: call the model, execute tool calls if any, repeat until done. The loop isn't where the complexity lives.
- Handle every stop reason explicitly. `end_turn` and `tool_use` are the obvious ones; `max_tokens`, `refusal`, `stop_sequence`, and `pause_turn` need named handling too. The default branch should raise, not silently succeed.
- Tool failures should reach the model as `tool_result` with `is_error: true`, not as Python exceptions. The model can usually recover; crashing the loop throws away that recovery.
- Three budgets, layered: turns (catches infinite loops), tokens (catches large turns), dollars (catches expensive model selection). All three are cheap to track and each catches a different incident.
- Loop detection — same tool with same args N times in a recent window — catches immediate runaway that budgets only catch eventually. Inject the loop notice as a tool_result so the model can recover.
- Run parallel tool calls concurrently with `asyncio.gather`, and per-tool exception handling so one crash doesn't abort the batch.
- Graceful termination from every exit path. Return an `AgentResult` with `terminated_by` and `partial=True` instead of throwing — the caller decides whether the partial result is useful.
- Structured per-turn logging plus a trace object the loop returns. Enough observability to debug on day one; Module 17 adds the rest.
:::

---

::: sam-arc
**Sam, after the cold open.**

Sam spent Tuesday afternoon staring at the $43 PDF trace. Spent Wednesday adding loop detection. Spent Thursday morning fixing the `max_tokens`-treated-as-success bug after the support ticket landed. Spent Thursday afternoon adding tool result truncation. By Friday at noon the loop matched the one in Section 11 — three days of work, mostly subtraction (removing the silent failure modes), not addition.

Sam ran the loop against the same staging traffic that had broken it. The $43 PDF read terminated at $0.04 with the loop notice, the model giving up after the third repeat: "I'm unable to retrieve this PDF — the URL appears to be returning errors." The eleven-second null returns turned into ~30-second runs that surfaced "I produced more than expected; the response may be incomplete" warnings to the user. The 80,000-token customer record turned into a 20K truncated record with a marker — the model adapted, narrowed its query, finished in three turns instead of seven.

Friday evening, Sam wrote in their personal log: *"The loop in Module 1 was correct. The loop in Module 2 is honest."*

Sam's arc this module: **competence with the loop. Not the loop's six lines — the loop's failure modes. Knowing what can go wrong is the first half of knowing what to ship.**
:::

---

## What's next

Module 3 backs out one level and treats the five workflow patterns properly — prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer. Most of what you build is going to be one of these patterns, even after this module. The loop is the right tool when you need it; the workflow patterns are the right tools when you don't.

We'll come back to the loop in Module 4 when we add tool design discipline (so the model has tools worth calling), and again in Module 7 when we add context engineering (so the loop's growing context doesn't strangle it). For now: take the hardened loop. Run it. Stress-test it. Build calibration with it.

The next chapters assume you've done that. Sam did. So can you.
