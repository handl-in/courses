#!/usr/bin/env python3
"""Module 16: Distributed Data Parallel (DDP) — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part VI · Module 16</div>
  <h1 class="module-title">Distributed Data Parallel <em>(DDP)</em></h1>
  <p class="module-sub">— the same model on every GPU, different data on each, and the gradient-bucketing trick that overlaps communication with backward to make multi-GPU training nearly free</p>
</div>

<p>You have collectives. You have <code>torchrun</code>. The simplest non-trivial use of both is also by far the most common: <em>data parallelism</em>. Same model on every GPU. Different mini-batch on each. All-reduce gradients after backward. Step. Repeat.</p>

<p>The math is trivial — you're computing a mean of gradients, no different than gradient accumulation (M11) but spread across hardware. The interesting parts are (a) doing it efficiently with bucketing and overlap, (b) avoiding the half-dozen subtle bugs that bite first-time DDP users, and (c) knowing when DDP runs out of room and you need FSDP (M17) instead.</p>

<div class="keyidea">
DDP wraps a model so every <code>backward()</code> fires <strong>all-reduce</strong>s on gradients in the background, <em>overlapping with the rest of the backward pass</em>. The optimizer step, run on each rank with the now-averaged gradients, takes the same step everywhere — keeping models in lockstep. The whole pattern is ~5 lines of code. The reason it works at scale is the gradient-bucketing trick that makes the all-reduce nearly free in wall-clock time.
</div>

<h2>One new face</h2>

<div class="character" style="--c: #133e3b;">
  <div class="avatar" style="background: #133e3b; color: #fff;">DDP</div>
  <div>
    <p class="who">DDP</p>
    <p class="name">"I'm the wrapper that watches your backward pass and fires off all-reduces in the background."</p>
    <p class="says">When you do <code>model = DDP(model)</code>, I attach hooks to every parameter's <code>grad_fn</code>. As backward computes each parameter's gradient, my hook fires and queues that gradient for all-reduce. I group small gradients into ~25 MB <em>buckets</em> to amortize collective latency, and I dispatch the all-reduce <em>while backward is still running</em> on other layers. By the time backward finishes, most of the network traffic is already done. After backward, the optimizer step uses the averaged gradients, keeping every rank's model identical. <strong>I assume every rank ran the same code path on the same parameters.</strong> If your forward had a conditional branch that ran differently on different ranks, I'll hang.</p>
  </div>
</div>

<h2>The semantics in one paragraph</h2>

<p>Each of N ranks gets its own data shard via DistributedSampler (M10). Each forward and backward runs independently — no communication during compute. After backward, each rank has its own gradient tensors, computed from its local mini-batch. DDP all-reduces those gradients with reduction op SUM, then divides by N (or with op AVG directly). Now every rank has the <em>average</em> of all per-rank gradients. The optimizer step runs on each rank with those identical gradients, producing identical parameter updates. The models stay in sync.</p>

<p>The model on rank 0 and the model on rank 7 are <em>bit-identical</em> at every step boundary. They diverge during forward and backward (different inputs!), but the all-reduce + identical optimizer brings them back to identical state by step end. This is what makes DDP "data-parallel" in the strict sense: only the data differs.</p>

<h2>The minimal DDP setup</h2>

<pre><code><span class="kw">import</span> os, torch
<span class="kw">import</span> torch.distributed <span class="kw">as</span> dist
<span class="kw">from</span> torch.nn.parallel <span class="kw">import</span> DistributedDataParallel <span class="kw">as</span> DDP
<span class="kw">from</span> torch.utils.data.distributed <span class="kw">import</span> DistributedSampler

dist.<span class="fn">init_process_group</span>(backend=<span class="str">'nccl'</span>)
local_rank = <span class="fn">int</span>(os.environ[<span class="str">'LOCAL_RANK'</span>])
torch.cuda.<span class="fn">set_device</span>(local_rank)
device = torch.<span class="fn">device</span>(<span class="fn">f"cuda:{local_rank}"</span>)

<span class="com"># Build model — every rank constructs its own copy of the same architecture</span>
model = <span class="fn">build_model</span>().<span class="fn">to</span>(device)
model = <span class="fn">DDP</span>(model, device_ids=[local_rank])

<span class="com"># Build dataloader — DistributedSampler shards the data across ranks</span>
sampler = <span class="fn">DistributedSampler</span>(dataset, shuffle=<span class="kw">True</span>)
loader = <span class="fn">DataLoader</span>(dataset, batch_size=batch_per_rank, sampler=sampler,
                    num_workers=<span class="num">4</span>, pin_memory=<span class="kw">True</span>, persistent_workers=<span class="kw">True</span>)

opt = <span class="fn">build_optimizer</span>(model)

<span class="kw">for</span> epoch <span class="kw">in</span> <span class="fn">range</span>(num_epochs):
    sampler.<span class="fn">set_epoch</span>(epoch)              <span class="com"># critical — different shuffle per epoch</span>
    <span class="kw">for</span> x, y <span class="kw">in</span> loader:
        x, y = x.<span class="fn">to</span>(device, non_blocking=<span class="kw">True</span>), y.<span class="fn">to</span>(device, non_blocking=<span class="kw">True</span>)
        <span class="kw">with</span> torch.<span class="fn">autocast</span>(<span class="str">'cuda'</span>, dtype=torch.bfloat16):
            loss = <span class="fn">criterion</span>(model(x), y)
        opt.<span class="fn">zero_grad</span>()
        loss.<span class="fn">backward</span>()                    <span class="com"># DDP fires all-reduces during this</span>
        opt.<span class="fn">step</span>()</code></pre>

<p>That's it. Five practical changes from a single-GPU training loop:</p>

<ol>
  <li><code>init_process_group</code> + device selection — from M15.</li>
  <li>Build the model and move it to <code>cuda:local_rank</code> <em>before</em> wrapping with DDP.</li>
  <li>Wrap with <code>DDP(model, device_ids=[local_rank])</code>.</li>
  <li>Use <code>DistributedSampler</code> in your DataLoader, with <code>sampler.set_epoch(epoch)</code> per epoch.</li>
  <li>Run the script with <code>torchrun --nproc_per_node=N train.py</code>.</li>
</ol>

<p>Backward and step are unchanged. The all-reduce happens automatically inside <code>backward()</code> via DDP's hooks.</p>

<h2>The bucketing-and-overlap trick</h2>

<p>The naive way to do data-parallel: wait for backward to finish, all-reduce every gradient one at a time, then step. Each all-reduce pays ~10-50 μs of latency, plus bandwidth proportional to the gradient size. For a 1B-param model with thousands of parameters, you'd issue thousands of all-reduces — most of them small. Latency dominates.</p>

<p>DDP does two things to fix this:</p>

<ol>
  <li><strong>Bucketing</strong>: group adjacent parameters' gradients into ~25 MB chunks. Issue one all-reduce per bucket. For a 1B-param model in fp32 (4 GB), that's about 160 all-reduces instead of thousands — latency-amortized.</li>
  <li><strong>Overlap</strong>: don't wait for backward to finish. Each bucket all-reduces <em>as soon as all its gradients are ready</em>, while backward continues computing earlier-layer gradients. The communication overlaps with the rest of the compute.</li>
</ol>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 360" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Why DDP is fast: bucketing + overlap with backward</text>

  <!-- Bad version: sequential -->
  <text x="20" y="55" font-size="12" font-weight="700" fill="#c1502e">① Naïve: backward, then all-reduce</text>
  <g transform="translate(20, 65)">
    <text x="-5" y="14" font-size="9" fill="#6b5d4f" text-anchor="end">GPU</text>
    <text x="-5" y="34" font-size="9" fill="#6b5d4f" text-anchor="end">NCCL</text>
    <!-- backward fully runs first -->
    <rect x="0" y="3" width="350" height="14" fill="#1f5f5b"/>
    <text x="175" y="13" text-anchor="middle" font-size="10" fill="#fff">backward (compute)</text>
    <!-- then all-reduces happen serially -->
    <rect x="350" y="23" width="60" height="14" fill="#fcecec" stroke="#c1502e" stroke-width="0.5"/>
    <rect x="412" y="23" width="60" height="14" fill="#fcecec" stroke="#c1502e" stroke-width="0.5"/>
    <rect x="474" y="23" width="60" height="14" fill="#fcecec" stroke="#c1502e" stroke-width="0.5"/>
    <rect x="536" y="23" width="60" height="14" fill="#fcecec" stroke="#c1502e" stroke-width="0.5"/>
    <rect x="598" y="23" width="60" height="14" fill="#fcecec" stroke="#c1502e" stroke-width="0.5"/>
    <text x="500" y="33" text-anchor="middle" font-size="10" fill="#c1502e">5 buckets all-reduced sequentially</text>
    <!-- total time line -->
    <line x1="0"   y1="50" x2="660" y2="50" stroke="#c1502e" stroke-width="1.5"/>
    <line x1="0"   y1="46" x2="0"   y2="54" stroke="#c1502e" stroke-width="1.5"/>
    <line x1="660" y1="46" x2="660" y2="54" stroke="#c1502e" stroke-width="1.5"/>
    <text x="330" y="64" text-anchor="middle" font-size="11" font-weight="700" fill="#c1502e">total step ≈ backward + all-reduce</text>
  </g>

  <!-- Good version: bucketed + overlapped -->
  <text x="20" y="180" font-size="12" font-weight="700" fill="#1f5f5b">② DDP: bucketed all-reduces overlap with backward</text>
  <g transform="translate(20, 190)">
    <text x="-5" y="14" font-size="9" fill="#6b5d4f" text-anchor="end">GPU</text>
    <text x="-5" y="34" font-size="9" fill="#6b5d4f" text-anchor="end">NCCL</text>
    <!-- backward as one continuous block -->
    <rect x="0" y="3" width="350" height="14" fill="#1f5f5b"/>
    <text x="175" y="13" text-anchor="middle" font-size="10" fill="#fff">backward (compute)</text>
    <!-- buckets ready in reverse order (last layer first), allreduces fire concurrently -->
    <rect x="70"  y="23" width="60" height="14" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="0.5"/>
    <text x="100" y="33" text-anchor="middle" font-size="9" fill="#1a1612">B5</text>
    <rect x="140" y="23" width="60" height="14" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="0.5"/>
    <text x="170" y="33" text-anchor="middle" font-size="9" fill="#1a1612">B4</text>
    <rect x="210" y="23" width="60" height="14" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="0.5"/>
    <text x="240" y="33" text-anchor="middle" font-size="9" fill="#1a1612">B3</text>
    <rect x="280" y="23" width="60" height="14" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="0.5"/>
    <text x="310" y="33" text-anchor="middle" font-size="9" fill="#1a1612">B2</text>
    <rect x="350" y="23" width="60" height="14" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="0.5"/>
    <text x="380" y="33" text-anchor="middle" font-size="9" fill="#1a1612">B1</text>
    <!-- total time line — much shorter -->
    <line x1="0"   y1="50" x2="410" y2="50" stroke="#1f5f5b" stroke-width="1.5"/>
    <line x1="0"   y1="46" x2="0"   y2="54" stroke="#1f5f5b" stroke-width="1.5"/>
    <line x1="410" y1="46" x2="410" y2="54" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="205" y="64" text-anchor="middle" font-size="11" font-weight="700" fill="#1f5f5b">total step ≈ backward + tail of last bucket</text>
  </g>

  <!-- Annotations -->
  <text x="370" y="280" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">backward emits gradients in REVERSE order (last layer first)</text>
  <text x="370" y="305" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">→ DDP fires bucket B5 while still computing earlier layers' gradients</text>
  <text x="370" y="335" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">communication is hidden behind compute — nearly free in wall-clock</text>
</svg>
</div>

<p>Look at the difference. The naïve schedule has total time = backward + all-reduce. DDP's schedule has total time ≈ backward + tail (just the last bucket's all-reduce, which can't be hidden). On a fast network, the all-reduce is essentially invisible. On a slow network, the tail can still be a significant fraction — but it's far better than the naïve serialization.</p>

<p>The bucket size (25 MB by default) is the result of an empirical tradeoff. Smaller buckets → more chunks → more concurrency with backward but more latency overhead. Larger buckets → less concurrency but better bandwidth utilization per collective. 25 MB is the rough sweet spot for typical models on typical networks. You can tune it with the <code>bucket_cap_mb</code> arg of DDP if you want to experiment.</p>

<h2>The four DDP foot-guns</h2>

<h3>1. <code>find_unused_parameters</code></h3>

<p>If your model has parameters that don't contribute to the loss for a given forward pass, DDP's bucket-tracking machinery doesn't know to expect their gradients — so it waits forever for them. The classic case: conditional branches in <code>forward</code>, or multi-task models where some heads are inactive on some samples.</p>

<p>The flag:</p>

<pre><code>model = <span class="fn">DDP</span>(model, device_ids=[local_rank], find_unused_parameters=<span class="kw">True</span>)</code></pre>

<p>This tells DDP to scan after each forward and identify parameters that didn't get used, marking them as "no gradient expected" so the all-reduce doesn't wait. <em>It's slower</em> (the scan adds overhead per step), so don't enable it unless you actually have unused parameters.</p>

<p>Better fix where possible: ensure <em>all parameters always contribute</em> to the loss, even tiny multiplicative factors. Some teams add a fake <code>+ 0 * sum(p.sum() for p in ...)</code> to the loss to keep everything live. Hacky but explicit.</p>

<h3>2. The forward must be identical-shape across ranks</h3>

<p>If rank 0's forward sees a batch of length 100 and rank 1's sees length 200, DDP doesn't directly care — it averages whatever gradients come out. But there are two indirect failure modes. (a) <em>BatchNorm running statistics</em>: with vanilla BN, each rank updates its own running stats from its own batch — so they diverge unless you use <code>SyncBatchNorm</code>. (b) <em>Variable-shape attention or convolution</em>: padding and masking still need to be consistent so gradient values agree.</p>

<p>For BN specifically:</p>

<pre><code>model = nn.SyncBatchNorm.<span class="fn">convert_sync_batchnorm</span>(model)
model = <span class="fn">DDP</span>(model, device_ids=[local_rank])</code></pre>

<p>SyncBN does a small all-reduce inside its forward to compute statistics across all ranks. Slower per step but correct. If your model uses BatchNorm at all in distributed training, you almost always want SyncBN.</p>

<h3>3. Gradient accumulation needs <code>no_sync()</code></h3>

<p>If you do gradient accumulation across K micro-batches before stepping (M11), you don't want DDP to all-reduce on every micro-batch — only on the last one. Otherwise you're paying network cost K-1 times for nothing.</p>

<pre><code><span class="kw">for</span> step, batch <span class="kw">in</span> <span class="fn">enumerate</span>(loader):
    is_accum_step = (step + <span class="num">1</span>) % accum_steps != <span class="num">0</span>
    <span class="kw">with</span> model.<span class="fn">no_sync</span>() <span class="kw">if</span> is_accum_step <span class="kw">else</span> contextlib.<span class="fn">nullcontext</span>():
        loss = <span class="fn">criterion</span>(model(batch.x), batch.y) / accum_steps
        loss.<span class="fn">backward</span>()                  <span class="com"># gradients accumulate locally; no all-reduce</span>

    <span class="kw">if</span> <span class="kw">not</span> is_accum_step:
        <span class="com"># this final backward fires the all-reduce on accumulated grads</span>
        opt.<span class="fn">step</span>()
        opt.<span class="fn">zero_grad</span>()</code></pre>

<p>The pattern: wrap micro-batch backwards (except the last in a group) in <code>model.no_sync()</code>. The grads accumulate locally without a network call. The final unwrapped backward fires the all-reduce on the now-accumulated gradient. Network traffic is reduced K× to once per <em>real</em> step.</p>

<h3>4. Save the unwrapped model</h3>

<p>From M7: a DDP-wrapped model's <code>state_dict()</code> has every key prefixed with <code>module.</code> — because <code>DDP</code> stores the original model as <code>self.module</code>. If you save and load on different rank counts (or unwrap for inference), the prefix mismatches.</p>

<pre><code><span class="com"># Save the underlying model, not the DDP wrapper:</span>
torch.<span class="fn">save</span>(model.module.<span class="fn">state_dict</span>(), <span class="str">'checkpoint.pt'</span>)

<span class="com"># Or, equivalently, strip the prefix on load:</span>
sd = torch.<span class="fn">load</span>(<span class="str">'checkpoint.pt'</span>)
sd = {k.<span class="fn">removeprefix</span>(<span class="str">'module.'</span>): v <span class="kw">for</span> k, v <span class="kw">in</span> sd.<span class="fn">items</span>()}
plain_model.<span class="fn">load_state_dict</span>(sd)</code></pre>

<p>Also: <strong>only rank 0 should save the checkpoint</strong>. All ranks have identical model state (DDP guarantees that), so there's no point saving N copies. Standard pattern:</p>

<pre><code><span class="kw">if</span> dist.<span class="fn">get_rank</span>() == <span class="num">0</span>:
    torch.<span class="fn">save</span>(model.module.<span class="fn">state_dict</span>(), <span class="fn">f"ckpt_step{step}.pt"</span>)
dist.<span class="fn">barrier</span>()         <span class="com"># every rank waits for rank 0 to finish writing</span></code></pre>

<p>The <code>dist.barrier()</code> ensures the other ranks don't blast ahead while rank 0 is still writing — useful especially before re-loading.</p>

<h2>The "find_unused_parameters" trace signature</h2>

<p>If you forget <code>find_unused_parameters=True</code> on a model that has unused parameters, DDP doesn't fail immediately. Instead, you get a hang at the <em>second</em> training step. The first step works (DDP hasn't yet pinned which buckets to expect). The second step waits for an all-reduce that will never arrive, because the corresponding gradient was never produced. <em>Symptom: training logs step 0, then nothing, no NCCL timeout for 30 minutes</em>.</p>

<p>The fix is to enable the flag, but the diagnosis is the lesson: <strong>look at the rank that's stuck and ask "which parameter's all-reduce is it waiting for?"</strong> The answer is whatever your model didn't run on this batch.</p>

<h2>Effective batch size and learning rate</h2>

<p>With DDP, your <em>effective batch size</em> is <code>per_rank_batch × N</code>. If you trained on 1 GPU at batch=64 and now run on 8 GPUs at the same per-rank batch=64, your effective batch is 512. The standard rules of thumb (M9):</p>

<ul>
  <li><strong>Linear scaling rule</strong> (square-root for adaptive optimizers): scale LR up roughly with batch size to maintain training dynamics. For Adam, often LR scales with √(batch_size_ratio); for SGD, LR scales linearly.</li>
  <li><strong>Warmup gets longer</strong>: bigger effective batches need longer warmup. A rule of thumb: warmup duration scales with batch size.</li>
</ul>

<p>If you don't adjust LR, you might see worse training despite more compute — because effectively you're under-stepping. The "trains fine on 1 GPU, diverges on 8 GPUs" bug from M9's exercises is precisely this.</p>

<div class="ndq">
<h4>About DDP</h4>

<p class="q">When does DDP run out of room and I need FSDP?</p>
<p class="a">When your model + gradients + optimizer state don't fit on a single GPU. DDP requires the full model on each rank — for a 7B parameter model in mixed-precision AdamW (~120 GB of state, M12), you simply can't fit on a single 80 GB H100. FSDP shards the parameters, gradients, and optimizer state across ranks, trading more communication for the ability to train models that don't fit. We'll do that in M17.</p>

<p class="q">My DDP run is slower per step than my single-GPU baseline. What's wrong?</p>
<p class="a">First, sanity check: <em>per-rank</em> batch should be the same as single-GPU; total batch is N×. If it's slower per-step at the same per-rank size, suspect (a) network is slower than NVLink (intra-node check via topology tools), (b) <code>find_unused_parameters=True</code> when it doesn't need to be (the scan adds overhead), or (c) bucket size doesn't match your model — try <code>bucket_cap_mb=50</code> or <code>10</code> and profile.</p>

<p class="q">What's <code>broadcast_buffers</code>?</p>
<p class="a">DDP option (default <code>True</code>) that broadcasts buffers (e.g., BN running stats) from rank 0 to all ranks at the start of each forward, keeping them in sync. Set <code>False</code> if your buffers shouldn't be synced (rare). For BN specifically, prefer SyncBatchNorm over relying on broadcast_buffers — broadcast just copies rank 0's stats, which weren't computed across ranks anyway.</p>

<p class="q">Can I have different model architectures on different ranks?</p>
<p class="a">No — DDP's invariant is "same model everywhere." If you need heterogeneous models (e.g., pipeline parallel where each rank holds a different stage), DDP isn't the right tool. M18 covers pipeline parallelism, where you do exactly that with different mechanisms (point-to-point sends/recvs).</p>

<p class="q">Does DDP work with <code>torch.compile</code>?</p>
<p class="a">Yes, but the order matters: compile the model first, then wrap with DDP. <code>model = torch.compile(model); model = DDP(model)</code>. Otherwise the compile boundary doesn't match DDP's hooks. M21 covers compile in detail.</p>
</div>

<h2>The complete DDP training script (skeleton)</h2>

<pre><code><span class="kw">import</span> os, contextlib, torch
<span class="kw">import</span> torch.distributed <span class="kw">as</span> dist
<span class="kw">import</span> torch.nn <span class="kw">as</span> nn
<span class="kw">from</span> torch.nn.parallel <span class="kw">import</span> DistributedDataParallel <span class="kw">as</span> DDP
<span class="kw">from</span> torch.utils.data.distributed <span class="kw">import</span> DistributedSampler
<span class="kw">from</span> torch.utils.data <span class="kw">import</span> DataLoader

<span class="kw">def</span> <span class="fn">train</span>(config):
    dist.<span class="fn">init_process_group</span>(backend=<span class="str">'nccl'</span>)
    rank        = dist.<span class="fn">get_rank</span>()
    world_size  = dist.<span class="fn">get_world_size</span>()
    local_rank  = <span class="fn">int</span>(os.environ[<span class="str">'LOCAL_RANK'</span>])
    torch.cuda.<span class="fn">set_device</span>(local_rank)
    device = torch.<span class="fn">device</span>(<span class="fn">f"cuda:{local_rank}"</span>)

    model = <span class="fn">build_model</span>(config).<span class="fn">to</span>(device)
    model = nn.SyncBatchNorm.<span class="fn">convert_sync_batchnorm</span>(model)        <span class="com"># if you have BN</span>
    model = <span class="fn">DDP</span>(model, device_ids=[local_rank])
    opt = <span class="fn">build_optimizer</span>(model)

    sampler = <span class="fn">DistributedSampler</span>(dataset, shuffle=<span class="kw">True</span>)
    loader = <span class="fn">DataLoader</span>(dataset, batch_size=config.batch_per_rank,
                        sampler=sampler, num_workers=<span class="num">4</span>,
                        pin_memory=<span class="kw">True</span>, persistent_workers=<span class="kw">True</span>)

    <span class="kw">for</span> epoch <span class="kw">in</span> <span class="fn">range</span>(config.num_epochs):
        sampler.<span class="fn">set_epoch</span>(epoch)
        model.<span class="fn">train</span>()
        <span class="kw">for</span> step, (x, y) <span class="kw">in</span> <span class="fn">enumerate</span>(loader):
            x, y = x.<span class="fn">to</span>(device, non_blocking=<span class="kw">True</span>), y.<span class="fn">to</span>(device, non_blocking=<span class="kw">True</span>)
            is_accum = (step + <span class="num">1</span>) % config.accum_steps != <span class="num">0</span>
            <span class="kw">with</span> model.<span class="fn">no_sync</span>() <span class="kw">if</span> is_accum <span class="kw">else</span> contextlib.<span class="fn">nullcontext</span>():
                <span class="kw">with</span> torch.<span class="fn">autocast</span>(<span class="str">'cuda'</span>, dtype=torch.bfloat16):
                    loss = <span class="fn">criterion</span>(model(x), y) / config.accum_steps
                loss.<span class="fn">backward</span>()
            <span class="kw">if</span> <span class="kw">not</span> is_accum:
                torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), <span class="num">1.0</span>)
                opt.<span class="fn">step</span>(); opt.<span class="fn">zero_grad</span>()

        <span class="kw">if</span> rank == <span class="num">0</span> <span class="kw">and</span> epoch % config.ckpt_every == <span class="num">0</span>:
            torch.<span class="fn">save</span>(model.module.<span class="fn">state_dict</span>(), <span class="fn">f"ckpt_e{epoch}.pt"</span>)
        dist.<span class="fn">barrier</span>()

    dist.<span class="fn">destroy_process_group</span>()</code></pre>

<p>Run with <code>torchrun --nproc_per_node=8 train.py</code>. That's a complete, production-flavored DDP training script. Note all the patterns from M10 (DistributedSampler + set_epoch), M11 (gradient accumulation, checkpoint discipline), M14 (autocast), and M15 (init_process_group + LOCAL_RANK device selection) all wired together.</p>

<h2>Code Magnets: assemble a DDP-correct backward step</h2>

<p>Build the inner DDP step that handles gradient accumulation correctly. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets into a working accumulation step.</p>

<div class="magnet-pool">
  <span class="magnet">is_accum = (step + 1) % accum_steps != 0</span>
  <span class="magnet">with model.no_sync() if is_accum else contextlib.nullcontext():</span>
  <span class="magnet">with model.no_sync():</span>
  <span class="magnet">    loss = criterion(model(x), y) / accum_steps</span>
  <span class="magnet">    loss.backward()</span>
  <span class="magnet">if not is_accum:</span>
  <span class="magnet">    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)</span>
  <span class="magnet">    opt.step(); opt.zero_grad()</span>
  <span class="magnet">opt.step(); opt.zero_grad()</span>
  <span class="magnet">dist.all_reduce(loss)</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code>is_accum = (step + <span class="num">1</span>) % accum_steps != <span class="num">0</span>
<span class="kw">with</span> model.<span class="fn">no_sync</span>() <span class="kw">if</span> is_accum <span class="kw">else</span> contextlib.<span class="fn">nullcontext</span>():
    loss = <span class="fn">criterion</span>(<span class="fn">model</span>(x), y) / accum_steps
    loss.<span class="fn">backward</span>()
<span class="kw">if</span> <span class="kw">not</span> is_accum:
    torch.nn.utils.<span class="fn">clip_grad_norm_</span>(model.<span class="fn">parameters</span>(), <span class="num">1.0</span>)
    opt.<span class="fn">step</span>(); opt.<span class="fn">zero_grad</span>()</code></pre>
<p>The traps:</p>
<ul>
  <li><code>with model.no_sync():</code> always, no condition — that suppresses the all-reduce <em>even on the final accum step</em>. Models drift apart, training is broken.</li>
  <li><code>opt.step(); opt.zero_grad()</code> outside the <code>if not is_accum</code> guard — the optimizer would step every micro-batch, defeating accumulation.</li>
  <li><code>dist.all_reduce(loss)</code> — DDP all-reduces the <em>gradients</em>, not the loss. The loss is local; for logging, you can optionally all-reduce it, but it's not part of the training-step correctness.</li>
</ul>
<p>The structure: <strong>no_sync only on micro-batches that are not the final one in a group</strong>; the final backward (without no_sync) fires the all-reduce on accumulated gradients; then the optimizer steps.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each DDP concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>DistributedDataParallel</div>
  <div>A. Wrapper that hooks into backward and fires bucket all-reduces in the background.</div>

  <div>Gradient bucketing</div>
  <div>B. Synchronizes BatchNorm statistics across ranks via mini-collectives in forward.</div>

  <div>find_unused_parameters=True</div>
  <div>C. Group ~25 MB of gradients per all-reduce to amortize collective latency.</div>

  <div>SyncBatchNorm</div>
  <div>D. Suppresses DDP's gradient sync for one backward pass — used in accumulation.</div>

  <div>model.no_sync()</div>
  <div>E. Tells DDP to scan for parameters that didn't get gradients, marking them as expected-missing.</div>

  <div>sampler.set_epoch(epoch)</div>
  <div>F. Without this, DistributedSampler shuffles the same way every epoch.</div>

  <div>save model.module.state_dict()</div>
  <div>G. Strips the DDP wrapper's <code>module.</code> prefix before saving — clean checkpoints.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>DistributedDataParallel</strong> → A<br>
<strong>Gradient bucketing</strong> → C<br>
<strong>find_unused_parameters=True</strong> → E<br>
<strong>SyncBatchNorm</strong> → B<br>
<strong>model.no_sync()</strong> → D<br>
<strong>sampler.set_epoch(epoch)</strong> → F<br>
<strong>save model.module.state_dict()</strong> → G
</p>
<p>The mental shortcut: <em>DDP wraps and hooks, bucketing amortizes collectives, find_unused handles divergent forwards, SyncBN cross-rank stats, no_sync suppresses one all-reduce, set_epoch reshuffles per epoch, .module unwraps the prefix</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> Your DDP run hangs after the first training step. <code>nvidia-smi</code> shows GPUs at 0%, and you don't see any error. What's the first thing to check?</p>
<details class="answer"><summary>show answer</summary>
<p><code>find_unused_parameters</code>. The hang-after-first-step pattern is DDP's signature for "the bucket-tracking machinery is waiting for an all-reduce on a parameter that didn't receive a gradient." Either (a) wrap with <code>find_unused_parameters=True</code> (slow but correct), or (b) restructure the forward so all parameters are always touched by the loss. If the model has conditional branches, you usually want (a). If it's a multi-task model where some heads are skipped, (a) again. Note that this hang doesn't trigger NCCL's timeout for 30 minutes by default — you have to recognize the pattern from the trace yourself.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> A team trained their CNN on 1 GPU at batch=64 and got 76% val accuracy. They scale to 8 GPUs at per-rank batch=64 (effective batch=512), run the same recipe, and get 71%. What's likely happening?</p>
<details class="answer"><summary>show answer</summary>
<p>Two candidates. (1) <strong>BatchNorm contamination</strong>: each rank's BN computes statistics from 64 samples instead of the full 512. The statistics are noisier, especially at lower batch sizes. Fix: <code>SyncBatchNorm.convert_sync_batchnorm(model)</code>. (2) <strong>Effective batch size shift</strong>: at 8× the batch, you typically need to scale LR up (linearly for SGD, by √8 for Adam) and extend warmup. Without that, you're effectively under-stepping. Try both fixes — usually you need both.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> You time your DDP backward and see it takes the same wall time as your single-GPU backward. Is that suspicious?</p>
<details class="answer"><summary>show answer</summary>
<p>No — that's actually <em>good news</em>. Backward in DDP includes the all-reduces, which are running concurrently with backward computation. If they're fully hidden by the overlap, your DDP backward time ≈ your single-GPU backward time. That's the magic of bucketing + overlap. If the times differ noticeably, then either (a) network is slow enough that the tail of the last bucket extends past backward, or (b) something is forcing a sync (printing intermediate values, calling .item() in the hot path). Profile to confirm.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> You want to all-reduce a per-batch metric (e.g., total loss across ranks for logging). Write the line.</p>
<details class="answer"><summary>show answer</summary>
<pre><code>loss_for_logging = loss.<span class="fn">detach</span>()
dist.<span class="fn">all_reduce</span>(loss_for_logging, op=dist.ReduceOp.AVG)
<span class="kw">if</span> rank == <span class="num">0</span>:
    <span class="fn">log</span>(<span class="fn">f"step {step}: loss {loss_for_logging.item():.4f}"</span>)</code></pre>
<p>Two important things. (1) <code>.detach()</code> first — you don't want autograd to think this all-reduce is part of the graph. (2) <code>op=dist.ReduceOp.AVG</code> is supported in modern PyTorch and avoids the need to manually divide by world_size. Older versions used SUM and divided manually. (3) Only log on rank 0; otherwise N copies of the same line spam stdout.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li><strong>DDP's contract</strong>: same model on every rank, different data, all-reduce gradients after backward. Models stay bit-identical at step boundaries.</li>
  <li><strong>The wrapper</strong>: <code>DDP(model, device_ids=[local_rank])</code> attaches autograd hooks that fire all-reduces during backward.</li>
  <li><strong>Bucketing</strong>: gradients grouped into ~25 MB buckets to amortize collective latency. ~hundreds of all-reduces per step instead of thousands.</li>
  <li><strong>Overlap</strong>: each bucket all-reduces as soon as its gradients are ready, while backward continues on earlier layers. Communication is largely hidden behind compute.</li>
  <li><strong>DistributedSampler</strong> shards data across ranks. Always call <code>sampler.set_epoch(epoch)</code> per epoch — otherwise shuffle is identical every epoch.</li>
  <li><strong>find_unused_parameters=True</strong>: needed when forward conditionally skips parameters. Symptom of forgetting it: hang on the second training step, no error.</li>
  <li><strong>SyncBatchNorm</strong>: replace BN layers with this when training distributed. <code>nn.SyncBatchNorm.convert_sync_batchnorm(model)</code> before DDP wrap.</li>
  <li><strong><code>model.no_sync()</code></strong>: context manager that suppresses DDP's all-reduce for one backward. Use it on micro-batches in gradient accumulation; let the final backward fire the all-reduce on accumulated grads.</li>
  <li><strong>Save <code>model.module.state_dict()</code></strong> — unwraps the DDP prefix. Save only on rank 0; <code>dist.barrier()</code> after.</li>
  <li><strong>Effective batch = per_rank × N</strong>. Adjust LR and warmup accordingly. Linear scaling for SGD, √-scaling for Adam, longer warmup with bigger batches.</li>
  <li><strong>DDP runs out when the model+grads+state don't fit on one GPU.</strong> Then you need FSDP — Module 17.</li>
  <li>The reflex: when DDP misbehaves, ask which collective is hanging or missing. Print rank IDs at suspect lines. Look at trace for the four bottleneck shapes from M13.</li>
</ul>
</div>

<p>Module 17 takes the model that doesn't fit on a single GPU and shards it. ZeRO-1, ZeRO-2, ZeRO-3. FSDP. The all-gather + reduce-scatter dance that lets you train 70B models on commodity hardware. The same identity from M15 (<code>all_reduce ≡ reduce_scatter + all_gather</code>) is the entire conceptual basis.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">16</span>
  <span>Distributed Data Parallel (DDP)</span>
</div>
"""

emit("16_ddp", "Module 16 — Distributed Data Parallel (DDP)", BODY)
