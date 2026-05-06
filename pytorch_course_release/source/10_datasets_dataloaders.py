#!/usr/bin/env python3
"""Module 10: Datasets & DataLoaders done right — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part IV · Module 10</div>
  <h1 class="module-title">Datasets &amp; DataLoaders, <em>done right</em></h1>
  <p class="module-sub">— the producer-consumer pipeline that feeds your GPU, the <code>__getitem__</code> trap that wastes 40% of training time, and the worker-RNG bug that breaks reproducibility silently</p>
</div>

<p>The data loader is the part of PyTorch that nobody photographs and everybody underrates. You spent two modules tuning your optimizer and three more on autograd, and now your GPU sits at 30% utilization because you wrote one slow line in <code>__getitem__</code>. Welcome to the unsexy half of training performance.</p>

<p>The good news is that the data pipeline has a small surface area: a <code>Dataset</code> that produces samples, a <code>DataLoader</code> that batches them, a few worker subprocesses that parallelize <code>__getitem__</code>, and pinned-memory hand-off to the GPU. The bad news is that every one of those four pieces has a way to be silently slow or silently wrong. This module teaches you the failure modes.</p>

<div class="keyidea">
A <code>DataLoader</code> is a <strong>producer-consumer pipeline</strong>. The producers are <code>num_workers</code> child processes, each running <code>__getitem__</code> on the indices a Sampler hands them. The consumer is your training loop. The pipeline buffer is set by <code>prefetch_factor</code>. The hand-off to GPU is sped up by <code>pin_memory=True</code>. Get any one of those wrong and the GPU sits idle while training drags. Get them all right and the data pipeline disappears into the background.
</div>

<h2>Three new players</h2>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">DL</div>
  <div>
    <p class="who">DataLoader</p>
    <p class="name">"I'm the foreman. I orchestrate the workers, batch their outputs, and hand things to the GPU."</p>
    <p class="says">When you wrap me around a Dataset, I forge a small army of subprocesses (Workers), give them indices to fetch, collect their results, batch them with <code>collate_fn</code>, and (if you asked) pin the memory before yielding. <code>num_workers</code> sets the army size. <code>prefetch_factor</code> sets how far ahead I let them work. <code>persistent_workers</code> tells me whether to keep them alive between epochs. Get those three right and your GPU never starves.</p>
  </div>
</div>

<div class="character" style="--c: #c1502e;">
  <div class="avatar" style="background: #c1502e; color: #fff;">W</div>
  <div>
    <p class="who">Worker</p>
    <p class="name">"I'm one of <code>num_workers</code> child processes. I do the slow stuff so the main process can keep the GPU busy."</p>
    <p class="says">I get spawned at DataLoader iteration start (or once, if <code>persistent_workers=True</code>). I receive indices, run <code>__getitem__</code> for each, return the results. I have my own copy of the Dataset, my own RNG state (which you have to handle — that's the bug), and my own memory. <em>I don't share GPU access</em> — one of you only at the very end, on the consumer side.</p>
  </div>
</div>

<div class="character" style="--c: #b85a6c;">
  <div class="avatar" style="background: #b85a6c; color: #fff;">S</div>
  <div>
    <p class="who">Sampler</p>
    <p class="name">"I decide what indices to fetch, in what order. That's it."</p>
    <p class="says">Default is <code>SequentialSampler</code> (0, 1, 2, ...) when <code>shuffle=False</code>, or <code>RandomSampler</code> (a random permutation) when <code>shuffle=True</code>. For distributed training there's <code>DistributedSampler</code> that gives each rank a disjoint slice. For weighted/balanced sampling there's <code>WeightedRandomSampler</code>. They all just produce indices; the Workers do the loading.</p>
  </div>
</div>

<h2>The pipeline, in one diagram</h2>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 320" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrDL" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">DataLoader: producer-consumer pipeline</text>

  <!-- Sampler -->
  <g transform="translate(20, 60)">
    <rect x="0" y="0" width="120" height="56" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="60" y="22" text-anchor="middle" font-size="13" font-weight="700" fill="#1a1612">Sampler</text>
    <text x="60" y="38" text-anchor="middle" font-size="11" fill="#1a1612">produces indices</text>
    <text x="60" y="50" text-anchor="middle" font-size="10" fill="#6b5d4f">[7, 12, 3, 4, ...]</text>
  </g>

  <!-- Workers (3) -->
  <g transform="translate(190, 30)">
    <rect x="0" y="0" width="160" height="48" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="80" y="20" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Worker 0</text>
    <text x="80" y="36" text-anchor="middle" font-size="10" fill="#1a1612">__getitem__(7) → sample</text>
  </g>
  <g transform="translate(190, 84)">
    <rect x="0" y="0" width="160" height="48" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="80" y="20" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Worker 1</text>
    <text x="80" y="36" text-anchor="middle" font-size="10" fill="#1a1612">__getitem__(12) → sample</text>
  </g>
  <g transform="translate(190, 138)">
    <rect x="0" y="0" width="160" height="48" fill="#fcecec" stroke="#c1502e" stroke-width="2"/>
    <text x="80" y="20" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">Worker 2</text>
    <text x="80" y="36" text-anchor="middle" font-size="10" fill="#1a1612">__getitem__(3) → sample</text>
  </g>
  <text x="270" y="200" font-size="10" fill="#6b5d4f" text-anchor="middle">(num_workers=3)</text>

  <!-- collate_fn -->
  <g transform="translate(380, 84)">
    <rect x="0" y="0" width="120" height="48" fill="#fff8a8" stroke="#1a1612" stroke-width="2"/>
    <text x="60" y="20" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">collate_fn</text>
    <text x="60" y="36" text-anchor="middle" font-size="10" fill="#1a1612">batch list of samples</text>
  </g>

  <!-- Pin memory -->
  <g transform="translate(525, 84)">
    <rect x="0" y="0" width="100" height="48" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="2"/>
    <text x="50" y="20" text-anchor="middle" font-size="12" font-weight="700" fill="#1a1612">pin_memory</text>
    <text x="50" y="36" text-anchor="middle" font-size="10" fill="#1a1612">page-lock</text>
  </g>

  <!-- GPU -->
  <g transform="translate(650, 84)">
    <rect x="0" y="0" width="80" height="48" fill="#1f5f5b" stroke="#1a1612" stroke-width="2"/>
    <text x="40" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#fff">GPU</text>
    <text x="40" y="38" text-anchor="middle" font-size="10" fill="#fff">async DMA</text>
  </g>

  <!-- arrows: sampler → workers -->
  <path d="M 145 88 L 188 54" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrDL)"/>
  <path d="M 145 88 L 188 108" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrDL)"/>
  <path d="M 145 88 L 188 162" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrDL)"/>

  <!-- arrows: workers → collate -->
  <path d="M 350 54 L 378 100" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrDL)"/>
  <path d="M 350 108 L 378 108" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrDL)"/>
  <path d="M 350 162 L 378 116" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrDL)"/>

  <!-- arrows: collate → pin → GPU -->
  <path d="M 500 108 L 523 108" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrDL)"/>
  <path d="M 625 108 L 648 108" stroke="#1f5f5b" stroke-width="1.5" fill="none" marker-end="url(#arrDL)"/>

  <!-- prefetch annotation -->
  <text x="270" y="240" font-family="'Caveat', cursive" font-size="20" fill="#c1502e" text-anchor="middle">prefetch_factor=2 → each worker keeps 2 batches "in flight"</text>
  <text x="370" y="270" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b" text-anchor="middle">if any link is slower than the GPU, the GPU stalls.</text>
  <text x="370" y="295" font-family="'Caveat', cursive" font-size="20" fill="#1f5f5b" text-anchor="middle">find the bottleneck. fix the bottleneck.</text>
</svg>
</div>

<p>That's the whole picture. Read it left to right: a Sampler emits indices, the indices are distributed across Workers running in subprocesses, each Worker calls <code>__getitem__</code> and returns a sample, the main process gathers samples into batches via <code>collate_fn</code>, optionally pins the result for fast GPU transfer, and your training loop consumes from the queue.</p>

<p><em>Every parameter you tune in <code>DataLoader(...)</code> controls one of those arrows.</em></p>

<h2>Datasets: map-style vs iterable-style</h2>

<p>Two different abstractions for "a thing that yields samples." Each fits a different data shape.</p>

<h3>Map-style: random access</h3>

<p>A dataset is a thing with <code>__len__</code> and <code>__getitem__</code>. The DataLoader can ask "give me sample 42." This is the right model when your data is finite and indexable — files in a folder, rows in a database, examples in a parquet file.</p>

<pre><code><span class="kw">from</span> torch.utils.data <span class="kw">import</span> Dataset

<span class="kw">class</span> <span class="ty">ImageDataset</span>(Dataset):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, paths, transform):
        self.paths = paths
        self.transform = transform

    <span class="kw">def</span> <span class="fn">__len__</span>(self):
        <span class="kw">return</span> <span class="fn">len</span>(self.paths)

    <span class="kw">def</span> <span class="fn">__getitem__</span>(self, idx):
        img = Image.<span class="fn">open</span>(self.paths[idx]).<span class="fn">convert</span>(<span class="str">'RGB'</span>)
        <span class="kw">return</span> self.<span class="fn">transform</span>(img), self.<span class="fn">parse_label</span>(idx)</code></pre>

<p>That's the canonical map-style Dataset. Note three things: (1) the constructor stores indexable references (paths, not images), (2) <code>__getitem__</code> does the actual I/O and transform per call, (3) the return is whatever you want — a tuple, a dict, a custom class. <code>collate_fn</code> will figure out how to batch it.</p>

<h3>Iterable-style: streaming</h3>

<p>For data that doesn't fit on disk, comes from a network stream, or has unknown size. Your dataset implements <code>__iter__</code>, not <code>__getitem__</code>. The DataLoader just calls <code>iter()</code> and pulls samples until exhausted.</p>

<pre><code><span class="kw">from</span> torch.utils.data <span class="kw">import</span> IterableDataset

<span class="kw">class</span> <span class="ty">StreamingDataset</span>(IterableDataset):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, url):
        self.url = url

    <span class="kw">def</span> <span class="fn">__iter__</span>(self):
        worker_info = torch.utils.data.<span class="fn">get_worker_info</span>()
        <span class="com"># split the stream across workers — each worker gets a disjoint slice</span>
        <span class="kw">if</span> worker_info <span class="kw">is</span> <span class="kw">None</span>:
            shards = <span class="fn">all_shards</span>(self.url)
        <span class="kw">else</span>:
            shards = <span class="fn">all_shards</span>(self.url)[worker_info.id::worker_info.num_workers]
        <span class="kw">for</span> shard <span class="kw">in</span> shards:
            <span class="kw">for</span> sample <span class="kw">in</span> <span class="fn">decode_shard</span>(shard):
                <span class="kw">yield</span> sample</code></pre>

<p>The catch with iterable datasets: you have to <em>shard the stream across workers yourself</em>. Otherwise every worker iterates the same data and you get duplicates. The <code>get_worker_info()</code> call exposes the worker id and total worker count, so you can do <code>shards[worker_id::num_workers]</code> as the standard pattern.</p>

<div class="table-wrap">
<table>
<caption>When to use which</caption>
<thead><tr><th></th><th>Map-style</th><th>Iterable-style</th></tr></thead>
<tbody>
<tr><td>Total size known?</td><td>Yes — <code>len(ds)</code></td><td>Often unknown</td></tr>
<tr><td>Random access?</td><td>Yes — any index</td><td>No — sequential only</td></tr>
<tr><td>Shuffling</td><td>Free — Sampler does it</td><td>Manual — buffer + shuffle</td></tr>
<tr><td>Worker sharding</td><td>Automatic via Sampler</td><td>You write it in <code>__iter__</code></td></tr>
<tr><td>Use for</td><td>Local files, databases, parquet</td><td>Streaming web data, infinite generators</td></tr>
</tbody>
</table>
</div>

<p>For >95% of training, map-style is what you want. Iterable-style shines for big-corpus LLM pretraining where the data lives in remote shards and you can't afford to materialize an index.</p>

<h2>The <code>__getitem__</code> trap</h2>

<p>This is the single most common cause of slow training. Watch:</p>

<pre><code><span class="kw">class</span> <span class="ty">SlowDataset</span>(Dataset):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, csv_path):
        self.df = pd.<span class="fn">read_csv</span>(csv_path)              <span class="com"># load metadata once — fine</span>

    <span class="kw">def</span> <span class="fn">__getitem__</span>(self, idx):
        row = self.df.<span class="fn">iloc</span>[idx]
        img = Image.<span class="fn">open</span>(row[<span class="str">'path'</span>])                  <span class="com"># I/O per call — fine</span>
        img = img.<span class="fn">convert</span>(<span class="str">'RGB'</span>).<span class="fn">resize</span>((<span class="num">224</span>, <span class="num">224</span>))   <span class="com"># compute per call — fine</span>
        big_lookup = pd.<span class="fn">read_parquet</span>(<span class="str">'metadata.parquet'</span>)  <span class="com"># 🔥 every call!</span>
        meta = big_lookup[big_lookup.id == row[<span class="str">'id'</span>]]      <span class="com"># 🔥 full scan!</span>
        <span class="kw">return</span> img, meta</code></pre>

<p>Spot the problem? Two of those lines run <em>every time the loader fetches a sample</em>. Across an epoch of 100k samples, you're reading the same parquet file 100,000 times. Even with workers, you waste minutes per epoch.</p>

<p>The fix: <em>everything that doesn't depend on <code>idx</code> belongs in <code>__init__</code>, not in <code>__getitem__</code>.</em></p>

<pre><code><span class="kw">class</span> <span class="ty">FastDataset</span>(Dataset):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, csv_path):
        self.df = pd.<span class="fn">read_csv</span>(csv_path)
        big_lookup = pd.<span class="fn">read_parquet</span>(<span class="str">'metadata.parquet'</span>)   <span class="com"># once</span>
        self.meta = big_lookup.<span class="fn">set_index</span>(<span class="str">'id'</span>).<span class="fn">to_dict</span>(<span class="str">'index'</span>)
        <span class="com"># Now meta lookup is O(1) per call</span>

    <span class="kw">def</span> <span class="fn">__getitem__</span>(self, idx):
        row = self.df.<span class="fn">iloc</span>[idx]
        img = Image.<span class="fn">open</span>(row[<span class="str">'path'</span>]).<span class="fn">convert</span>(<span class="str">'RGB'</span>).<span class="fn">resize</span>((<span class="num">224</span>, <span class="num">224</span>))
        meta = self.meta[row[<span class="str">'id'</span>]]
        <span class="kw">return</span> img, meta</code></pre>

<p>The rule of thumb: profile <code>__getitem__</code> with a Python time-it on a single call. If it takes more than a few hundred microseconds for in-memory work or more than a few milliseconds for I/O-bound work, you've got a hot path. Move the per-call constants to <code>__init__</code>.</p>

<div class="warn">
<strong>The pandas trap.</strong> <code>df.iloc[idx]</code> is fast. But <code>df[df.id == x]</code> is a full scan — O(n) per call. Ditto for <code>df.loc[df.col == x]</code>. If you're indexing by anything other than position, build a dict (<code>set_index().to_dict()</code>) in <code>__init__</code> for O(1) lookups. This single change has saved more training-time-wasted-on-data-loading hours than any other.
</div>

<h2>num_workers, prefetch_factor, persistent_workers</h2>

<p>Three knobs that work together. They control the producer side of the pipeline.</p>

<h3><code>num_workers</code>: how many subprocess loaders</h3>

<pre><code>loader = DataLoader(dataset, batch_size=<span class="num">64</span>, num_workers=<span class="num">4</span>)</code></pre>

<p>Sets the number of <em>subprocesses</em> that run <code>__getitem__</code>. <code>num_workers=0</code> means "do it in the main process" — synchronous, slow, but easy to debug.</p>

<p>Tuning: it's not "more workers = faster." Past a point, you saturate disk I/O or CPU and adding workers just adds overhead. Rule of thumb:</p>

<ul>
  <li><strong>I/O-bound dataset</strong> (loading images from disk): start with <code>num_workers = 4 to 8</code>, scale up to where I/O saturates.</li>
  <li><strong>CPU-bound dataset</strong> (heavy augmentation, decoding): <code>num_workers ≈ os.cpu_count()</code>, capped at 16 or so.</li>
  <li><strong>Memory-tight</strong>: each worker has a full copy of the Dataset object's state. If your <code>__init__</code> loaded a 5 GB lookup table, each worker has 5 GB. Cap workers accordingly.</li>
</ul>

<p>The right answer is usually 4-8. Most code-bases that I've seen with <code>num_workers=16</code> don't get faster than <code>num_workers=8</code> — they just use more RAM.</p>

<h3><code>prefetch_factor</code>: how far ahead each worker runs</h3>

<pre><code>loader = DataLoader(dataset, batch_size=<span class="num">64</span>, num_workers=<span class="num">4</span>, prefetch_factor=<span class="num">2</span>)
<span class="com"># Each worker keeps 2 batches "ready in advance" → 8 batches buffered total</span></code></pre>

<p>Default is 2. Increasing it (e.g., to 4) buffers more batches in advance, smoothing out variable-latency datasets at the cost of more RAM. Decreasing to 1 saves RAM but makes the GPU more sensitive to per-sample slowness.</p>

<h3><code>persistent_workers</code>: don't kill them between epochs</h3>

<pre><code>loader = DataLoader(dataset, batch_size=<span class="num">64</span>, num_workers=<span class="num">4</span>, persistent_workers=<span class="kw">True</span>)</code></pre>

<p>By default, workers spawn at the start of every epoch and die at the end. If your <code>Dataset.__init__</code> is expensive (e.g., loading a big index), this is a per-epoch tax. <code>persistent_workers=True</code> keeps the workers alive across epochs. <strong>Recommended for almost all cases</strong> — the only downside is that worker state persists, which is sometimes a debugging nuisance.</p>

<h2>collate_fn: how samples become batches</h2>

<p><code>collate_fn</code> takes a list of samples (whatever <code>__getitem__</code> returned) and produces a batch. The default is <code>default_collate</code>, which is smart enough for most cases:</p>

<pre><code><span class="com"># If __getitem__ returns (tensor, int), default_collate produces (batched_tensor, batched_int)</span>
<span class="com"># If __getitem__ returns dict, default_collate produces dict-of-batched-values</span>
<span class="com"># If shapes vary across samples — defaults BREAKS, you need a custom collate</span></code></pre>

<p>The classic case for a custom collate is variable-length sequences. Your dataset returns tokenized text of different lengths. The default collate tries to <code>torch.stack</code> them and fails because shapes don't match. You write a custom collate that pads:</p>

<pre><code><span class="kw">def</span> <span class="fn">pad_collate</span>(samples, pad_value=<span class="num">0</span>):
    <span class="com"># samples: list of (tokens, label), each with variable-length tokens</span>
    tokens, labels = <span class="fn">zip</span>(*samples)
    max_len = <span class="fn">max</span>(t.<span class="fn">size</span>(<span class="num">0</span>) <span class="kw">for</span> t <span class="kw">in</span> tokens)
    padded = torch.<span class="fn">full</span>((<span class="fn">len</span>(tokens), max_len), pad_value, dtype=tokens[<span class="num">0</span>].dtype)
    lengths = torch.<span class="fn">tensor</span>([t.<span class="fn">size</span>(<span class="num">0</span>) <span class="kw">for</span> t <span class="kw">in</span> tokens])
    <span class="kw">for</span> i, t <span class="kw">in</span> <span class="fn">enumerate</span>(tokens):
        padded[i, :t.<span class="fn">size</span>(<span class="num">0</span>)] = t
    <span class="kw">return</span> padded, torch.<span class="fn">stack</span>(labels), lengths

loader = <span class="fn">DataLoader</span>(dataset, batch_size=<span class="num">32</span>, collate_fn=pad_collate)</code></pre>

<p>Notes: (1) <code>pad_value</code> is dataset-specific — for transformers, often the tokenizer's pad token id; (2) returning <code>lengths</code> alongside is a useful idiom because downstream code can build a mask from it (M2's <code>positions[None, :] &lt; lengths[:, None]</code> trick); (3) for batches of dicts of tensors, write a generic dict-aware collate that handles each key.</p>

<h3>Bucketing: a smart collation pattern</h3>

<p>If your sequences vary wildly in length, padding everything to the longest sample wastes compute on padding. Better: sort/group similar-length samples into the same batch. The standard tool is a <code>BatchSampler</code> that groups indices by length, plus your collate doing only minor padding within each batch.</p>

<pre><code><span class="kw">class</span> <span class="ty">LengthBucketSampler</span>(torch.utils.data.Sampler):
    <span class="kw">def</span> <span class="fn">__init__</span>(self, lengths, batch_size):
        self.batches = []
        <span class="com"># sort by length, then chunk into batches</span>
        sorted_indices = <span class="fn">sorted</span>(<span class="fn">range</span>(<span class="fn">len</span>(lengths)), key=<span class="kw">lambda</span> i: lengths[i])
        <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="num">0</span>, <span class="fn">len</span>(sorted_indices), batch_size):
            self.batches.<span class="fn">append</span>(sorted_indices[i:i+batch_size])
        random.<span class="fn">shuffle</span>(self.batches)              <span class="com"># shuffle order, not contents</span>

    <span class="kw">def</span> <span class="fn">__iter__</span>(self):
        <span class="kw">return</span> <span class="fn">iter</span>(self.batches)

    <span class="kw">def</span> <span class="fn">__len__</span>(self):
        <span class="kw">return</span> <span class="fn">len</span>(self.batches)

loader = <span class="fn">DataLoader</span>(dataset, batch_sampler=bucket_sampler, collate_fn=pad_collate)</code></pre>

<p>Notice <code>batch_sampler</code>, not <code>sampler</code> — the batch sampler yields <em>lists of indices</em>, one per batch. Used in NLP training pipelines for huge throughput wins (often 2-3× faster than naive padding).</p>

<h2>Pin memory in the DataLoader</h2>

<p>From M3: pinned memory is page-locked CPU memory, allowing async DMA transfers to GPU. The DataLoader does this for you with one flag:</p>

<pre><code>loader = <span class="fn">DataLoader</span>(dataset, batch_size=<span class="num">64</span>, num_workers=<span class="num">4</span>, pin_memory=<span class="kw">True</span>)

<span class="com"># Then in your training loop:</span>
<span class="kw">for</span> x, y <span class="kw">in</span> loader:
    x = x.<span class="fn">to</span>(<span class="str">'cuda'</span>, non_blocking=<span class="kw">True</span>)         <span class="com"># async transfer, only useful with pin_memory</span>
    y = y.<span class="fn">to</span>(<span class="str">'cuda'</span>, non_blocking=<span class="kw">True</span>)
    out = <span class="fn">model</span>(x)
    ...</code></pre>

<p>Two things: (1) <code>pin_memory=True</code> alone does nothing — you also need <code>non_blocking=True</code> on the <code>.to()</code> for the async-ness to kick in; (2) it's a small but real win, typically 5-15% throughput on data-loading-bound training. Free if you can afford the (small) pinned-memory pool.</p>

<div class="warn">
<strong>Don't pin if you have nothing to gain.</strong> Pinning adds a copy step (allocate pinned memory, copy from worker's regular memory). If you're bottlenecked on compute and the GPU is at 95% utilization, pinning won't help and might marginally hurt. Profile first (Module 13).
</div>

<h2>Worker RNG: the silent reproducibility bug</h2>

<p>Here's a bug most teams ship without noticing. You set <code>torch.manual_seed(42)</code>, you run training twice, you get different results. What gives?</p>

<p>Each Worker is a child process. Child processes inherit a copy of the parent's RNG state at fork time, but PyTorch's default behavior depends on the start method (<code>fork</code> vs <code>spawn</code>) and the platform. On many setups, all workers end up with the <em>same RNG state</em>, producing the same random augmentations on the same indices — yes, that's a bug. On other setups, workers have <em>independent random state derived from system entropy</em>, which is fine for correctness but breaks reproducibility.</p>

<p>The fix: a <code>worker_init_fn</code> that explicitly seeds each worker:</p>

<pre><code><span class="kw">def</span> <span class="fn">seed_worker</span>(worker_id):
    <span class="com"># Each worker gets a deterministic but different seed</span>
    worker_seed = torch.<span class="fn">initial_seed</span>() % <span class="num">2</span>**<span class="num">32</span>
    np.random.<span class="fn">seed</span>(worker_seed)
    random.<span class="fn">seed</span>(worker_seed)

g = torch.<span class="fn">Generator</span>()
g.<span class="fn">manual_seed</span>(<span class="num">42</span>)

loader = <span class="fn">DataLoader</span>(
    dataset, batch_size=<span class="num">64</span>, num_workers=<span class="num">4</span>,
    worker_init_fn=seed_worker, generator=g,
)</code></pre>

<p>The <code>generator</code> argument seeds the loader's main-process RNG (used by samplers); <code>worker_init_fn</code> seeds each worker's auxiliary RNGs (NumPy, Python's <code>random</code>). With both set, each worker has a unique-but-deterministic seed derived from the loader's master seed plus the worker id.</p>

<p>Without this dance, your training is non-reproducible <em>even with all the seeds set in the main process</em>. Most teams ship without it and never notice; if you care about reproducibility (you should, for science and for debugging), set it.</p>

<h2>The DistributedSampler for multi-GPU</h2>

<p>For DDP training (Module 16), each rank needs its own <em>disjoint</em> slice of the dataset. <code>DistributedSampler</code> does this:</p>

<pre><code><span class="kw">from</span> torch.utils.data.distributed <span class="kw">import</span> DistributedSampler

sampler = <span class="fn">DistributedSampler</span>(dataset, num_replicas=world_size, rank=rank, shuffle=<span class="kw">True</span>)
loader = <span class="fn">DataLoader</span>(dataset, batch_size=<span class="num">64</span>, sampler=sampler, num_workers=<span class="num">4</span>)

<span class="com"># Important: call set_epoch each epoch so the shuffle pattern changes</span>
<span class="kw">for</span> epoch <span class="kw">in</span> <span class="fn">range</span>(num_epochs):
    sampler.<span class="fn">set_epoch</span>(epoch)             <span class="com"># otherwise every epoch sees the SAME shuffle</span>
    <span class="kw">for</span> x, y <span class="kw">in</span> loader:
        ...</code></pre>

<p>Two gotchas: (1) <code>shuffle</code> goes on the <em>sampler</em>, not the DataLoader; passing <code>shuffle=True</code> to DataLoader with a custom sampler errors out. (2) The <code>set_epoch</code> call is mandatory — otherwise every epoch shuffles the same way and the model sees the same effective ordering. Forgetting this is a real bug.</p>

<h2>Code Magnets: build the production training DataLoader</h2>

<p>You're setting up the data pipeline for transformer training. Variable-length sequences (so you need a pad collate), 4 workers, pinned memory, persistent workers across epochs, deterministic seeding.</p>

<div class="magnets">
<p>Arrange the magnets into a working DataLoader setup. Three are red herrings.</p>

<div class="magnet-pool">
  <span class="magnet">def seed_worker(worker_id):</span>
  <span class="magnet">    worker_seed = torch.initial_seed() % 2**32</span>
  <span class="magnet">    np.random.seed(worker_seed); random.seed(worker_seed)</span>
  <span class="magnet">g = torch.Generator(); g.manual_seed(42)</span>
  <span class="magnet">loader = DataLoader(dataset, batch_size=32, num_workers=4,</span>
  <span class="magnet">    pin_memory=True, persistent_workers=True,</span>
  <span class="magnet">    collate_fn=pad_collate,</span>
  <span class="magnet">    shuffle=True,</span>
  <span class="magnet">    worker_init_fn=seed_worker, generator=g)</span>
  <span class="magnet">    sampler=RandomSampler(dataset),</span>
  <span class="magnet">    prefetch_factor=16,</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">seed_worker</span>(worker_id):
    worker_seed = torch.<span class="fn">initial_seed</span>() % <span class="num">2</span>**<span class="num">32</span>
    np.random.<span class="fn">seed</span>(worker_seed); random.<span class="fn">seed</span>(worker_seed)

g = torch.<span class="fn">Generator</span>(); g.<span class="fn">manual_seed</span>(<span class="num">42</span>)

loader = <span class="fn">DataLoader</span>(dataset, batch_size=<span class="num">32</span>, num_workers=<span class="num">4</span>,
    pin_memory=<span class="kw">True</span>, persistent_workers=<span class="kw">True</span>,
    collate_fn=pad_collate,
    shuffle=<span class="kw">True</span>,
    worker_init_fn=seed_worker, generator=g)</code></pre>
<p>The traps:</p>
<ul>
  <li><code>sampler=RandomSampler(dataset)</code>: redundant with <code>shuffle=True</code>, and passing both errors. Pick one. For single-GPU shuffle, <code>shuffle=True</code> is enough.</li>
  <li><code>prefetch_factor=16</code>: way too high. Default 2 is almost always right; 4-8 if you have variable-latency I/O. 16 mostly just wastes RAM.</li>
</ul>
<p>The minimum reproducible production setup is exactly what's left: pin, persistent, collate, shuffle, seed. Everything else is tuning.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each DataLoader knob/concept to its real job.</p>

<div class="match-grid">
  <div class="header">Knob / concept</div>
  <div class="header">Real job</div>

  <div>num_workers</div>
  <div>A. Pads variable-length samples or otherwise turns a list of samples into a batch.</div>

  <div>prefetch_factor</div>
  <div>B. Decides which indices to fetch in what order.</div>

  <div>persistent_workers</div>
  <div>C. Number of subprocesses running __getitem__ in parallel.</div>

  <div>pin_memory</div>
  <div>D. Page-locks the batch in CPU memory so GPU DMA can read async.</div>

  <div>collate_fn</div>
  <div>E. How many batches each worker keeps "ready in advance" — buffer depth.</div>

  <div>Sampler</div>
  <div>F. Keeps workers alive across epochs (avoids re-running expensive __init__).</div>

  <div>worker_init_fn</div>
  <div>G. Per-worker hook to seed RNGs and ensure each worker has unique-but-deterministic state.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>num_workers</strong> → C<br>
<strong>prefetch_factor</strong> → E<br>
<strong>persistent_workers</strong> → F<br>
<strong>pin_memory</strong> → D<br>
<strong>collate_fn</strong> → A<br>
<strong>Sampler</strong> → B<br>
<strong>worker_init_fn</strong> → G
</p>
<p>The mental shortcut: <em>num_workers = parallelism, prefetch = buffer depth, persistent = lifetime across epochs, pin = async DMA, collate = batching, sampler = ordering, worker_init = RNG hygiene</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team's training loop is GPU-bound for the first 1000 steps and then suddenly becomes data-bound. What's the most likely cause?</p>
<details class="answer"><summary>show answer</summary>
<p>The first epoch's <code>__getitem__</code> calls populated the OS file cache, so I/O was fast. After warmup, something else changed — most likely the workers died (no <code>persistent_workers=True</code>) and the next epoch had to respawn them, re-loading the Dataset's <code>__init__</code> in each worker. If <code>__init__</code> reads big files or builds big indexes, this is a per-epoch tax. Set <code>persistent_workers=True</code> and the issue typically disappears.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> You're training a vision model and notice that with <code>num_workers=8</code> the GPU sits at ~70% utilization. Doubling to 16 doesn't help. What should you check?</p>
<details class="answer"><summary>show answer</summary>
<p>Likely a slow <code>__getitem__</code> bottleneck that's not parallel. Time a single <code>__getitem__</code> call. If it's slow because of CPU augmentation (e.g., heavy PIL transforms), more workers eventually saturate the CPU; consider GPU-side augmentation (kornia, NVIDIA DALI). If it's slow because of disk I/O, more workers don't help past disk bandwidth — switch to a faster format (webdataset, parquet) or pre-decode to a faster representation. The next-step diagnosis tool is <code>torch.profiler</code> from M13.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> Implement a custom collate that handles a dataset where <code>__getitem__</code> returns <code>{'image': tensor, 'caption': str, 'metadata': dict}</code>. Tensors should stack, strings should become a list, dicts should remain a list of dicts.</p>
<details class="answer"><summary>show answer</summary>
<pre><code><span class="kw">def</span> <span class="fn">multimodal_collate</span>(samples):
    images = torch.<span class="fn">stack</span>([s[<span class="str">'image'</span>] <span class="kw">for</span> s <span class="kw">in</span> samples])
    captions = [s[<span class="str">'caption'</span>] <span class="kw">for</span> s <span class="kw">in</span> samples]
    metadata = [s[<span class="str">'metadata'</span>] <span class="kw">for</span> s <span class="kw">in</span> samples]
    <span class="kw">return</span> {<span class="str">'image'</span>: images, <span class="str">'caption'</span>: captions, <span class="str">'metadata'</span>: metadata}</code></pre>
<p>The default collate would <em>try</em> to stack the captions and choke. The pattern: explicitly stack the things that should be tensors, and pass through the rest as lists. For more complex cases (variable-shape tensors), pad in the collate.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> A team's distributed training run is wasting 30% of every epoch on a synchronization stall. They use <code>DistributedSampler</code> but don't call <code>set_epoch</code>. What's happening, and is the stall related?</p>
<details class="answer"><summary>show answer</summary>
<p>The stall is probably unrelated (likely a slow <code>__getitem__</code> on some ranks or imbalanced data shards), but <em>not calling <code>set_epoch</code></em> is its own bug. Without it, every epoch's shuffle pattern is the same — the model sees the same effective sample order every epoch, defeating the point of shuffling. This doesn't directly cause stalls but makes training less effective. Always call <code>sampler.set_epoch(epoch)</code> at the start of each training epoch.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>A <code>DataLoader</code> is a <strong>producer-consumer pipeline</strong>: Sampler emits indices, Workers run <code>__getitem__</code>, collate_fn batches, pin_memory enables async GPU transfer.</li>
  <li><strong>Map-style datasets</strong> (<code>__len__</code> + <code>__getitem__</code>) are the default; <strong>iterable-style</strong> for streaming/infinite/sharded data.</li>
  <li><strong>The <code>__getitem__</code> trap</strong>: anything that doesn't depend on <code>idx</code> belongs in <code>__init__</code>, not in <code>__getitem__</code>. Especially: don't read big files or do full DataFrame scans per call.</li>
  <li>Index DataFrames by position (<code>iloc</code>) or by pre-built dict, not by value-equality scans.</li>
  <li><code>num_workers</code> = 4-8 for most setups. More than that usually saturates the next bottleneck (disk, CPU). Each worker has a full Dataset copy — memory matters.</li>
  <li><code>prefetch_factor=2</code> default is fine. Bump to 4-8 only for variable-latency I/O.</li>
  <li><code>persistent_workers=True</code> avoids the per-epoch worker respawn tax. Almost always the right setting.</li>
  <li><code>collate_fn</code> turns a list of samples into a batch. Default works for fixed shapes; write a custom one for padding, multimodal data, or anything weird.</li>
  <li><strong>Bucketing by length</strong> via <code>batch_sampler</code> is the standard NLP trick — 2-3× throughput by avoiding wasted padding compute.</li>
  <li><code>pin_memory=True</code> + <code>.to('cuda', non_blocking=True)</code> = async DMA hand-off. Free 5-15% in data-bound training.</li>
  <li><strong>Worker RNG is the silent reproducibility bug.</strong> Set <code>worker_init_fn</code> + <code>generator</code> on the DataLoader. Otherwise seeds in the main process don't propagate predictably.</li>
  <li><strong>DistributedSampler</strong>: each rank sees a disjoint slice. Always call <code>sampler.set_epoch(epoch)</code> before each epoch — otherwise shuffling is identical every epoch.</li>
  <li>The reflex: when training is slow, time <code>next(iter(loader))</code> in isolation. If that's slow, your data pipeline is bound; the diagnosis tree branches into worker count, <code>__getitem__</code> hot lines, and collate cost.</li>
</ul>
</div>

<p>Module 11 closes Part IV with the training loop itself: how to wire all of this together into something you can resume from a checkpoint, debug with overfit-a-batch, and run for a million steps without it falling apart.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">10</span>
  <span>Datasets & DataLoaders done right</span>
</div>
"""

emit("10_datasets_dataloaders", "Module 10 — Datasets & DataLoaders done right", BODY)
