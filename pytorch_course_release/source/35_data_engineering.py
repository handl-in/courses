#!/usr/bin/env python3
"""Module 35: Pretraining data engineering at scale — full HF vibe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_module import emit

BODY = r"""
<div class="module-header">
  <div class="module-tag">Part X · Module 35 · the data layer</div>
  <h1 class="module-title"><em>Pretraining data engineering</em> at scale</h1>
  <p class="module-sub">— from raw Common Crawl to a 15T-token training set: the pipeline, the deduplication, the quality classifiers, the domain mixing, and the streaming infrastructure that quietly does most of the work behind modern model quality</p>
</div>

<p>Architecture is largely settled. Optimization is largely settled. Post-training is converging fast. What still varies enormously between models — and what most teams underinvest in — is the <strong>data</strong>. Two models with the same architecture, same compute, same hyperparameters can differ by 10+ percentage points on benchmarks because one was trained on better data. Understanding why, and how, is the focus of this module.</p>

<p>Pretraining data engineering is also the most under-documented topic in the field. Architectures are described in papers; data pipelines are described in vague footnotes ("we filtered for quality and deduplicated"). What "filtered for quality" actually means in practice is folklore distributed across labs. This module collects the public knowledge — what FineWeb, Dolma, RedPajama, and similar open efforts have published — into an engineering playbook. <em>The engineering is unglamorous and crucial</em>; readers who internalize this module will have an edge most ML engineers don't.</p>

<div class="keyidea">
A modern pretraining run consumes on the order of 10-15 trillion high-quality tokens. The journey from Common Crawl (hundreds of petabytes of raw web data) to those final tokens involves <strong>order-of-magnitude rejections at each pipeline stage</strong>: language identification (~70% kept), quality filtering (~20-30% of the remaining), deduplication (~30-50% of what's left), contamination removal (~5-15% additional). The pipeline ends with maybe 1-3% of the input bytes, and that 1-3% is what the model actually trains on. <strong>Quality classifiers</strong> (small fastText models trained to predict "is this document high-quality educational content?") and <strong>MinHash-based near-duplicate detection</strong> are the engineering centerpieces. <strong>Domain mixing</strong> (what fraction of web vs code vs books vs synthetic) is then a separate optimization problem, often solved with DoReMi or empirical sweeps. <strong>Contamination</strong> — eval data leaking into training — is the silent quality killer; n-gram-based detection catches most but not all. Tokenizer training, streaming infrastructure, and synthetic data generation round out the system.
</div>

<h2>Two new faces — the data pipeline pair</h2>

<div class="character" style="--c: #6b5d4f;">
  <div class="avatar" style="background: #6b5d4f; color: #fff;">C</div>
  <div>
    <p class="who">Curator</p>
    <p class="name">"I throw away 99% of what comes in. What remains is what matters."</p>
    <p class="says">My job is to look at every document and decide: is this worth training on? Most of Common Crawl is boilerplate, SEO spam, machine-generated junk, and broken HTML. <em>The internet at scale is mostly noise</em>. I run language ID, then heuristic filters (length, perplexity ratios, special-character density), then a learned quality classifier — usually a tiny fastText model trained to recognize "educational content." Each stage is cheap; together they reduce raw bytes by 50-100×. Quality matters far more than quantity past a threshold: <strong>a billion well-curated tokens beats ten billion noisy ones</strong>. The frontier labs spend more compute on me than on training the model in some cases.</p>
  </div>
</div>

<div class="character" style="--c: #1f5f5b;">
  <div class="avatar" style="background: #1f5f5b; color: #fff;">D</div>
  <div>
    <p class="who">Deduper</p>
    <p class="name">"I find near-duplicates at scale. Two documents that share 80% of their 5-grams are the same document."</p>
    <p class="says">The internet has a lot of repetition: same article republished on different sites, same Wikipedia stub mirrored a hundred times, near-paraphrases. If I don't catch these, the model overweights what's repeated — and overweighting one source is a strange way to train. I use <strong>MinHash with LSH</strong>: hash each document's 5-grams, signature it, bucket by signature, find collisions. <em>Linear in document count, finds near-duplicates that differ by edits, runs on petabytes</em>. I also do exact substring dedup with suffix arrays for the long-string case. Removing duplicates typically improves perplexity by 5-15% at fixed compute — bigger than most architectural choices. <strong>Most teams underestimate me.</strong></p>
  </div>
</div>

<h2>The shape of the problem</h2>

<p>Pretraining data lives at a scale that breaks the assumptions you'd bring from "normal" data engineering. A few reference numbers as of 2026:</p>

<ul>
  <li><strong>Common Crawl</strong>: the canonical raw web corpus. Each monthly snapshot is ~250 TB of compressed WARC files; cumulative archives are 10+ PB.</li>
  <li><strong>Realistic training set sizes</strong>: GPT-3 (300B tokens), Llama-2 (2T tokens), Llama-3 (15T tokens), Llama-4 / GPT-4-class (rumored 20-30T+ tokens). Each token is roughly 4 characters of text — so 15T tokens is ~60 TB of UTF-8 text.</li>
  <li><strong>The funnel ratio</strong>: from raw Common Crawl bytes to training tokens, you're keeping somewhere in the range of 1-3% by volume. The other 97-99% is filtered out.</li>
</ul>

<p>This shape — petabytes of input, terabytes of output, multi-stage filtering — drives the engineering. You can't load it all into memory. You can't even loop over it on one machine in reasonable time. Every stage of the pipeline has to be parallel, fault-tolerant, and resumable. The pipeline itself is closer to a Spark / dataflow job than to a training script.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 380" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs>
    <marker id="arrF" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f5b"/>
    </marker>
  </defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">The pretraining data funnel: 100s of PB → 10s of TB</text>

  <!-- Funnel boxes, decreasing widths -->
  <g transform="translate(0, 50)">
    <!-- Stage 1: Common Crawl -->
    <rect x="40" y="0" width="660" height="40" fill="#fff8a8" stroke="#1a1612" stroke-width="1.5"/>
    <text x="370" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">① Common Crawl raw — ~250 TB / month, 10+ PB cumulative</text>
    <text x="370" y="32" text-anchor="middle" font-size="9" fill="#1a1612">WARC files, raw HTML, hundreds of billions of pages</text>
  </g>

  <text x="710" y="74" font-size="9" fill="#c1502e">↓ keep ~70% (English+major languages)</text>

  <g transform="translate(0, 95)">
    <rect x="80" y="0" width="580" height="40" fill="#d4ecc8" stroke="#1f5f5b" stroke-width="1.5"/>
    <text x="370" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">② After language ID + text extraction — ~70 PB</text>
    <text x="370" y="32" text-anchor="middle" font-size="9" fill="#1a1612">trafilatura/jusText extraction, fastText language ID, drop garbage HTML</text>
  </g>

  <text x="660" y="159" font-size="9" fill="#c1502e">↓ keep ~25-30% (heuristics + quality classifier)</text>

  <g transform="translate(0, 180)">
    <rect x="160" y="0" width="420" height="40" fill="#fff5d8" stroke="#d4a017" stroke-width="1.5"/>
    <text x="370" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">③ After quality filtering — ~20 PB</text>
    <text x="370" y="32" text-anchor="middle" font-size="9" fill="#1a1612">heuristics (length, repetition, special chars) + fastText quality classifier</text>
  </g>

  <text x="580" y="244" font-size="9" fill="#c1502e">↓ keep ~50-70% (after dedup)</text>

  <g transform="translate(0, 265)">
    <rect x="240" y="0" width="260" height="40" fill="#fcecec" stroke="#c1502e" stroke-width="1.5"/>
    <text x="370" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">④ After deduplication — ~12 PB</text>
    <text x="370" y="32" text-anchor="middle" font-size="9" fill="#1a1612">MinHash LSH near-dup; exact substring dedup</text>
  </g>

  <text x="500" y="329" font-size="9" fill="#c1502e">↓ keep ~85-95% (contamination filter)</text>

  <g transform="translate(0, 350)">
    <rect x="280" y="0" width="180" height="22" fill="#ffd5dc" stroke="#b85a6c" stroke-width="2"/>
    <text x="370" y="15" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">⑤ Final training tokens — ~10-15 TB</text>
  </g>
</svg>
</div>

<p>Read top-to-bottom: each stage cuts the data by some fraction, and the cuts compound. Final outputs are 1-3% of inputs — the rest is filtered out for being non-English, low-quality, duplicated, or contaminated with eval sets. <em>If you've never run this pipeline before, the loss rate is shocking on the first run.</em> The shock is the right reaction; the noise floor of the internet is genuinely that high.</p>

<h2>Stage 1: Ingestion and language identification</h2>

<p>The starting point is usually Common Crawl. Each WARC file is a sequence of HTTP responses with metadata. The first job is text extraction: pulling readable content out of HTML, dropping navigation/sidebars/footers/scripts. Standard tools:</p>

<ul>
  <li><strong>trafilatura</strong>: best-in-class HTML-to-text extraction; handles boilerplate removal well.</li>
  <li><strong>jusText</strong>: older but still common; faster but slightly worse extraction quality.</li>
  <li><strong>resiliparse</strong>: fast HTML parser, often used as a preprocessing layer before trafilatura.</li>
</ul>

<p>After extraction, language identification with <strong>fastText's lid.176 model</strong> (or CLD3): given a document's text, predict which of 176 languages it's in, with confidence. Documents below a confidence threshold (typically 0.65) are dropped — they're often very short, mixed-language, or garbage that the language ID can't classify. English-only training keeps documents classified as English with high confidence; multilingual training keeps documents in a configured set of languages with thresholds tuned per-language.</p>

<p>Engineering at this stage: <strong>this is where you write to a parallel file system</strong>. Ingestion typically writes to JSONL (one document per line) sharded across thousands of files, or to a columnar format like Parquet for analytics-friendly downstream stages. Each file is a few GB — large enough that random access is amortized, small enough that one worker can handle one file. <em>Choose the shard size to match your downstream parallelism budget.</em></p>

<h2>Stage 2: Heuristic quality filters</h2>

<p>Before any learned classifier, simple heuristics catch the most obvious junk. The Gopher paper's filters are still the standard reference:</p>

<div class="table-wrap">
<table>
<caption>Gopher-style quality heuristics — each filter typically rejects 1-10% of documents</caption>
<thead><tr><th>Filter</th><th>Rejects</th><th>Why</th></tr></thead>
<tbody>
<tr><td>Document &lt; 50 words</td><td>Stub pages, error messages</td><td>Too short to learn structure from</td></tr>
<tr><td>Document &gt; 100K words</td><td>Auto-generated logs, large dumps</td><td>Often machine output or scraped junk</td></tr>
<tr><td>Mean word length &lt; 3 or &gt; 10</td><td>Token soup, code in disguise as text</td><td>Real text has ~5 char/word</td></tr>
<tr><td>&gt; 10% words contain digits</td><td>Phone books, log files, data dumps</td><td>Not natural language</td></tr>
<tr><td>&gt; 30% lines end with ellipsis</td><td>Truncated previews, paywalled content</td><td>Reading those teaches the model to stop mid-sentence</td></tr>
<tr><td>&lt; 50% lines are sentences (no punct)</td><td>Lists, menus, broken extraction</td><td>Word salad, no syntactic structure</td></tr>
<tr><td>Repetition: top 2-gram &gt; 20% of text</td><td>Boilerplate ("Click here ... Click here ...")</td><td>Highly repetitive content over-trains specific sequences</td></tr>
<tr><td>Top stopword frequency &lt; 2%</td><td>Non-English in disguise, code, garbage</td><td>Real English has 7-10% stopwords</td></tr>
</tbody>
</table>
</div>

<p>Each filter is cheap (no model inference, just regex / counting). Implemented as a pipeline of predicates, applied document-by-document. Reject rates compound — you typically lose 30-50% of post-language-ID documents to heuristic filters alone, and that's before the quality classifier even runs.</p>

<h2>Stage 3: The learned quality classifier</h2>

<p>Heuristics catch obvious junk; a learned classifier catches subtler quality signals. The mainstream approach (FineWeb-Edu, DCLM, Llama-3): train a tiny classifier — a fastText model or a distilled BERT-base — to predict "is this document high-quality educational content?"</p>

<p>The bootstrap problem: how do you get labels in the first place? Two approaches in production use:</p>

<ol>
  <li><strong>Heuristic-bootstrapped</strong>: start with manually curated high-quality corpora (Wikipedia, ArXiv, books) as positive examples; sample from raw web data as negatives. Train a binary classifier. Apply to web data to score each document.</li>
  <li><strong>LLM-judged</strong> (FineWeb-Edu approach): use a strong LLM (Llama-3-70B-Instruct, GPT-4) to rate ~500K documents on educational quality (0-5 score). Use these ratings as training data for the small classifier. The small classifier then scores the full corpus at scale — it's millions of times cheaper than running the LLM on every document.</li>
</ol>

<p>Method 2 dominates in 2024-2026 because LLM-as-judge is now reliable enough for this use case, and the small classifier inherits the LLM's quality judgment at fastText speed. The training procedure is roughly:</p>

<pre><code><span class="kw">import</span> fasttext

<span class="com"># Step 1: Get LLM ratings (done once, expensively)</span>
<span class="com"># For ~500K-1M documents, use GPT-4 / Llama-3-70B to score 0-5</span>
<span class="com"># Save as: <score>\t<text> per line</span>

<span class="com"># Step 2: Train fastText classifier on these labels</span>
model = fasttext.<span class="fn">train_supervised</span>(
    input=<span class="str">"quality_labels.txt"</span>,
    epoch=<span class="num">5</span>,
    lr=<span class="num">0.5</span>,
    wordNgrams=<span class="num">2</span>,
    minCount=<span class="num">3</span>,
    dim=<span class="num">100</span>,
    loss=<span class="str">"hs"</span>,           <span class="com"># hierarchical softmax — fast for many classes</span>
)
<span class="com"># Trained model is ~few hundred MB, scores documents at 1-10ms each</span>
model.<span class="fn">save_model</span>(<span class="str">"quality_classifier.bin"</span>)

<span class="com"># Step 3: Score the full corpus and threshold</span>
<span class="kw">def</span> <span class="fn">score_document</span>(text, model, threshold=<span class="num">3.5</span>):
    label, prob = model.<span class="fn">predict</span>(text.<span class="fn">replace</span>(<span class="str">"\n"</span>, <span class="str">" "</span>))
    score = <span class="fn">int</span>(label[<span class="num">0</span>].<span class="fn">replace</span>(<span class="str">"__label__"</span>, <span class="str">""</span>))
    <span class="kw">return</span> score &gt;= threshold</code></pre>

<p>The threshold (typically 3 or 3.5 out of 5) is a hyperparameter tuned by ablating training runs at small scale. Higher threshold = less data, higher quality per document; lower threshold = more data, more noise. <strong>FineWeb-Edu's published results showed that threshold-3 classifier-filtered data outperformed unfiltered FineWeb at the same compute budget</strong> — quality won over quantity at the margin.</p>

<p>The 1-10ms per document inference cost matters when you're scoring 100B documents. At 5ms/doc with 1000 worker cores, scoring takes ~5 days; with 10K cores, ~12 hours. <em>Make the classifier as small as you can while preserving quality</em> — fastText's hash-based features and hierarchical softmax exist for this reason.</p>

<h2>Stage 4: Deduplication</h2>

<p>The web has many copies of many documents. Wikipedia is mirrored on hundreds of sites. News articles are republished, paraphrased, summarized. Boilerplate ("subscribe to our newsletter") appears identically across millions of pages. <strong>Without deduplication, the model spends a disproportionate share of its compute relearning the same content</strong> — and its outputs reflect that overweighting.</p>

<p>The published evidence is clear: aggressive dedup typically improves perplexity by 5-15% at fixed FLOPs. The Lee et al. paper, the SemDedup paper, and Llama-3's recipe all converge on this conclusion. <em>Dedup is one of the highest-leverage data interventions you can run.</em></p>

<p>Three flavors of dedup, each catching different patterns:</p>

<ul>
  <li><strong>Exact dedup</strong>: hash each document; keep one copy per hash. Catches verbatim repeats. Cheap (one hash per doc) but misses anything with edits.</li>
  <li><strong>Near-duplicate detection (MinHash LSH)</strong>: documents that share most of their 5-grams or shingles are flagged as duplicates. Catches republished articles with minor reformatting. Typical recall: documents with Jaccard similarity ≥ 0.8 are reliably caught.</li>
  <li><strong>Substring dedup</strong>: find long substrings (e.g., 50+ tokens) that appear multiple times across the corpus; remove all but one occurrence. Catches boilerplate and shared paragraphs. Done with suffix array on tokenized corpus.</li>
</ul>

<h3>MinHash LSH in real terms</h3>

<p>The MinHash LSH approach is the workhorse. The intuition: instead of comparing every document to every other (O(N²) — impossible at scale), build a signature for each document such that <em>similar documents have similar signatures</em>. Group documents by signature; near-duplicates cluster together.</p>

<pre><code><span class="kw">def</span> <span class="fn">minhash_signature</span>(text, num_hashes=<span class="num">128</span>, ngram_size=<span class="num">5</span>):
    <span class="com"># Step 1: extract n-grams (shingles).</span>
    tokens = text.<span class="fn">split</span>()
    ngrams = <span class="fn">set</span>(<span class="str">" "</span>.<span class="fn">join</span>(tokens[i:i+ngram_size])
                 <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(tokens) - ngram_size + <span class="num">1</span>))

    <span class="com"># Step 2: compute MinHash signature.</span>
    <span class="com"># For each of num_hashes hash functions, find min hash value across all n-grams.</span>
    signature = []
    <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(num_hashes):
        seed = <span class="num">12345</span> + i
        min_val = <span class="fn">min</span>(<span class="fn">murmurhash3</span>(ngram, seed=seed) <span class="kw">for</span> ngram <span class="kw">in</span> ngrams)
        signature.<span class="fn">append</span>(min_val)
    <span class="kw">return</span> signature

<span class="kw">def</span> <span class="fn">lsh_buckets</span>(signature, bands=<span class="num">16</span>, rows_per_band=<span class="num">8</span>):
    <span class="com"># Split signature into bands; documents that match in any band → candidates.</span>
    <span class="fn">assert</span> bands * rows_per_band == <span class="fn">len</span>(signature)  <span class="com"># 16 * 8 = 128</span>
    buckets = []
    <span class="kw">for</span> b <span class="kw">in</span> <span class="fn">range</span>(bands):
        band_sig = <span class="fn">tuple</span>(signature[b*rows_per_band : (b+<span class="num">1</span>)*rows_per_band])
        bucket_id = (b, <span class="fn">hash</span>(band_sig))
        buckets.<span class="fn">append</span>(bucket_id)
    <span class="kw">return</span> buckets

<span class="com"># Pipeline: for each doc, compute signature + bucket IDs.</span>
<span class="com"># Group docs by bucket ID. Within each bucket, mark all but one as duplicates.</span></code></pre>

<p>The math: with bands=16 and rows_per_band=8, the probability that two documents with Jaccard similarity <code>s</code> match in at least one band is <code>1 - (1 - s^8)^16</code>. This S-curve has a sharp threshold around <code>s ≈ 0.8</code>: documents above 80% similarity almost always match; documents below 70% almost never do. <em>Tune bands and rows_per_band to set your similarity threshold</em>.</p>

<p>Engineering at scale: signature computation is embarrassingly parallel (per-document). Bucket grouping is a distributed shuffle. Within each bucket, deduplication picks one canonical document (typically the longest, or the highest-quality-classifier-scored). Production implementations: HuggingFace's <code>datatrove</code>, the <code>text-dedup</code> package, custom Spark/Ray jobs at frontier labs.</p>

<h3>What dedup misses</h3>

<p>MinHash LSH catches near-duplicates with high textual overlap. It misses:</p>

<ul>
  <li><strong>Heavy paraphrases</strong>: same content, totally different wording. Two articles describing the same news event in different journalistic styles. SemDedup uses embedding similarity to catch these.</li>
  <li><strong>Translations</strong>: same content in different languages.</li>
  <li><strong>Summaries vs full text</strong>: one document is a summary of another. Different lengths break shingle overlap.</li>
</ul>

<p>For these, semantic deduplication (compute sentence embeddings, cluster by cosine similarity) is the next-level technique. More expensive: requires an embedding model and an ANN index over billions of embeddings. Used by some frontier labs; not yet common in open recipes.</p>

<h2>Stage 5: Contamination filtering</h2>

<p>Eval sets are public. The internet has copies of GSM8K, MATH, MMLU, HumanEval. If those copies end up in your training data, your eval scores are inflated — the model is memorizing, not generalizing. <strong>Contamination is the silent quality killer</strong>: training loss looks normal, eval looks great, but the eval is meaningless because the model has seen the answers.</p>

<p>The published numbers are sobering. Multiple analyses have found that 5-15% of canonical eval sets are present in raw web crawls. Without explicit filtering, a non-trivial fraction of your reported eval performance is contamination. The Llama-3 paper, GPT-4 technical report, and similar all describe contamination filtering steps and report contamination percentages.</p>

<p>The standard approach: <strong>n-gram overlap detection</strong>. For each eval example, extract its n-grams (typically 13-grams or 50-token spans). Hash them. For each training document, check if it contains any eval n-gram. If the overlap exceeds a threshold (e.g., 1 matching 50-gram = 0.1% overlap), reject the document.</p>

<pre><code><span class="kw">def</span> <span class="fn">build_eval_ngram_set</span>(eval_examples, n=<span class="num">13</span>):
    <span class="com"># Build a set of all n-grams from all eval examples (use Bloom filter for memory)</span>
    all_ngrams = <span class="fn">set</span>()
    <span class="kw">for</span> ex <span class="kw">in</span> eval_examples:
        text = ex[<span class="str">"question"</span>] + <span class="str">" "</span> + ex[<span class="str">"answer"</span>]
        tokens = text.<span class="fn">split</span>()
        <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(tokens) - n + <span class="num">1</span>):
            ngram = <span class="str">" "</span>.<span class="fn">join</span>(tokens[i:i+n])
            all_ngrams.<span class="fn">add</span>(<span class="fn">hash</span>(ngram))
    <span class="kw">return</span> all_ngrams

<span class="kw">def</span> <span class="fn">contamination_score</span>(doc, eval_ngrams, n=<span class="num">13</span>):
    tokens = doc.<span class="fn">split</span>()
    <span class="kw">if</span> <span class="fn">len</span>(tokens) &lt; n:
        <span class="kw">return</span> <span class="num">0.0</span>
    matches = <span class="fn">sum</span>(
        <span class="num">1</span> <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(tokens) - n + <span class="num">1</span>)
        <span class="kw">if</span> <span class="fn">hash</span>(<span class="str">" "</span>.<span class="fn">join</span>(tokens[i:i+n])) <span class="kw">in</span> eval_ngrams
    )
    <span class="kw">return</span> matches / <span class="fn">max</span>(<span class="fn">len</span>(tokens) - n + <span class="num">1</span>, <span class="num">1</span>)</code></pre>

<p>For very large training sets, even building the full n-gram set in memory is impractical. Bloom filters (with controlled false-positive rate) and bit-set hashing replace the Python set. The pipeline pattern: build the eval n-gram Bloom filter once (a few GB), then check each training document against it as a streaming filter.</p>

<p>What n-gram detection misses: <strong>paraphrased contamination</strong>. If MMLU's question is "What is the capital of France?" and the training data has "Of all the cities in France, the capital is Paris," n-gram overlap is low but the model still learns the answer. Catching paraphrased contamination requires semantic similarity, which is more expensive and less standard. <em>This is an open problem</em>; published frontier-model contamination reports almost certainly underreport because they only measure n-gram contamination.</p>

<h2>Domain mixing</h2>

<p>You now have a clean, deduplicated, uncontaminated corpus. But it's a <em>web-heavy</em> corpus — Common Crawl is mostly forums, blogs, news, and SEO content. To train a high-quality model, you want to mix in other domains: code (GitHub), books (project Gutenberg / commercial sources), academic papers (ArXiv), reference material (Wikipedia, StackExchange), perhaps math-specific data, perhaps multilingual data.</p>

<p>The question becomes: <em>what fractions of each?</em> Too much code hurts general performance; too little code hurts code-specific tasks. Too much Wikipedia overweights factual recall but underweights conversational fluency. The optimization landscape is real and matters.</p>

<div class="tensor-vis" style="margin: 32px 0; text-align: center;">
<svg viewBox="0 0 740 320" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto; font-family: 'IBM Plex Mono', monospace;">
  <defs></defs>
  <text x="370" y="22" font-size="14" font-weight="700" fill="#1a1612" text-anchor="middle">Domain mixing tradeoff: quality on different categories vs code-fraction</text>

  <!-- Axes -->
  <line x1="80" y1="280" x2="700" y2="280" stroke="#1a1612" stroke-width="1.5"/>
  <line x1="80" y1="280" x2="80" y2="60" stroke="#1a1612" stroke-width="1.5"/>
  <text x="390" y="308" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612">code fraction in training mix (%)</text>
  <text x="50" y="170" text-anchor="middle" font-size="11" font-weight="700" fill="#1a1612" transform="rotate(-90 50 170)">eval performance</text>

  <!-- X-axis ticks -->
  <text x="80" y="295" text-anchor="middle" font-size="9" fill="#6b5d4f">0%</text>
  <text x="220" y="295" text-anchor="middle" font-size="9" fill="#6b5d4f">10%</text>
  <text x="360" y="295" text-anchor="middle" font-size="9" fill="#6b5d4f">25%</text>
  <text x="500" y="295" text-anchor="middle" font-size="9" fill="#6b5d4f">50%</text>
  <text x="640" y="295" text-anchor="middle" font-size="9" fill="#6b5d4f">100%</text>

  <!-- General/chat performance: peaks low, declines steeply -->
  <path d="M 80 130 Q 150 115 220 105 Q 290 110 360 130 Q 430 165 500 200 Q 570 235 640 260" 
        stroke="#1f5f5b" stroke-width="2" fill="none"/>
  <text x="710" y="103" font-size="10" fill="#1f5f5b">general/chat</text>

  <!-- Code performance: starts low, climbs sharply -->
  <path d="M 80 270 Q 150 230 220 180 Q 290 140 360 110 Q 430 90 500 80 Q 570 75 640 73" 
        stroke="#c1502e" stroke-width="2" fill="none"/>
  <text x="710" y="73" font-size="10" fill="#c1502e">code</text>

  <!-- Math: peaks at moderate code fraction -->
  <path d="M 80 200 Q 150 165 220 145 Q 290 130 360 125 Q 430 130 500 145 Q 570 170 640 200" 
        stroke="#d4a017" stroke-width="2" fill="none"/>
  <text x="710" y="125" font-size="10" fill="#d4a017">math</text>

  <!-- Composite (weighted average): peaks near 15-20% -->
  <path d="M 80 180 Q 150 155 220 140 Q 290 135 360 145 Q 430 165 500 180 Q 570 200 640 220" 
        stroke="#b85a6c" stroke-width="3" stroke-dasharray="0" fill="none"/>
  <text x="710" y="143" font-size="10" font-weight="700" fill="#b85a6c">composite ★</text>

  <!-- Annotation -->
  <line x1="220" y1="60" x2="220" y2="280" stroke="#6b5d4f" stroke-width="0.5" stroke-dasharray="3 2"/>
  <text x="220" y="55" text-anchor="middle" font-size="10" font-weight="700" fill="#1a1612">~15% sweet spot</text>

  <text x="370" y="335" font-family="'Caveat', cursive" font-size="20" fill="#1a1612" text-anchor="middle">code teaches general reasoning too — but past a point, you lose conversational fluency</text>
</svg>
</div>

<p>Three patterns to internalize from the chart:</p>

<ol>
  <li><strong>Code helps general performance</strong> at low fractions (0% → 15% improves general/chat eval somewhat). The hypothesis: code teaches structured reasoning that transfers to non-code tasks.</li>
  <li><strong>Code hurts general past a point</strong>: at 50%+ code, general performance degrades significantly. The model becomes a programmer who can barely chat.</li>
  <li><strong>Optimal mixing depends on the eval distribution</strong>: if you weight code-heavy benchmarks, more code wins. If you weight chat-heavy benchmarks, less code wins. The composite weighted average peaks somewhere in the 10-25% code range for typical product evals.</li>
</ol>

<h3>DoReMi: principled domain mixing</h3>

<p>Sweep-based domain optimization (try N ratios, pick the best) is expensive: each ratio requires a full small-scale training run for ablation. The DoReMi paper proposes a more efficient method: <strong>train a small model with multiple-domain weights as a hyperparameter, then use the model's loss on each domain to update the weights via group DRO (distributionally robust optimization)</strong>.</p>

<p>The mechanics: maintain weights <code>w_d</code> for each domain <code>d</code>. After each training step, compute the loss on each domain's held-out validation set; the domain with the highest loss-to-reference-loss ratio gets its weight increased. Over training, the weights converge to a mixture that's "hard for the model to learn" — usually corresponding to underrepresented or harder domains.</p>

<p>DoReMi is one published recipe; in practice, frontier labs use combinations of: small-scale sweep ablations, DoReMi-style algorithmic optimization, and human judgment about what categories to prioritize. <em>The choice of mixing ratios is more art than science</em>; there's no universal answer because different products have different eval distributions.</p>

<h3>Other knobs in the mix</h3>

<ul>
  <li><strong>Curriculum scheduling</strong>: change the mix during training. Some labs put more web data early (general knowledge), more code/math/papers late (sharper skills). Not universally adopted.</li>
  <li><strong>Quality up-weighting</strong>: oversample high-quality-classifier-scored data within each domain. Higher-scored docs appear in 2-3 epochs; lower-scored in 1.</li>
  <li><strong>Repetition</strong>: at 15T tokens you can't easily get more unique high-quality data, so high-quality subsets get repeated 2-4 epochs. Empirically this works up to a point; beyond ~5 epochs, returns diminish.</li>
</ul>

<h2>Tokenizer training</h2>

<p>The tokenizer determines how text becomes integer sequences. It's trained <em>once</em> on a representative sample of the final corpus, then frozen for the entire pretraining run. Bad tokenizer choices can permanently hurt model quality in subtle ways.</p>

<p>The standard algorithms:</p>

<ul>
  <li><strong>BPE (Byte-Pair Encoding)</strong>: greedily merge most-frequent token pairs. Used by GPT, Llama, Mistral. Most common.</li>
  <li><strong>Unigram</strong>: probabilistic model where the tokenizer is trained to maximize likelihood of the corpus given a token vocabulary. Used by SentencePiece's default. Slightly different segmentation behavior.</li>
  <li><strong>WordPiece</strong>: similar to BPE but uses likelihood ratios to select merges. Used by BERT.</li>
</ul>

<p>Vocab size tradeoffs:</p>

<ul>
  <li><strong>Small vocab (~32K)</strong>: each token represents less; sequences are longer; models need more context per text amount. Used by older models (Llama-1, Llama-2 at 32K).</li>
  <li><strong>Medium vocab (~128K)</strong>: balanced. Used by Llama-3 (128K), Mistral.</li>
  <li><strong>Large vocab (~256K+)</strong>: each token covers more text; sequences are shorter for a given text length; better for long-context efficiency. Used by Gemma-2 (256K), some multilingual models.</li>
</ul>

<p>Larger vocab pushes the cost from sequence length to embedding/lm_head matrix size. With tied embeddings (M29), a vocab of 256K with d=4096 adds ~1B parameters of vocab embedding — substantial. <em>For multilingual models or long-context-heavy applications, large vocab pays off; for English-only chat, 128K is the sweet spot.</em></p>

<p>Tokenizer training itself is fast — a few hours on a single machine for a few-GB sample of the corpus. The sample matters: train the tokenizer on a sample that reflects your final mix (e.g., the 15% code, 5% math, 80% web). If the tokenizer was trained on web-only and you train on 30% code, code tokens are inefficient (long sequences for short code).</p>

<h2>Streaming infrastructure</h2>

<p>You can't fit a 15TB tokenized corpus on a single machine. You can't fit it on the local disk of every node in a 1000-node cluster either. The standard pattern: <strong>store the corpus on object storage (S3, GCS), stream it during training</strong>.</p>

<p>The format and library landscape:</p>

<ul>
  <li><strong>WebDataset</strong>: tar files of (key, value) pairs. Standard for vision-text mixed corpora; works for text-only.</li>
  <li><strong>Mosaic Streaming (MDS)</strong>: optimized for LLM pretraining. Sharded binary format with deterministic shuffle ordering.</li>
  <li><strong>HuggingFace datasets streaming mode</strong>: streams from Parquet on object storage; convenient but slower than MDS.</li>
  <li><strong>Custom binary formats</strong>: most frontier labs roll their own — typically .bin files of pre-tokenized uint32 sequences with index files for random access.</li>
</ul>

<p>The key engineering concern is <strong>determinism under shuffling</strong>. With many ranks, many epochs, and async I/O, you need each rank to see a different, non-overlapping shard at each step, AND the assignment must be reproducible across resumptions and across re-runs. MDS handles this via a deterministic shuffle algorithm seeded on (epoch, global_step). Rolling your own requires care — one of the easiest ways to silently corrupt training is to have ranks see overlapping data.</p>

<p>Another concern: <strong>throughput-balanced reads</strong>. Some shards have shorter sequences (faster to consume), some longer. If shards aren't balanced, fast workers idle waiting for slow ones. Sequence-packing (binning sequences to fill fixed-length contexts) addresses this and is now standard in production training.</p>

<h2>Synthetic data</h2>

<p>The newest frontier in pretraining data engineering: <strong>generate training data with another LLM</strong>. The Cosmopedia paper (Hugging Face) and Phi family (Microsoft) demonstrated that synthetic textbooks, instruction data, and explanations can substantially boost model quality, especially at smaller scales.</p>

<p>The recipe typically:</p>

<ol>
  <li>Take a strong base model (frontier-quality) as the teacher.</li>
  <li>Use it to generate text in specific styles: textbook explanations, code with comments, step-by-step problem solutions, conversational data.</li>
  <li>Apply quality filters (the same ones used on real data).</li>
  <li>Mix the synthetic data into the pretraining corpus, typically at 5-25% by token count.</li>
</ol>

<p>Why does this work? Synthetic data can be more controlled than web data: every "textbook" can be accurate, well-structured, and pedagogically clear. The teacher model's knowledge is distilled into the student model via the synthetic data, in a more efficient form than raw web crawls would provide.</p>

<p>The risks:</p>

<ul>
  <li><strong>Mode collapse</strong>: if the synthetic data is too narrow stylistically, the student inherits that narrowness. Diverse generation prompts and topic coverage matter.</li>
  <li><strong>Hallucinations propagating</strong>: if the teacher hallucinates, the student learns hallucinated facts as if they were real.</li>
  <li><strong>Recursive collapse</strong>: training on outputs of models trained on outputs leads to quality degradation. Most published synthetic-data recipes are careful to use real-data anchors and not chain synthetic generations.</li>
</ul>

<p>Synthetic data is one of the few areas where <em>more compute spent on data generation</em> can substitute for <em>more compute spent on training</em>. For smaller models (1B-10B), well-designed synthetic data can give multi-percentage-point benchmark gains at fixed training compute.</p>

<h2>Putting it all together: a 15T-token training set</h2>

<p>The numbers behind a real frontier-quality 15T-token pretraining corpus, very approximately:</p>

<div class="table-wrap">
<table>
<caption>Approximate composition of a 15T-token corpus (illustrative; specific labs vary)</caption>
<thead><tr><th>Source</th><th>Token fraction</th><th>Engineering effort</th></tr></thead>
<tbody>
<tr><td>Filtered Common Crawl (web)</td><td>~50-60%</td><td>The bulk of the pipeline; quality classifier + dedup dominate</td></tr>
<tr><td>Books and academic papers</td><td>~10-15%</td><td>Licensing, OCR for scanned books, format normalization</td></tr>
<tr><td>Code (GitHub, language-specific)</td><td>~10-20%</td><td>License filtering, deduplication (code is highly duplicated), language balance</td></tr>
<tr><td>Wikipedia and reference</td><td>~3-5%</td><td>High-quality, repeated 2-3 epochs typically</td></tr>
<tr><td>Math-specific data</td><td>~2-5%</td><td>Curated math forums, ArXiv math, generated problem sets</td></tr>
<tr><td>Synthetic data</td><td>~5-15%</td><td>Generation pipeline + quality filters</td></tr>
<tr><td>Multilingual non-English</td><td>variable</td><td>Per-language pipeline replication</td></tr>
</tbody>
</table>
</div>

<p>The total engineering effort is significant. Public data efforts (Dolma, FineWeb, DCLM, RedPajama-2) have published infrastructure that handles parts of this; private labs have substantially more elaborate pipelines. The compute cost of running the data pipeline can be 10-30% of the training compute itself — and that's just to prepare the data, not train the model.</p>

<p><strong>The investment pays off</strong>: small differences in data curation produce large differences in final model quality. Reproducing a competitive frontier model with public weights is now feasible if you have the compute; reproducing one with high quality without the data infrastructure to match is not.</p>

<div class="ndq">
<h4>About data engineering</h4>

<p class="q">How important is the order of pipeline stages? Could I dedupe before quality-filter?</p>
<p class="a">Order matters in two ways. First, <strong>compute cost</strong>: the early stages (language ID, heuristics) are cheapest per document; they should run first to reduce the document count for expensive stages (classifier, dedup). Running classifier on 100 PB and then dedup-ing 25 PB of survivors is much cheaper than dedup-ing 100 PB and then classifying 70 PB. Second, <strong>signal quality</strong>: the quality classifier was likely trained on already-cleaned data. Running it on raw HTML-y output can produce confused scores. Standard order — language ID, heuristics, classifier, then dedup — minimizes both compute waste and signal corruption. Dedup after classifier means you keep the highest-classifier-scored doc when there are duplicates, which is what you want.</p>

<p class="q">If 99% of Common Crawl is junk, why not just use Wikipedia + ArXiv + books?</p>
<p class="a">Two reasons. (1) <strong>Volume</strong>: even with all of Wikipedia + ArXiv + book corpora, you have maybe 100-300B tokens. A frontier model wants 10T+. The web is where the volume comes from. (2) <strong>Distribution</strong>: Wikipedia is encyclopedic, ArXiv is academic, books are narrative. None of them prepare a model for chat-style language, casual writing, or domain-specific jargon that appears in product use. Web data — even after aggressive filtering — provides the distributional breadth that makes a model feel "natural" across topics. The recipe is: high-quality curated sources for facts and structure; web-derived high-quality content for breadth and style.</p>

<p class="q">How do I know if my deduplication is working?</p>
<p class="a">Three diagnostics. (1) <strong>Reduction ratio</strong>: typical post-dedup reduction is 30-50% of post-classifier data. If you're seeing 5-10%, your dedup is too lenient (raise similarity threshold by lowering bands, or use larger n-grams). If you're seeing 80%+, it's too aggressive (reverse). (2) <strong>Manual sampling</strong>: take 100 random surviving documents; check if any are obviously near-duplicates. If yes, lowering the threshold further. (3) <strong>Train ablation</strong>: train a small model on dedup'd vs not-dedup'd data of the same final size; compare perplexity on a held-out validation set. Dedup'd should win by 5-15%; if it doesn't, your dedup isn't catching the right duplicates. Most production teams do all three.</p>

<p class="q">Can I just use a published corpus like FineWeb instead of building my own pipeline?</p>
<p class="a">Yes, often. FineWeb-Edu, DCLM-baseline, RedPajama-2, Dolma — these are published, free, high-quality, and well-deduplicated. For most teams, starting with one of these and adding domain-specific data (your own code corpus, your own books, etc.) is the right move. You only need a custom pipeline if (1) you need data sources the public efforts don't cover, (2) you're at frontier scale and the public data isn't enough, or (3) you have specific quality criteria that differ from the public efforts. <em>For research and most production work, the published corpora are an excellent starting point</em>; building from scratch is a substantial engineering investment that's hard to justify when high-quality public alternatives exist.</p>

<p class="q">How do I detect if my eval is contaminated?</p>
<p class="a">Three approaches. (1) <strong>N-gram overlap</strong>: as covered, check if eval n-grams appear in training. Standard. (2) <strong>Memorization probes</strong>: prompt the model with the first half of an eval question; see if it produces the second half verbatim. If yes, contamination. (3) <strong>Comparison to held-out variants</strong>: take the same task with different phrasings or freshly-generated examples (e.g., GSM8K-Platinum is a contamination-resistant variant). Large performance gap between original and variant indicates contamination. <em>Any new eval should be checked for contamination as part of release</em>; old evals with high contamination should be deprecated. The community is moving toward "live" evals (continuously generated, never published) for this reason.</p>

<p class="q">What's the right vocab size for my model?</p>
<p class="a">Three rules of thumb. (1) <strong>For English-only chat</strong>: 128K is the sweet spot. Llama-3, Mistral, most modern English models use this range. (2) <strong>For multilingual</strong>: 256K+ for good cross-lingual coverage. The vocab needs slots for non-English tokens to be efficient. (3) <strong>For very small models</strong> (&lt;1B): smaller vocab (32K-64K) reduces the embedding parameter count, which is proportionally large at small model sizes. Beyond these heuristics, you can sweep at small scale — the cost of training a tokenizer is negligible compared to model training. <em>Don't over-optimize</em>: vocab size matters but is rarely the bottleneck in real applications.</p>
</div>

<h2>Code Magnets: implement MinHash LSH bucketing</h2>

<p>You're computing LSH buckets from a MinHash signature for deduplication. Three magnets are wrong choices.</p>

<div class="magnets">
<p>Arrange the magnets to compute LSH buckets from the signature.</p>

<div class="magnet-pool">
  <span class="magnet">def lsh_buckets(signature, bands=16, rows_per_band=8):</span>
  <span class="magnet">    assert bands * rows_per_band == len(signature)</span>
  <span class="magnet">    assert bands + rows_per_band == len(signature)</span>
  <span class="magnet">    buckets = []</span>
  <span class="magnet">    for b in range(bands):</span>
  <span class="magnet">        band_sig = tuple(signature[b*rows_per_band : (b+1)*rows_per_band])</span>
  <span class="magnet">        band_sig = tuple(signature[b : b+rows_per_band])</span>
  <span class="magnet">        bucket_id = (b, hash(band_sig))</span>
  <span class="magnet">        bucket_id = hash(band_sig)</span>
  <span class="magnet">        buckets.append(bucket_id)</span>
  <span class="magnet">    return buckets</span>
</div>

<details class="answer"><summary>show solution</summary>
<pre><code><span class="kw">def</span> <span class="fn">lsh_buckets</span>(signature, bands=<span class="num">16</span>, rows_per_band=<span class="num">8</span>):
    <span class="fn">assert</span> bands * rows_per_band == <span class="fn">len</span>(signature)
    buckets = []
    <span class="kw">for</span> b <span class="kw">in</span> <span class="fn">range</span>(bands):
        band_sig = <span class="fn">tuple</span>(signature[b*rows_per_band : (b+<span class="num">1</span>)*rows_per_band])
        bucket_id = (b, <span class="fn">hash</span>(band_sig))
        buckets.<span class="fn">append</span>(bucket_id)
    <span class="kw">return</span> buckets</code></pre>
<p>The traps:</p>
<ul>
  <li><code>assert bands + rows_per_band == len(signature)</code>: addition instead of multiplication. The signature must split exactly into <code>bands</code> non-overlapping bands of <code>rows_per_band</code> each. With 128-element signature, bands=16, rows_per_band=8, you need 16 × 8 = 128. Adding gives 24, completely wrong, and the slicing logic would silently produce overlapping or partial bands.</li>
  <li><code>band_sig = tuple(signature[b : b+rows_per_band])</code>: wrong slicing. This produces overlapping windows starting at positions 0, 1, 2, ... — bands b=0 and b=1 share most of their values. The non-overlapping form is <code>signature[b*rows_per_band : (b+1)*rows_per_band]</code> — band 0 is positions 0-7, band 1 is 8-15, etc. With overlapping bands, similar documents collide far more often than they should, ruining the dedup precision.</li>
  <li><code>bucket_id = hash(band_sig)</code>: missing the band index. Two documents that match in band 5 should share a bucket, but two documents that happen to have the same hash in <em>different</em> bands shouldn't be confused. The bucket ID is <code>(band_index, band_hash)</code> — the tuple ensures bands don't accidentally collide. Without it, you'd get false-positive duplicate matches between unrelated bands.</li>
</ul>
<p>The pattern: <strong>signature splits into non-overlapping bands → each band hashed → bucket ID is (band_index, band_hash) → documents in same bucket are candidates</strong>. The math of LSH (the S-curve of similarity vs match probability) only works with this exact partitioning; deviations silently break the algorithm.</p>
</details>
</div>

<h2>Who does what?</h2>

<div class="matching">
<p class="intro">Match each data engineering concept to its real role.</p>

<div class="match-grid">
  <div class="header">Concept</div>
  <div class="header">Real role</div>

  <div>Common Crawl</div>
  <div>A. Petabyte-scale raw web corpus; the starting point for pretraining data.</div>

  <div>Quality classifier (fastText)</div>
  <div>B. Tiny model trained to score documents on educational quality; runs on the full corpus cheaply.</div>

  <div>MinHash LSH</div>
  <div>C. Near-duplicate detection at scale: signature each document, bucket by signature, find collisions.</div>

  <div>N-gram contamination filter</div>
  <div>D. Detects eval set leakage by checking training docs for eval n-grams (typically 13-grams).</div>

  <div>Domain mixing / DoReMi</div>
  <div>E. Algorithmic optimization of the per-domain weights in the training mix.</div>

  <div>Streaming MDS shards</div>
  <div>F. Petabyte-scale corpus served from object storage with deterministic shuffle.</div>

  <div>Synthetic textbook data</div>
  <div>G. LLM-generated pretraining data (Cosmopedia, Phi); 5-25% of corpus typically.</div>
</div>

<details class="answer"><summary>show solution</summary>
<p>
<strong>Common Crawl</strong> → A<br>
<strong>Quality classifier</strong> → B<br>
<strong>MinHash LSH</strong> → C<br>
<strong>N-gram contamination filter</strong> → D<br>
<strong>Domain mixing / DoReMi</strong> → E<br>
<strong>Streaming MDS shards</strong> → F<br>
<strong>Synthetic textbook data</strong> → G
</p>
<p>The mental shortcut: <em>Common Crawl ingests, quality classifier filters, MinHash dedupes, n-gram filter catches contamination, DoReMi mixes domains, MDS streams, synthetic supplements</em>.</p>
</details>
</div>

<h2>Exercises</h2>

<div class="exercise">
<p><strong>1.</strong> A team trains a model on a 5T-token corpus. Eval scores look great. Months later, they discover that 8% of MMLU questions appeared verbatim in the training data. They ran an n-gram contamination filter; it didn't catch them. What likely happened?</p>
<details class="answer"><summary>show answer</summary>
<p>Several possibilities, in order of likelihood:</p>
<p>(1) <strong>Filter ran on the wrong data</strong>: maybe the n-gram filter was applied before some final data-mixing or augmentation step that re-introduced eval content. Order of operations matters; the contamination filter must be the last step before tokenization.</p>
<p>(2) <strong>N-gram size too large</strong>: with 13-grams, very short eval questions (under 13 tokens) can't match. MMLU questions average ~30 tokens but some are short. Lowering n to 8 catches more but also produces more false positives. There's a tradeoff; production filters often run multiple n sizes and union the matches.</p>
<p>(3) <strong>Bloom filter false-negatives</strong>: at petabyte scale, even tuned Bloom filters can have false negatives. Verify the false-negative rate is acceptable for your eval.</p>
<p>(4) <strong>Light paraphrasing</strong>: even small modifications ("What is the capital city of France?" vs "What is the capital of France?") break n-gram matching but the model still sees essentially the same content. This is the hardest contamination to catch; semantic detection (sentence embeddings + cosine similarity) is the next-level fix.</p>
<p>The takeaway: <strong>n-gram contamination detection is a floor, not a ceiling</strong>. Production teams complement it with memorization probes (prompt model with question; check if it completes the answer verbatim) and held-out variants of evals (GSM8K-Platinum, etc.). For high-stakes claims, multiple detection methods should agree.</p>
</details>
</div>

<div class="exercise">
<p><strong>2.</strong> Walk through why aggressive deduplication typically improves perplexity by 5-15%, given that you're throwing away training data.</p>
<details class="answer"><summary>show answer</summary>
<p>Two effects compound, both in your favor.</p>
<p>(1) <strong>Effective dataset size grows</strong>: with duplicates, the model spends some of its compute relearning the same content. After dedup, that compute reaches new content. Even though raw token count drops, <em>diverse</em> token count rises (or is preserved). The model sees more distinct material per training step.</p>
<p>(2) <strong>Distributional balance improves</strong>: duplicated content shifts the corpus distribution toward whatever's repeated. If "subscribe to our newsletter" appears in 1% of documents and 50% of pages on each, those tokens dominate. Dedup restores a more natural distribution where each document contributes once. The model learns the actual content distribution rather than the duplicate-weighted version.</p>
<p>The Lee et al. and Llama-3 papers report this empirically: training on dedup'd data of size N produces lower perplexity than training on raw data of size 1.5N. Per-token value is much higher post-dedup.</p>
<p>The exception: if your duplicates correspond to actually-important repetition (e.g., common code idioms, frequent grammatical patterns), aggressive dedup can hurt. Modern recipes use moderate dedup thresholds (Jaccard ≥ 0.8) that catch obvious duplicates without over-dedup'ing legitimate frequent content.</p>
</details>
</div>

<div class="exercise">
<p><strong>3.</strong> A team is training a 7B model on 2T tokens. They have a 100-million-document filtered corpus and want to add 20% code. They scrape ~30M code documents from GitHub, dedup, and add them to the pipeline. The model trains successfully but underperforms on code benchmarks vs published 7B models. What might they have missed?</p>
<details class="answer"><summary>show answer</summary>
<p>Several common pitfalls in code-data engineering:</p>
<p>(1) <strong>License filtering</strong>: many GitHub repos have restrictive licenses (GPL, no-redistribution). Mainstream code corpora (The Stack v2) filter to permissive licenses (MIT, Apache, BSD). If they grabbed everything, they may face later legal issues, but more relevantly: license-permissive filtering is correlated with quality (well-maintained projects tend to choose permissive licenses).</p>
<p>(2) <strong>Code-specific dedup is harder</strong>: code has many near-duplicates (forks, copies, same function across projects). Jaccard-based MinHash on character n-grams under-catches code duplicates because variable naming differences break shingle matching. The Stack uses tokenized code dedup (lex first, then dedup) plus exact substring removal. Without this, the model sees common functions hundreds of times, biasing it.</p>
<p>(3) <strong>Language balance</strong>: GitHub is dominated by JavaScript and Python by file count, but training across many languages helps multi-language code performance. Hand-balancing per-language token counts (e.g., upweight Rust, Haskell, Lean to match Python) improves cross-language transfer.</p>
<p>(4) <strong>Quality filtering for code</strong>: heuristics that work for prose (length, stopword frequency) don't apply. Code-specific heuristics: line length, identifier-quality scoring, presence of tests, presence of documentation. The fastText quality classifier needs code-specific training data, not the same model used for prose.</p>
<p>(5) <strong>Tokenizer mismatch</strong>: if the tokenizer was trained on prose-heavy data, code is inefficient (long sequences). Retraining the tokenizer on the new prose+code mix can recover 10-20% of effective context length on code.</p>
<p>Most likely the team hit (2) and (4). Code data engineering has its own subdiscipline; assuming web-data techniques transfer is a common mistake.</p>
</details>
</div>

<div class="exercise">
<p><strong>4.</strong> Synthetic data is 5-15% of modern pretraining corpora. What's the upper limit on this fraction, and why?</p>
<details class="answer"><summary>show answer</summary>
<p>The empirical answer is roughly 25-30% before quality starts degrading on diverse benchmarks; some published recipes go higher with careful curation. The reasons compound:</p>
<p>(1) <strong>Distributional narrowness</strong>: synthetic data tends to be stylistically uniform — the teacher model has its own writing style, even with diverse prompts. At small fractions, this is fine; at large fractions, the student inherits the teacher's stylistic limitations. Real web data, despite being noisier, has stylistic breadth that's hard to synthesize.</p>
<p>(2) <strong>Hallucination propagation</strong>: synthetic content reflects the teacher's mistakes. A frontier teacher hallucinates rarely, but at hundreds of billions of synthetic tokens, even rare hallucinations accumulate into measurable error rates in the student. Real data has its own errors but they're less correlated.</p>
<p>(3) <strong>Recursive collapse</strong>: training mostly on outputs of models trained on outputs leads to distribution narrowing across generations — the "model collapse" effect studied in recent papers. Even with one teacher generation, this is a concern at high synthetic fractions.</p>
<p>(4) <strong>Diversity constraints on prompting</strong>: getting truly diverse synthetic data requires diverse prompts, which require careful curation that's hard to scale. Easier to generate 100B tokens of "physics textbook chapters" than 100B tokens covering the breadth of human writing.</p>
<p>The current consensus: synthetic data is a useful supplement (5-25%) for specific quality boosts (math reasoning, structured explanations, instruction-style data). It's not a replacement for real-data breadth. <em>Future progress may shift this</em> — better-controlled synthesis, multi-teacher diversity, evolutionary methods — but as of 2026, the ceiling is still in the 25-30% range for most teams. Phi-style recipes that go higher exist but are highly specialized.</p>
</details>
</div>

<div class="bullet-points">
<h3>What just happened?</h3>
<ul>
  <li>Pretraining data engineering quietly does most of the work behind modern model quality. The architecture and optimizers are largely settled; the data is where teams differentiate.</li>
  <li>The pipeline is a <strong>funnel</strong>: hundreds of petabytes of raw Common Crawl → tens of TB of training tokens. Order-of-magnitude rejections at each stage.</li>
  <li><strong>Stage 1 — Ingestion + Language ID</strong>: trafilatura/jusText for HTML extraction; fastText lid.176 for language identification. Keep ~70% (English + major languages).</li>
  <li><strong>Stage 2 — Heuristic filters</strong>: Gopher-style rules on length, repetition, stopwords, special characters. Reject 50%+ of documents quickly.</li>
  <li><strong>Stage 3 — Quality classifier</strong>: fastText model trained on LLM-judged labels (the FineWeb-Edu approach). Score the full corpus at scale; threshold ~3.5 of 5.</li>
  <li><strong>Stage 4 — Deduplication</strong>: MinHash LSH for near-duplicates (Jaccard ≥ 0.8); exact substring dedup for boilerplate; semantic dedup as a frontier extension.</li>
  <li><strong>Stage 5 — Contamination filtering</strong>: n-gram overlap detection against eval sets. Catches verbatim contamination; misses paraphrases. Memorization probes complement it.</li>
  <li><strong>Domain mixing</strong>: web (~50-60%), books/papers (~10-15%), code (~10-20%), Wikipedia (~3-5%), math, synthetic. Sweet spot for code is ~15%; depends on eval distribution.</li>
  <li><strong>DoReMi</strong>: principled per-domain weight optimization via group DRO during a small-model training run.</li>
  <li><strong>Tokenizer training</strong>: BPE/Unigram on a representative sample. 128K vocab for English, 256K+ for multilingual.</li>
  <li><strong>Streaming infrastructure</strong>: WebDataset / Mosaic Streaming / custom binary; deterministic shuffle, sequence packing, object-storage streaming.</li>
  <li><strong>Synthetic data</strong>: LLM-generated content (textbooks, instruction data); 5-25% of corpus typically; risks of mode collapse and hallucination propagation.</li>
  <li>The end-to-end engineering effort is substantial: 10-30% of training compute is spent on data, not the model itself.</li>
  <li>The reflex: when a model underperforms despite good architecture and tuning, the data is usually the suspect. Audit the pipeline; check for under-dedup, under-filtered junk, undercaught contamination, or domain imbalance.</li>
</ul>
</div>

<p>Module 36 (if Part X continues) would tackle <strong>mechanistic interpretability</strong> — sparse autoencoders, feature circuits, and the engineering of interpretability at production scale. Fast becoming a required skill with no good engineering-level treatment elsewhere.</p>

<div class="module-footer">
  <span>PyTorch · From Tensor to Kernel</span>
  <span class="num">35</span>
  <span>Pretraining data engineering at scale</span>
</div>
"""

emit("35_data_engineering", "Module 35 — Pretraining data engineering at scale", BODY)
