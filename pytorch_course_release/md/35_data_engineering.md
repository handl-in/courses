# Module 35 — Pretraining data engineering at scale

# _Pretraining data engineering_ at scale

_Part X · Module 35 · the data layer_

— from raw Common Crawl to a 15T-token training set: the pipeline, the deduplication, the quality classifiers, the domain mixing, and the streaming infrastructure that quietly does most of the work behind modern model quality

\--- 

Architecture is largely settled. Optimization is largely settled. Post-training is converging fast. What still varies enormously between models — and what most teams underinvest in — is the **data**. Two models with the same architecture, same compute, same hyperparameters can differ by 10+ percentage points on benchmarks because one was trained on better data. Understanding why, and how, is the focus of this module.

Pretraining data engineering is also the most under-documented topic in the field. Architectures are described in papers; data pipelines are described in vague footnotes ("we filtered for quality and deduplicated"). What "filtered for quality" actually means in practice is folklore distributed across labs. This module collects the public knowledge — what FineWeb, Dolma, RedPajama, and similar open efforts have published — into an engineering playbook. _The engineering is unglamorous and crucial_ ; readers who internalize this module will have an edge most ML engineers don't.

> **★ KEY IDEA**  
>  A modern pretraining run consumes on the order of 10-15 trillion high-quality tokens. The journey from Common Crawl (hundreds of petabytes of raw web data) to those final tokens involves **order-of-magnitude rejections at each pipeline stage** : language identification (~70% kept), quality filtering (~20-30% of the remaining), deduplication (~30-50% of what's left), contamination removal (~5-15% additional). The pipeline ends with maybe 1-3% of the input bytes, and that 1-3% is what the model actually trains on. **Quality classifiers** (small fastText models trained to predict "is this document high-quality educational content?") and **MinHash-based near-duplicate detection** are the engineering centerpieces. **Domain mixing** (what fraction of web vs code vs books vs synthetic) is then a separate optimization problem, often solved with DoReMi or empirical sweeps. **Contamination** — eval data leaking into training — is the silent quality killer; n-gram-based detection catches most but not all. Tokenizer training, streaming infrastructure, and synthetic data generation round out the system. 

## Two new faces — the data pipeline pair

C

Curator

"I throw away 99% of what comes in. What remains is what matters."

My job is to look at every document and decide: is this worth training on? Most of Common Crawl is boilerplate, SEO spam, machine-generated junk, and broken HTML. _The internet at scale is mostly noise_. I run language ID, then heuristic filters (length, perplexity ratios, special-character density), then a learned quality classifier — usually a tiny fastText model trained to recognize "educational content." Each stage is cheap; together they reduce raw bytes by 50-100×. Quality matters far more than quantity past a threshold: **a billion well-curated tokens beats ten billion noisy ones**. The frontier labs spend more compute on me than on training the model in some cases.

D

Deduper

"I find near-duplicates at scale. Two documents that share 80% of their 5-grams are the same document."

The internet has a lot of repetition: same article republished on different sites, same Wikipedia stub mirrored a hundred times, near-paraphrases. If I don't catch these, the model overweights what's repeated — and overweighting one source is a strange way to train. I use **MinHash with LSH** : hash each document's 5-grams, signature it, bucket by signature, find collisions. _Linear in document count, finds near-duplicates that differ by edits, runs on petabytes_. I also do exact substring dedup with suffix arrays for the long-string case. Removing duplicates typically improves perplexity by 5-15% at fixed compute — bigger than most architectural choices. **Most teams underestimate me.**

## The shape of the problem

Pretraining data lives at a scale that breaks the assumptions you'd bring from "normal" data engineering. A few reference numbers as of 2026:

  * **Common Crawl** : the canonical raw web corpus. Each monthly snapshot is ~250 TB of compressed WARC files; cumulative archives are 10+ PB.
  * **Realistic training set sizes** : GPT-3 (300B tokens), Llama-2 (2T tokens), Llama-3 (15T tokens), Llama-4 / GPT-4-class (rumored 20-30T+ tokens). Each token is roughly 4 characters of text — so 15T tokens is ~60 TB of UTF-8 text.
  * **The funnel ratio** : from raw Common Crawl bytes to training tokens, you're keeping somewhere in the range of 1-3% by volume. The other 97-99% is filtered out.

This shape — petabytes of input, terabytes of output, multi-stage filtering — drives the engineering. You can't load it all into memory. You can't even loop over it on one machine in reasonable time. Every stage of the pipeline has to be parallel, fault-tolerant, and resumable. The pipeline itself is closer to a Spark / dataflow job than to a training script.

The pretraining data funnel: 100s of PB → 10s of TB ① Common Crawl raw — ~250 TB / month, 10+ PB cumulative WARC files, raw HTML, hundreds of billions of pages ↓ keep ~70% (English+major languages) ② After language ID + text extraction — ~70 PB trafilatura/jusText extraction, fastText language ID, drop garbage HTML ↓ keep ~25-30% (heuristics + quality classifier) ③ After quality filtering — ~20 PB heuristics (length, repetition, special chars) + fastText quality classifier ↓ keep ~50-70% (after dedup) ④ After deduplication — ~12 PB MinHash LSH near-dup; exact substring dedup ↓ keep ~85-95% (contamination filter) ⑤ Final training tokens — ~10-15 TB

Read top-to-bottom: each stage cuts the data by some fraction, and the cuts compound. Final outputs are 1-3% of inputs — the rest is filtered out for being non-English, low-quality, duplicated, or contaminated with eval sets. _If you've never run this pipeline before, the loss rate is shocking on the first run._ The shock is the right reaction; the noise floor of the internet is genuinely that high.

## Stage 1: Ingestion and language identification

The starting point is usually Common Crawl. Each WARC file is a sequence of HTTP responses with metadata. The first job is text extraction: pulling readable content out of HTML, dropping navigation/sidebars/footers/scripts. Standard tools:

  * **trafilatura** : best-in-class HTML-to-text extraction; handles boilerplate removal well.
  * **jusText** : older but still common; faster but slightly worse extraction quality.
  * **resiliparse** : fast HTML parser, often used as a preprocessing layer before trafilatura.

After extraction, language identification with **fastText's lid.176 model** (or CLD3): given a document's text, predict which of 176 languages it's in, with confidence. Documents below a confidence threshold (typically 0.65) are dropped — they're often very short, mixed-language, or garbage that the language ID can't classify. English-only training keeps documents classified as English with high confidence; multilingual training keeps documents in a configured set of languages with thresholds tuned per-language.

Engineering at this stage: **this is where you write to a parallel file system**. Ingestion typically writes to JSONL (one document per line) sharded across thousands of files, or to a columnar format like Parquet for analytics-friendly downstream stages. Each file is a few GB — large enough that random access is amortized, small enough that one worker can handle one file. _Choose the shard size to match your downstream parallelism budget._

## Stage 2: Heuristic quality filters

Before any learned classifier, simple heuristics catch the most obvious junk. The Gopher paper's filters are still the standard reference:

Gopher-style quality heuristics — each filter typically rejects 1-10% of documents Filter| Rejects| Why  
---|---|---  
Document < 50 words| Stub pages, error messages| Too short to learn structure from  
Document > 100K words| Auto-generated logs, large dumps| Often machine output or scraped junk  
Mean word length < 3 or > 10| Token soup, code in disguise as text| Real text has ~5 char/word  
> 10% words contain digits| Phone books, log files, data dumps| Not natural language  
> 30% lines end with ellipsis| Truncated previews, paywalled content| Reading those teaches the model to stop mid-sentence  
< 50% lines are sentences (no punct)| Lists, menus, broken extraction| Word salad, no syntactic structure  
Repetition: top 2-gram > 20% of text| Boilerplate ("Click here ... Click here ...")| Highly repetitive content over-trains specific sequences  
Top stopword frequency < 2%| Non-English in disguise, code, garbage| Real English has 7-10% stopwords  
  
Each filter is cheap (no model inference, just regex / counting). Implemented as a pipeline of predicates, applied document-by-document. Reject rates compound — you typically lose 30-50% of post-language-ID documents to heuristic filters alone, and that's before the quality classifier even runs.

## Stage 3: The learned quality classifier

Heuristics catch obvious junk; a learned classifier catches subtler quality signals. The mainstream approach (FineWeb-Edu, DCLM, Llama-3): train a tiny classifier — a fastText model or a distilled BERT-base — to predict "is this document high-quality educational content?"

The bootstrap problem: how do you get labels in the first place? Two approaches in production use:

  1. **Heuristic-bootstrapped** : start with manually curated high-quality corpora (Wikipedia, ArXiv, books) as positive examples; sample from raw web data as negatives. Train a binary classifier. Apply to web data to score each document.
  2. **LLM-judged** (FineWeb-Edu approach): use a strong LLM (Llama-3-70B-Instruct, GPT-4) to rate ~500K documents on educational quality (0-5 score). Use these ratings as training data for the small classifier. The small classifier then scores the full corpus at scale — it's millions of times cheaper than running the LLM on every document.

Method 2 dominates in 2024-2026 because LLM-as-judge is now reliable enough for this use case, and the small classifier inherits the LLM's quality judgment at fastText speed. The training procedure is roughly:
    
    
    import fasttext
    
    # Step 1: Get LLM ratings (done once, expensively)
    # For ~500K-1M documents, use GPT-4 / Llama-3-70B to score 0-5
    # Save as: \t per line
    
    # Step 2: Train fastText classifier on these labels
    model = fasttext.train_supervised(
        input="quality_labels.txt",
        epoch=5,
        lr=0.5,
        wordNgrams=2,
        minCount=3,
        dim=100,
        loss="hs",           # hierarchical softmax — fast for many classes
    )
    # Trained model is ~few hundred MB, scores documents at 1-10ms each
    model.save_model("quality_classifier.bin")
    
    # Step 3: Score the full corpus and threshold
    def score_document(text, model, threshold=3.5):
        label, prob = model.predict(text.replace("\n", " "))
        score = int(label[0].replace("__label__", ""))
        return score >= threshold

The threshold (typically 3 or 3.5 out of 5) is a hyperparameter tuned by ablating training runs at small scale. Higher threshold = less data, higher quality per document; lower threshold = more data, more noise. **FineWeb-Edu's published results showed that threshold-3 classifier-filtered data outperformed unfiltered FineWeb at the same compute budget** — quality won over quantity at the margin.

The 1-10ms per document inference cost matters when you're scoring 100B documents. At 5ms/doc with 1000 worker cores, scoring takes ~5 days; with 10K cores, ~12 hours. _Make the classifier as small as you can while preserving quality_ — fastText's hash-based features and hierarchical softmax exist for this reason.

## Stage 4: Deduplication

The web has many copies of many documents. Wikipedia is mirrored on hundreds of sites. News articles are republished, paraphrased, summarized. Boilerplate ("subscribe to our newsletter") appears identically across millions of pages. **Without deduplication, the model spends a disproportionate share of its compute relearning the same content** — and its outputs reflect that overweighting.

The published evidence is clear: aggressive dedup typically improves perplexity by 5-15% at fixed FLOPs. The Lee et al. paper, the SemDedup paper, and Llama-3's recipe all converge on this conclusion. _Dedup is one of the highest-leverage data interventions you can run._

Three flavors of dedup, each catching different patterns:

  * **Exact dedup** : hash each document; keep one copy per hash. Catches verbatim repeats. Cheap (one hash per doc) but misses anything with edits.
  * **Near-duplicate detection (MinHash LSH)** : documents that share most of their 5-grams or shingles are flagged as duplicates. Catches republished articles with minor reformatting. Typical recall: documents with Jaccard similarity ≥ 0.8 are reliably caught.
  * **Substring dedup** : find long substrings (e.g., 50+ tokens) that appear multiple times across the corpus; remove all but one occurrence. Catches boilerplate and shared paragraphs. Done with suffix array on tokenized corpus.

### MinHash LSH in real terms

The MinHash LSH approach is the workhorse. The intuition: instead of comparing every document to every other (O(N²) — impossible at scale), build a signature for each document such that _similar documents have similar signatures_. Group documents by signature; near-duplicates cluster together.
    
    
    def minhash_signature(text, num_hashes=128, ngram_size=5):
        # Step 1: extract n-grams (shingles).
        tokens = text.split()
        ngrams = set(" ".join(tokens[i:i+ngram_size])
                     for i in range(len(tokens) - ngram_size + 1))
    
        # Step 2: compute MinHash signature.
        # For each of num_hashes hash functions, find min hash value across all n-grams.
        signature = []
        for i in range(num_hashes):
            seed = 12345 + i
            min_val = min(murmurhash3(ngram, seed=seed) for ngram in ngrams)
            signature.append(min_val)
        return signature
    
    def lsh_buckets(signature, bands=16, rows_per_band=8):
        # Split signature into bands; documents that match in any band → candidates.
        assert bands * rows_per_band == len(signature)  # 16 * 8 = 128
        buckets = []
        for b in range(bands):
            band_sig = tuple(signature[b*rows_per_band : (b+1)*rows_per_band])
            bucket_id = (b, hash(band_sig))
            buckets.append(bucket_id)
        return buckets
    
    # Pipeline: for each doc, compute signature + bucket IDs.
    # Group docs by bucket ID. Within each bucket, mark all but one as duplicates.

The math: with bands=16 and rows_per_band=8, the probability that two documents with Jaccard similarity `s` match in at least one band is `1 - (1 - s^8)^16`. This S-curve has a sharp threshold around `s ≈ 0.8`: documents above 80% similarity almost always match; documents below 70% almost never do. _Tune bands and rows_per_band to set your similarity threshold_.

Engineering at scale: signature computation is embarrassingly parallel (per-document). Bucket grouping is a distributed shuffle. Within each bucket, deduplication picks one canonical document (typically the longest, or the highest-quality-classifier-scored). Production implementations: HuggingFace's `datatrove`, the `text-dedup` package, custom Spark/Ray jobs at frontier labs.

### What dedup misses

MinHash LSH catches near-duplicates with high textual overlap. It misses:

  * **Heavy paraphrases** : same content, totally different wording. Two articles describing the same news event in different journalistic styles. SemDedup uses embedding similarity to catch these.
  * **Translations** : same content in different languages.
  * **Summaries vs full text** : one document is a summary of another. Different lengths break shingle overlap.

For these, semantic deduplication (compute sentence embeddings, cluster by cosine similarity) is the next-level technique. More expensive: requires an embedding model and an ANN index over billions of embeddings. Used by some frontier labs; not yet common in open recipes.

## Stage 5: Contamination filtering

Eval sets are public. The internet has copies of GSM8K, MATH, MMLU, HumanEval. If those copies end up in your training data, your eval scores are inflated — the model is memorizing, not generalizing. **Contamination is the silent quality killer** : training loss looks normal, eval looks great, but the eval is meaningless because the model has seen the answers.

The published numbers are sobering. Multiple analyses have found that 5-15% of canonical eval sets are present in raw web crawls. Without explicit filtering, a non-trivial fraction of your reported eval performance is contamination. The Llama-3 paper, GPT-4 technical report, and similar all describe contamination filtering steps and report contamination percentages.

The standard approach: **n-gram overlap detection**. For each eval example, extract its n-grams (typically 13-grams or 50-token spans). Hash them. For each training document, check if it contains any eval n-gram. If the overlap exceeds a threshold (e.g., 1 matching 50-gram = 0.1% overlap), reject the document.
    
    
    def build_eval_ngram_set(eval_examples, n=13):
        # Build a set of all n-grams from all eval examples (use Bloom filter for memory)
        all_ngrams = set()
        for ex in eval_examples:
            text = ex["question"] + " " + ex["answer"]
            tokens = text.split()
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i+n])
                all_ngrams.add(hash(ngram))
        return all_ngrams
    
    def contamination_score(doc, eval_ngrams, n=13):
        tokens = doc.split()
        if len(tokens) < n:
            return 0.0
        matches = sum(
            1 for i in range(len(tokens) - n + 1)
            if hash(" ".join(tokens[i:i+n])) in eval_ngrams
        )
        return matches / max(len(tokens) - n + 1, 1)

For very large training sets, even building the full n-gram set in memory is impractical. Bloom filters (with controlled false-positive rate) and bit-set hashing replace the Python set. The pipeline pattern: build the eval n-gram Bloom filter once (a few GB), then check each training document against it as a streaming filter.

What n-gram detection misses: **paraphrased contamination**. If MMLU's question is "What is the capital of France?" and the training data has "Of all the cities in France, the capital is Paris," n-gram overlap is low but the model still learns the answer. Catching paraphrased contamination requires semantic similarity, which is more expensive and less standard. _This is an open problem_ ; published frontier-model contamination reports almost certainly underreport because they only measure n-gram contamination.

## Domain mixing

You now have a clean, deduplicated, uncontaminated corpus. But it's a _web-heavy_ corpus — Common Crawl is mostly forums, blogs, news, and SEO content. To train a high-quality model, you want to mix in other domains: code (GitHub), books (project Gutenberg / commercial sources), academic papers (ArXiv), reference material (Wikipedia, StackExchange), perhaps math-specific data, perhaps multilingual data.

The question becomes: _what fractions of each?_ Too much code hurts general performance; too little code hurts code-specific tasks. Too much Wikipedia overweights factual recall but underweights conversational fluency. The optimization landscape is real and matters.

Domain mixing tradeoff: quality on different categories vs code-fraction code fraction in training mix (%) eval performance 0% 10% 25% 50% 100% general/chat code math composite ★ ~15% sweet spot code teaches general reasoning too — but past a point, you lose conversational fluency

Three patterns to internalize from the chart:

  1. **Code helps general performance** at low fractions (0% → 15% improves general/chat eval somewhat). The hypothesis: code teaches structured reasoning that transfers to non-code tasks.
  2. **Code hurts general past a point** : at 50%+ code, general performance degrades significantly. The model becomes a programmer who can barely chat.
  3. **Optimal mixing depends on the eval distribution** : if you weight code-heavy benchmarks, more code wins. If you weight chat-heavy benchmarks, less code wins. The composite weighted average peaks somewhere in the 10-25% code range for typical product evals.

### DoReMi: principled domain mixing

Sweep-based domain optimization (try N ratios, pick the best) is expensive: each ratio requires a full small-scale training run for ablation. The DoReMi paper proposes a more efficient method: **train a small model with multiple-domain weights as a hyperparameter, then use the model's loss on each domain to update the weights via group DRO (distributionally robust optimization)**.

The mechanics: maintain weights `w_d` for each domain `d`. After each training step, compute the loss on each domain's held-out validation set; the domain with the highest loss-to-reference-loss ratio gets its weight increased. Over training, the weights converge to a mixture that's "hard for the model to learn" — usually corresponding to underrepresented or harder domains.

DoReMi is one published recipe; in practice, frontier labs use combinations of: small-scale sweep ablations, DoReMi-style algorithmic optimization, and human judgment about what categories to prioritize. _The choice of mixing ratios is more art than science_ ; there's no universal answer because different products have different eval distributions.

### Other knobs in the mix

  * **Curriculum scheduling** : change the mix during training. Some labs put more web data early (general knowledge), more code/math/papers late (sharper skills). Not universally adopted.
  * **Quality up-weighting** : oversample high-quality-classifier-scored data within each domain. Higher-scored docs appear in 2-3 epochs; lower-scored in 1.
  * **Repetition** : at 15T tokens you can't easily get more unique high-quality data, so high-quality subsets get repeated 2-4 epochs. Empirically this works up to a point; beyond ~5 epochs, returns diminish.

## Tokenizer training

The tokenizer determines how text becomes integer sequences. It's trained _once_ on a representative sample of the final corpus, then frozen for the entire pretraining run. Bad tokenizer choices can permanently hurt model quality in subtle ways.

The standard algorithms:

  * **BPE (Byte-Pair Encoding)** : greedily merge most-frequent token pairs. Used by GPT, Llama, Mistral. Most common.
  * **Unigram** : probabilistic model where the tokenizer is trained to maximize likelihood of the corpus given a token vocabulary. Used by SentencePiece's default. Slightly different segmentation behavior.
  * **WordPiece** : similar to BPE but uses likelihood ratios to select merges. Used by BERT.

Vocab size tradeoffs:

  * **Small vocab (~32K)** : each token represents less; sequences are longer; models need more context per text amount. Used by older models (Llama-1, Llama-2 at 32K).
  * **Medium vocab (~128K)** : balanced. Used by Llama-3 (128K), Mistral.
  * **Large vocab (~256K+)** : each token covers more text; sequences are shorter for a given text length; better for long-context efficiency. Used by Gemma-2 (256K), some multilingual models.

Larger vocab pushes the cost from sequence length to embedding/lm_head matrix size. With tied embeddings (M29), a vocab of 256K with d=4096 adds ~1B parameters of vocab embedding — substantial. _For multilingual models or long-context-heavy applications, large vocab pays off; for English-only chat, 128K is the sweet spot._

Tokenizer training itself is fast — a few hours on a single machine for a few-GB sample of the corpus. The sample matters: train the tokenizer on a sample that reflects your final mix (e.g., the 15% code, 5% math, 80% web). If the tokenizer was trained on web-only and you train on 30% code, code tokens are inefficient (long sequences for short code).

## Streaming infrastructure

You can't fit a 15TB tokenized corpus on a single machine. You can't fit it on the local disk of every node in a 1000-node cluster either. The standard pattern: **store the corpus on object storage (S3, GCS), stream it during training**.

The format and library landscape:

  * **WebDataset** : tar files of (key, value) pairs. Standard for vision-text mixed corpora; works for text-only.
  * **Mosaic Streaming (MDS)** : optimized for LLM pretraining. Sharded binary format with deterministic shuffle ordering.
  * **HuggingFace datasets streaming mode** : streams from Parquet on object storage; convenient but slower than MDS.
  * **Custom binary formats** : most frontier labs roll their own — typically .bin files of pre-tokenized uint32 sequences with index files for random access.

The key engineering concern is **determinism under shuffling**. With many ranks, many epochs, and async I/O, you need each rank to see a different, non-overlapping shard at each step, AND the assignment must be reproducible across resumptions and across re-runs. MDS handles this via a deterministic shuffle algorithm seeded on (epoch, global_step). Rolling your own requires care — one of the easiest ways to silently corrupt training is to have ranks see overlapping data.

Another concern: **throughput-balanced reads**. Some shards have shorter sequences (faster to consume), some longer. If shards aren't balanced, fast workers idle waiting for slow ones. Sequence-packing (binning sequences to fill fixed-length contexts) addresses this and is now standard in production training.

## Synthetic data

The newest frontier in pretraining data engineering: **generate training data with another LLM**. The Cosmopedia paper (Hugging Face) and Phi family (Microsoft) demonstrated that synthetic textbooks, instruction data, and explanations can substantially boost model quality, especially at smaller scales.

The recipe typically:

  1. Take a strong base model (frontier-quality) as the teacher.
  2. Use it to generate text in specific styles: textbook explanations, code with comments, step-by-step problem solutions, conversational data.
  3. Apply quality filters (the same ones used on real data).
  4. Mix the synthetic data into the pretraining corpus, typically at 5-25% by token count.

Why does this work? Synthetic data can be more controlled than web data: every "textbook" can be accurate, well-structured, and pedagogically clear. The teacher model's knowledge is distilled into the student model via the synthetic data, in a more efficient form than raw web crawls would provide.

The risks:

  * **Mode collapse** : if the synthetic data is too narrow stylistically, the student inherits that narrowness. Diverse generation prompts and topic coverage matter.
  * **Hallucinations propagating** : if the teacher hallucinates, the student learns hallucinated facts as if they were real.
  * **Recursive collapse** : training on outputs of models trained on outputs leads to quality degradation. Most published synthetic-data recipes are careful to use real-data anchors and not chain synthetic generations.

Synthetic data is one of the few areas where _more compute spent on data generation_ can substitute for _more compute spent on training_. For smaller models (1B-10B), well-designed synthetic data can give multi-percentage-point benchmark gains at fixed training compute.

## Putting it all together: a 15T-token training set

The numbers behind a real frontier-quality 15T-token pretraining corpus, very approximately:

Approximate composition of a 15T-token corpus (illustrative; specific labs vary) Source| Token fraction| Engineering effort  
---|---|---  
Filtered Common Crawl (web)| ~50-60%| The bulk of the pipeline; quality classifier + dedup dominate  
Books and academic papers| ~10-15%| Licensing, OCR for scanned books, format normalization  
Code (GitHub, language-specific)| ~10-20%| License filtering, deduplication (code is highly duplicated), language balance  
Wikipedia and reference| ~3-5%| High-quality, repeated 2-3 epochs typically  
Math-specific data| ~2-5%| Curated math forums, ArXiv math, generated problem sets  
Synthetic data| ~5-15%| Generation pipeline + quality filters  
Multilingual non-English| variable| Per-language pipeline replication  
  
The total engineering effort is significant. Public data efforts (Dolma, FineWeb, DCLM, RedPajama-2) have published infrastructure that handles parts of this; private labs have substantially more elaborate pipelines. The compute cost of running the data pipeline can be 10-30% of the training compute itself — and that's just to prepare the data, not train the model.

**The investment pays off** : small differences in data curation produce large differences in final model quality. Reproducing a competitive frontier model with public weights is now feasible if you have the compute; reproducing one with high quality without the data infrastructure to match is not.

#### Q&A; — About data engineering **Q:** How important is the order of pipeline stages? Could I dedupe before quality-filter? **A:** Order matters in two ways. First, **compute cost** : the early stages (language ID, heuristics) are cheapest per document; they should run first to reduce the document count for expensive stages (classifier, dedup). Running classifier on 100 PB and then dedup-ing 25 PB of survivors is much cheaper than dedup-ing 100 PB and then classifying 70 PB. Second, **signal quality** : the quality classifier was likely trained on already-cleaned data. Running it on raw HTML-y output can produce confused scores. Standard order — language ID, heuristics, classifier, then dedup — minimizes both compute waste and signal corruption. Dedup after classifier means you keep the highest-classifier-scored doc when there are duplicates, which is what you want. **Q:** If 99% of Common Crawl is junk, why not just use Wikipedia + ArXiv + books? **A:** Two reasons. (1) **Volume** : even with all of Wikipedia + ArXiv + book corpora, you have maybe 100-300B tokens. A frontier model wants 10T+. The web is where the volume comes from. (2) **Distribution** : Wikipedia is encyclopedic, ArXiv is academic, books are narrative. None of them prepare a model for chat-style language, casual writing, or domain-specific jargon that appears in product use. Web data — even after aggressive filtering — provides the distributional breadth that makes a model feel "natural" across topics. The recipe is: high-quality curated sources for facts and structure; web-derived high-quality content for breadth and style. **Q:** How do I know if my deduplication is working? **A:** Three diagnostics. (1) **Reduction ratio** : typical post-dedup reduction is 30-50% of post-classifier data. If you're seeing 5-10%, your dedup is too lenient (raise similarity threshold by lowering bands, or use larger n-grams). If you're seeing 80%+, it's too aggressive (reverse). (2) **Manual sampling** : take 100 random surviving documents; check if any are obviously near-duplicates. If yes, lowering the threshold further. (3) **Train ablation** : train a small model on dedup'd vs not-dedup'd data of the same final size; compare perplexity on a held-out validation set. Dedup'd should win by 5-15%; if it doesn't, your dedup isn't catching the right duplicates. Most production teams do all three. **Q:** Can I just use a published corpus like FineWeb instead of building my own pipeline? **A:** Yes, often. FineWeb-Edu, DCLM-baseline, RedPajama-2, Dolma — these are published, free, high-quality, and well-deduplicated. For most teams, starting with one of these and adding domain-specific data (your own code corpus, your own books, etc.) is the right move. You only need a custom pipeline if (1) you need data sources the public efforts don't cover, (2) you're at frontier scale and the public data isn't enough, or (3) you have specific quality criteria that differ from the public efforts. _For research and most production work, the published corpora are an excellent starting point_ ; building from scratch is a substantial engineering investment that's hard to justify when high-quality public alternatives exist. **Q:** How do I detect if my eval is contaminated? **A:** Three approaches. (1) **N-gram overlap** : as covered, check if eval n-grams appear in training. Standard. (2) **Memorization probes** : prompt the model with the first half of an eval question; see if it produces the second half verbatim. If yes, contamination. (3) **Comparison to held-out variants** : take the same task with different phrasings or freshly-generated examples (e.g., GSM8K-Platinum is a contamination-resistant variant). Large performance gap between original and variant indicates contamination. _Any new eval should be checked for contamination as part of release_ ; old evals with high contamination should be deprecated. The community is moving toward "live" evals (continuously generated, never published) for this reason. **Q:** What's the right vocab size for my model? **A:** Three rules of thumb. (1) **For English-only chat** : 128K is the sweet spot. Llama-3, Mistral, most modern English models use this range. (2) **For multilingual** : 256K+ for good cross-lingual coverage. The vocab needs slots for non-English tokens to be efficient. (3) **For very small models** (<1B): smaller vocab (32K-64K) reduces the embedding parameter count, which is proportionally large at small model sizes. Beyond these heuristics, you can sweep at small scale — the cost of training a tokenizer is negligible compared to model training. _Don't over-optimize_ : vocab size matters but is rarely the bottleneck in real applications. 

## Code Magnets: implement MinHash LSH bucketing

You're computing LSH buckets from a MinHash signature for deduplication. Three magnets are wrong choices.

Arrange the magnets to compute LSH buckets from the signature.

def lsh_buckets(signature, bands=16, rows_per_band=8): assert bands * rows_per_band == len(signature) assert bands + rows_per_band == len(signature) buckets = [] for b in range(bands): band_sig = tuple(signature[b*rows_per_band : (b+1)*rows_per_band]) band_sig = tuple(signature[b : b+rows_per_band]) bucket_id = (b, hash(band_sig)) bucket_id = hash(band_sig) buckets.append(bucket_id) return buckets

show solution
    
    
    def lsh_buckets(signature, bands=16, rows_per_band=8):
        assert bands * rows_per_band == len(signature)
        buckets = []
        for b in range(bands):
            band_sig = tuple(signature[b*rows_per_band : (b+1)*rows_per_band])
            bucket_id = (b, hash(band_sig))
            buckets.append(bucket_id)
        return buckets

The traps:

  * `assert bands + rows_per_band == len(signature)`: addition instead of multiplication. The signature must split exactly into `bands` non-overlapping bands of `rows_per_band` each. With 128-element signature, bands=16, rows_per_band=8, you need 16 × 8 = 128. Adding gives 24, completely wrong, and the slicing logic would silently produce overlapping or partial bands.
  * `band_sig = tuple(signature[b : b+rows_per_band])`: wrong slicing. This produces overlapping windows starting at positions 0, 1, 2, ... — bands b=0 and b=1 share most of their values. The non-overlapping form is `signature[b*rows_per_band : (b+1)*rows_per_band]` — band 0 is positions 0-7, band 1 is 8-15, etc. With overlapping bands, similar documents collide far more often than they should, ruining the dedup precision.
  * `bucket_id = hash(band_sig)`: missing the band index. Two documents that match in band 5 should share a bucket, but two documents that happen to have the same hash in _different_ bands shouldn't be confused. The bucket ID is `(band_index, band_hash)` — the tuple ensures bands don't accidentally collide. Without it, you'd get false-positive duplicate matches between unrelated bands.

The pattern: **signature splits into non-overlapping bands → each band hashed → bucket ID is (band_index, band_hash) → documents in same bucket are candidates**. The math of LSH (the S-curve of similarity vs match probability) only works with this exact partitioning; deviations silently break the algorithm.

## Who does what?

Match each data engineering concept to its real role.

Concept

Real role

Common Crawl

A. Petabyte-scale raw web corpus; the starting point for pretraining data.

Quality classifier (fastText)

B. Tiny model trained to score documents on educational quality; runs on the full corpus cheaply.

MinHash LSH

C. Near-duplicate detection at scale: signature each document, bucket by signature, find collisions.

N-gram contamination filter

D. Detects eval set leakage by checking training docs for eval n-grams (typically 13-grams).

Domain mixing / DoReMi

E. Algorithmic optimization of the per-domain weights in the training mix.

Streaming MDS shards

F. Petabyte-scale corpus served from object storage with deterministic shuffle.

Synthetic textbook data

G. LLM-generated pretraining data (Cosmopedia, Phi); 5-25% of corpus typically.

show solution

**Common Crawl** → A  
**Quality classifier** → B  
**MinHash LSH** → C  
**N-gram contamination filter** → D  
**Domain mixing / DoReMi** → E  
**Streaming MDS shards** → F  
**Synthetic textbook data** → G 

The mental shortcut: _Common Crawl ingests, quality classifier filters, MinHash dedupes, n-gram filter catches contamination, DoReMi mixes domains, MDS streams, synthetic supplements_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team trains a model on a 5T-token corpus. Eval scores look great. Months later, they discover that 8% of MMLU questions appeared verbatim in the training data. They ran an n-gram contamination filter; it didn't catch them. What likely happened?

show answer

Several possibilities, in order of likelihood:

(1) **Filter ran on the wrong data** : maybe the n-gram filter was applied before some final data-mixing or augmentation step that re-introduced eval content. Order of operations matters; the contamination filter must be the last step before tokenization.

(2) **N-gram size too large** : with 13-grams, very short eval questions (under 13 tokens) can't match. MMLU questions average ~30 tokens but some are short. Lowering n to 8 catches more but also produces more false positives. There's a tradeoff; production filters often run multiple n sizes and union the matches.

(3) **Bloom filter false-negatives** : at petabyte scale, even tuned Bloom filters can have false negatives. Verify the false-negative rate is acceptable for your eval.

(4) **Light paraphrasing** : even small modifications ("What is the capital city of France?" vs "What is the capital of France?") break n-gram matching but the model still sees essentially the same content. This is the hardest contamination to catch; semantic detection (sentence embeddings + cosine similarity) is the next-level fix.

The takeaway: **n-gram contamination detection is a floor, not a ceiling**. Production teams complement it with memorization probes (prompt model with question; check if it completes the answer verbatim) and held-out variants of evals (GSM8K-Platinum, etc.). For high-stakes claims, multiple detection methods should agree.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** Walk through why aggressive deduplication typically improves perplexity by 5-15%, given that you're throwing away training data.

show answer

Two effects compound, both in your favor.

(1) **Effective dataset size grows** : with duplicates, the model spends some of its compute relearning the same content. After dedup, that compute reaches new content. Even though raw token count drops, _diverse_ token count rises (or is preserved). The model sees more distinct material per training step.

(2) **Distributional balance improves** : duplicated content shifts the corpus distribution toward whatever's repeated. If "subscribe to our newsletter" appears in 1% of documents and 50% of pages on each, those tokens dominate. Dedup restores a more natural distribution where each document contributes once. The model learns the actual content distribution rather than the duplicate-weighted version.

The Lee et al. and Llama-3 papers report this empirically: training on dedup'd data of size N produces lower perplexity than training on raw data of size 1.5N. Per-token value is much higher post-dedup.

The exception: if your duplicates correspond to actually-important repetition (e.g., common code idioms, frequent grammatical patterns), aggressive dedup can hurt. Modern recipes use moderate dedup thresholds (Jaccard ≥ 0.8) that catch obvious duplicates without over-dedup'ing legitimate frequent content.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** A team is training a 7B model on 2T tokens. They have a 100-million-document filtered corpus and want to add 20% code. They scrape ~30M code documents from GitHub, dedup, and add them to the pipeline. The model trains successfully but underperforms on code benchmarks vs published 7B models. What might they have missed?

show answer

Several common pitfalls in code-data engineering:

(1) **License filtering** : many GitHub repos have restrictive licenses (GPL, no-redistribution). Mainstream code corpora (The Stack v2) filter to permissive licenses (MIT, Apache, BSD). If they grabbed everything, they may face later legal issues, but more relevantly: license-permissive filtering is correlated with quality (well-maintained projects tend to choose permissive licenses).

(2) **Code-specific dedup is harder** : code has many near-duplicates (forks, copies, same function across projects). Jaccard-based MinHash on character n-grams under-catches code duplicates because variable naming differences break shingle matching. The Stack uses tokenized code dedup (lex first, then dedup) plus exact substring removal. Without this, the model sees common functions hundreds of times, biasing it.

(3) **Language balance** : GitHub is dominated by JavaScript and Python by file count, but training across many languages helps multi-language code performance. Hand-balancing per-language token counts (e.g., upweight Rust, Haskell, Lean to match Python) improves cross-language transfer.

(4) **Quality filtering for code** : heuristics that work for prose (length, stopword frequency) don't apply. Code-specific heuristics: line length, identifier-quality scoring, presence of tests, presence of documentation. The fastText quality classifier needs code-specific training data, not the same model used for prose.

(5) **Tokenizer mismatch** : if the tokenizer was trained on prose-heavy data, code is inefficient (long sequences). Retraining the tokenizer on the new prose+code mix can recover 10-20% of effective context length on code.

Most likely the team hit (2) and (4). Code data engineering has its own subdiscipline; assuming web-data techniques transfer is a common mistake.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** Synthetic data is 5-15% of modern pretraining corpora. What's the upper limit on this fraction, and why?

show answer

The empirical answer is roughly 25-30% before quality starts degrading on diverse benchmarks; some published recipes go higher with careful curation. The reasons compound:

(1) **Distributional narrowness** : synthetic data tends to be stylistically uniform — the teacher model has its own writing style, even with diverse prompts. At small fractions, this is fine; at large fractions, the student inherits the teacher's stylistic limitations. Real web data, despite being noisier, has stylistic breadth that's hard to synthesize.

(2) **Hallucination propagation** : synthetic content reflects the teacher's mistakes. A frontier teacher hallucinates rarely, but at hundreds of billions of synthetic tokens, even rare hallucinations accumulate into measurable error rates in the student. Real data has its own errors but they're less correlated.

(3) **Recursive collapse** : training mostly on outputs of models trained on outputs leads to distribution narrowing across generations — the "model collapse" effect studied in recent papers. Even with one teacher generation, this is a concern at high synthetic fractions.

(4) **Diversity constraints on prompting** : getting truly diverse synthetic data requires diverse prompts, which require careful curation that's hard to scale. Easier to generate 100B tokens of "physics textbook chapters" than 100B tokens covering the breadth of human writing.

The current consensus: synthetic data is a useful supplement (5-25%) for specific quality boosts (math reasoning, structured explanations, instruction-style data). It's not a replacement for real-data breadth. _Future progress may shift this_ — better-controlled synthesis, multi-teacher diversity, evolutionary methods — but as of 2026, the ceiling is still in the 25-30% range for most teams. Phi-style recipes that go higher exist but are highly specialized.

### What just happened?

  * Pretraining data engineering quietly does most of the work behind modern model quality. The architecture and optimizers are largely settled; the data is where teams differentiate.
  * The pipeline is a **funnel** : hundreds of petabytes of raw Common Crawl → tens of TB of training tokens. Order-of-magnitude rejections at each stage.
  * **Stage 1 — Ingestion + Language ID** : trafilatura/jusText for HTML extraction; fastText lid.176 for language identification. Keep ~70% (English + major languages).
  * **Stage 2 — Heuristic filters** : Gopher-style rules on length, repetition, stopwords, special characters. Reject 50%+ of documents quickly.
  * **Stage 3 — Quality classifier** : fastText model trained on LLM-judged labels (the FineWeb-Edu approach). Score the full corpus at scale; threshold ~3.5 of 5.
  * **Stage 4 — Deduplication** : MinHash LSH for near-duplicates (Jaccard ≥ 0.8); exact substring dedup for boilerplate; semantic dedup as a frontier extension.
  * **Stage 5 — Contamination filtering** : n-gram overlap detection against eval sets. Catches verbatim contamination; misses paraphrases. Memorization probes complement it.
  * **Domain mixing** : web (~50-60%), books/papers (~10-15%), code (~10-20%), Wikipedia (~3-5%), math, synthetic. Sweet spot for code is ~15%; depends on eval distribution.
  * **DoReMi** : principled per-domain weight optimization via group DRO during a small-model training run.
  * **Tokenizer training** : BPE/Unigram on a representative sample. 128K vocab for English, 256K+ for multilingual.
  * **Streaming infrastructure** : WebDataset / Mosaic Streaming / custom binary; deterministic shuffle, sequence packing, object-storage streaming.
  * **Synthetic data** : LLM-generated content (textbooks, instruction data); 5-25% of corpus typically; risks of mode collapse and hallucination propagation.
  * The end-to-end engineering effort is substantial: 10-30% of training compute is spent on data, not the model itself.
  * The reflex: when a model underperforms despite good architecture and tuning, the data is usually the suspect. Audit the pipeline; check for under-dedup, under-filtered junk, undercaught contamination, or domain imbalance.

Module 36 (if Part X continues) would tackle **mechanistic interpretability** — sparse autoencoders, feature circuits, and the engineering of interpretability at production scale. Fast becoming a required skill with no good engineering-level treatment elsewhere.
