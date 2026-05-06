# Module 10 — Datasets & DataLoaders done right

# Datasets & DataLoaders, _done right_

_Part IV · Module 10_

— the producer-consumer pipeline that feeds your GPU, the `__getitem__` trap that wastes 40% of training time, and the worker-RNG bug that breaks reproducibility silently

\--- 

The data loader is the part of PyTorch that nobody photographs and everybody underrates. You spent two modules tuning your optimizer and three more on autograd, and now your GPU sits at 30% utilization because you wrote one slow line in `__getitem__`. Welcome to the unsexy half of training performance.

The good news is that the data pipeline has a small surface area: a `Dataset` that produces samples, a `DataLoader` that batches them, a few worker subprocesses that parallelize `__getitem__`, and pinned-memory hand-off to the GPU. The bad news is that every one of those four pieces has a way to be silently slow or silently wrong. This module teaches you the failure modes.

> **★ KEY IDEA**  
>  A `DataLoader` is a **producer-consumer pipeline**. The producers are `num_workers` child processes, each running `__getitem__` on the indices a Sampler hands them. The consumer is your training loop. The pipeline buffer is set by `prefetch_factor`. The hand-off to GPU is sped up by `pin_memory=True`. Get any one of those wrong and the GPU sits idle while training drags. Get them all right and the data pipeline disappears into the background. 

## Three new players

DL

DataLoader

"I'm the foreman. I orchestrate the workers, batch their outputs, and hand things to the GPU."

When you wrap me around a Dataset, I forge a small army of subprocesses (Workers), give them indices to fetch, collect their results, batch them with `collate_fn`, and (if you asked) pin the memory before yielding. `num_workers` sets the army size. `prefetch_factor` sets how far ahead I let them work. `persistent_workers` tells me whether to keep them alive between epochs. Get those three right and your GPU never starves.

W

Worker

"I'm one of `num_workers` child processes. I do the slow stuff so the main process can keep the GPU busy."

I get spawned at DataLoader iteration start (or once, if `persistent_workers=True`). I receive indices, run `__getitem__` for each, return the results. I have my own copy of the Dataset, my own RNG state (which you have to handle — that's the bug), and my own memory. _I don't share GPU access_ — one of you only at the very end, on the consumer side.

S

Sampler

"I decide what indices to fetch, in what order. That's it."

Default is `SequentialSampler` (0, 1, 2, ...) when `shuffle=False`, or `RandomSampler` (a random permutation) when `shuffle=True`. For distributed training there's `DistributedSampler` that gives each rank a disjoint slice. For weighted/balanced sampling there's `WeightedRandomSampler`. They all just produce indices; the Workers do the loading.

## The pipeline, in one diagram

DataLoader: producer-consumer pipeline Sampler produces indices [7, 12, 3, 4, ...] Worker 0 __getitem__(7) → sample Worker 1 __getitem__(12) → sample Worker 2 __getitem__(3) → sample (num_workers=3) collate_fn batch list of samples pin_memory page-lock GPU async DMA prefetch_factor=2 → each worker keeps 2 batches "in flight" if any link is slower than the GPU, the GPU stalls. find the bottleneck. fix the bottleneck.

That's the whole picture. Read it left to right: a Sampler emits indices, the indices are distributed across Workers running in subprocesses, each Worker calls `__getitem__` and returns a sample, the main process gathers samples into batches via `collate_fn`, optionally pins the result for fast GPU transfer, and your training loop consumes from the queue.

_Every parameter you tune in`DataLoader(...)` controls one of those arrows._

## Datasets: map-style vs iterable-style

Two different abstractions for "a thing that yields samples." Each fits a different data shape.

### Map-style: random access

A dataset is a thing with `__len__` and `__getitem__`. The DataLoader can ask "give me sample 42." This is the right model when your data is finite and indexable — files in a folder, rows in a database, examples in a parquet file.
    
    
    from torch.utils.data import Dataset
    
    class ImageDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform
    
        def __len__(self):
            return len(self.paths)
    
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('RGB')
            return self.transform(img), self.parse_label(idx)

That's the canonical map-style Dataset. Note three things: (1) the constructor stores indexable references (paths, not images), (2) `__getitem__` does the actual I/O and transform per call, (3) the return is whatever you want — a tuple, a dict, a custom class. `collate_fn` will figure out how to batch it.

### Iterable-style: streaming

For data that doesn't fit on disk, comes from a network stream, or has unknown size. Your dataset implements `__iter__`, not `__getitem__`. The DataLoader just calls `iter()` and pulls samples until exhausted.
    
    
    from torch.utils.data import IterableDataset
    
    class StreamingDataset(IterableDataset):
        def __init__(self, url):
            self.url = url
    
        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            # split the stream across workers — each worker gets a disjoint slice
            if worker_info is None:
                shards = all_shards(self.url)
            else:
                shards = all_shards(self.url)[worker_info.id::worker_info.num_workers]
            for shard in shards:
                for sample in decode_shard(shard):
                    yield sample

The catch with iterable datasets: you have to _shard the stream across workers yourself_. Otherwise every worker iterates the same data and you get duplicates. The `get_worker_info()` call exposes the worker id and total worker count, so you can do `shards[worker_id::num_workers]` as the standard pattern.

When to use which | Map-style| Iterable-style  
---|---|---  
Total size known?| Yes — `len(ds)`| Often unknown  
Random access?| Yes — any index| No — sequential only  
Shuffling| Free — Sampler does it| Manual — buffer + shuffle  
Worker sharding| Automatic via Sampler| You write it in `__iter__`  
Use for| Local files, databases, parquet| Streaming web data, infinite generators  
  
For >95% of training, map-style is what you want. Iterable-style shines for big-corpus LLM pretraining where the data lives in remote shards and you can't afford to materialize an index.

## The `__getitem__` trap

This is the single most common cause of slow training. Watch:
    
    
    class SlowDataset(Dataset):
        def __init__(self, csv_path):
            self.df = pd.read_csv(csv_path)              # load metadata once — fine
    
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row['path'])                  # I/O per call — fine
            img = img.convert('RGB').resize((224, 224))   # compute per call — fine
            big_lookup = pd.read_parquet('metadata.parquet')  # 🔥 every call!
            meta = big_lookup[big_lookup.id == row['id']]      # 🔥 full scan!
            return img, meta

Spot the problem? Two of those lines run _every time the loader fetches a sample_. Across an epoch of 100k samples, you're reading the same parquet file 100,000 times. Even with workers, you waste minutes per epoch.

The fix: _everything that doesn't depend on`idx` belongs in `__init__`, not in `__getitem__`._
    
    
    class FastDataset(Dataset):
        def __init__(self, csv_path):
            self.df = pd.read_csv(csv_path)
            big_lookup = pd.read_parquet('metadata.parquet')   # once
            self.meta = big_lookup.set_index('id').to_dict('index')
            # Now meta lookup is O(1) per call
    
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row['path']).convert('RGB').resize((224, 224))
            meta = self.meta[row['id']]
            return img, meta

The rule of thumb: profile `__getitem__` with a Python time-it on a single call. If it takes more than a few hundred microseconds for in-memory work or more than a few milliseconds for I/O-bound work, you've got a hot path. Move the per-call constants to `__init__`.

> **⚠ WARNING**  
>  **The pandas trap.** `df.iloc[idx]` is fast. But `df[df.id == x]` is a full scan — O(n) per call. Ditto for `df.loc[df.col == x]`. If you're indexing by anything other than position, build a dict (`set_index().to_dict()`) in `__init__` for O(1) lookups. This single change has saved more training-time-wasted-on-data-loading hours than any other. 

## num_workers, prefetch_factor, persistent_workers

Three knobs that work together. They control the producer side of the pipeline.

### `num_workers`: how many subprocess loaders
    
    
    loader = DataLoader(dataset, batch_size=64, num_workers=4)

Sets the number of _subprocesses_ that run `__getitem__`. `num_workers=0` means "do it in the main process" — synchronous, slow, but easy to debug.

Tuning: it's not "more workers = faster." Past a point, you saturate disk I/O or CPU and adding workers just adds overhead. Rule of thumb:

  * **I/O-bound dataset** (loading images from disk): start with `num_workers = 4 to 8`, scale up to where I/O saturates.
  * **CPU-bound dataset** (heavy augmentation, decoding): `num_workers ≈ os.cpu_count()`, capped at 16 or so.
  * **Memory-tight** : each worker has a full copy of the Dataset object's state. If your `__init__` loaded a 5 GB lookup table, each worker has 5 GB. Cap workers accordingly.

The right answer is usually 4-8. Most code-bases that I've seen with `num_workers=16` don't get faster than `num_workers=8` — they just use more RAM.

### `prefetch_factor`: how far ahead each worker runs
    
    
    loader = DataLoader(dataset, batch_size=64, num_workers=4, prefetch_factor=2)
    # Each worker keeps 2 batches "ready in advance" → 8 batches buffered total

Default is 2. Increasing it (e.g., to 4) buffers more batches in advance, smoothing out variable-latency datasets at the cost of more RAM. Decreasing to 1 saves RAM but makes the GPU more sensitive to per-sample slowness.

### `persistent_workers`: don't kill them between epochs
    
    
    loader = DataLoader(dataset, batch_size=64, num_workers=4, persistent_workers=True)

By default, workers spawn at the start of every epoch and die at the end. If your `Dataset.__init__` is expensive (e.g., loading a big index), this is a per-epoch tax. `persistent_workers=True` keeps the workers alive across epochs. **Recommended for almost all cases** — the only downside is that worker state persists, which is sometimes a debugging nuisance.

## collate_fn: how samples become batches

`collate_fn` takes a list of samples (whatever `__getitem__` returned) and produces a batch. The default is `default_collate`, which is smart enough for most cases:
    
    
    # If __getitem__ returns (tensor, int), default_collate produces (batched_tensor, batched_int)
    # If __getitem__ returns dict, default_collate produces dict-of-batched-values
    # If shapes vary across samples — defaults BREAKS, you need a custom collate

The classic case for a custom collate is variable-length sequences. Your dataset returns tokenized text of different lengths. The default collate tries to `torch.stack` them and fails because shapes don't match. You write a custom collate that pads:
    
    
    def pad_collate(samples, pad_value=0):
        # samples: list of (tokens, label), each with variable-length tokens
        tokens, labels = zip(*samples)
        max_len = max(t.size(0) for t in tokens)
        padded = torch.full((len(tokens), max_len), pad_value, dtype=tokens[0].dtype)
        lengths = torch.tensor([t.size(0) for t in tokens])
        for i, t in enumerate(tokens):
            padded[i, :t.size(0)] = t
        return padded, torch.stack(labels), lengths
    
    loader = DataLoader(dataset, batch_size=32, collate_fn=pad_collate)

Notes: (1) `pad_value` is dataset-specific — for transformers, often the tokenizer's pad token id; (2) returning `lengths` alongside is a useful idiom because downstream code can build a mask from it (M2's `positions[None, :] < lengths[:, None]` trick); (3) for batches of dicts of tensors, write a generic dict-aware collate that handles each key.

### Bucketing: a smart collation pattern

If your sequences vary wildly in length, padding everything to the longest sample wastes compute on padding. Better: sort/group similar-length samples into the same batch. The standard tool is a `BatchSampler` that groups indices by length, plus your collate doing only minor padding within each batch.
    
    
    class LengthBucketSampler(torch.utils.data.Sampler):
        def __init__(self, lengths, batch_size):
            self.batches = []
            # sort by length, then chunk into batches
            sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
            for i in range(0, len(sorted_indices), batch_size):
                self.batches.append(sorted_indices[i:i+batch_size])
            random.shuffle(self.batches)              # shuffle order, not contents
    
        def __iter__(self):
            return iter(self.batches)
    
        def __len__(self):
            return len(self.batches)
    
    loader = DataLoader(dataset, batch_sampler=bucket_sampler, collate_fn=pad_collate)

Notice `batch_sampler`, not `sampler` — the batch sampler yields _lists of indices_ , one per batch. Used in NLP training pipelines for huge throughput wins (often 2-3× faster than naive padding).

## Pin memory in the DataLoader

From M3: pinned memory is page-locked CPU memory, allowing async DMA transfers to GPU. The DataLoader does this for you with one flag:
    
    
    loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
    
    # Then in your training loop:
    for x, y in loader:
        x = x.to('cuda', non_blocking=True)         # async transfer, only useful with pin_memory
        y = y.to('cuda', non_blocking=True)
        out = model(x)
        ...

Two things: (1) `pin_memory=True` alone does nothing — you also need `non_blocking=True` on the `.to()` for the async-ness to kick in; (2) it's a small but real win, typically 5-15% throughput on data-loading-bound training. Free if you can afford the (small) pinned-memory pool.

> **⚠ WARNING**  
>  **Don't pin if you have nothing to gain.** Pinning adds a copy step (allocate pinned memory, copy from worker's regular memory). If you're bottlenecked on compute and the GPU is at 95% utilization, pinning won't help and might marginally hurt. Profile first (Module 13). 

## Worker RNG: the silent reproducibility bug

Here's a bug most teams ship without noticing. You set `torch.manual_seed(42)`, you run training twice, you get different results. What gives?

Each Worker is a child process. Child processes inherit a copy of the parent's RNG state at fork time, but PyTorch's default behavior depends on the start method (`fork` vs `spawn`) and the platform. On many setups, all workers end up with the _same RNG state_ , producing the same random augmentations on the same indices — yes, that's a bug. On other setups, workers have _independent random state derived from system entropy_ , which is fine for correctness but breaks reproducibility.

The fix: a `worker_init_fn` that explicitly seeds each worker:
    
    
    def seed_worker(worker_id):
        # Each worker gets a deterministic but different seed
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(42)
    
    loader = DataLoader(
        dataset, batch_size=64, num_workers=4,
        worker_init_fn=seed_worker, generator=g,
    )

The `generator` argument seeds the loader's main-process RNG (used by samplers); `worker_init_fn` seeds each worker's auxiliary RNGs (NumPy, Python's `random`). With both set, each worker has a unique-but-deterministic seed derived from the loader's master seed plus the worker id.

Without this dance, your training is non-reproducible _even with all the seeds set in the main process_. Most teams ship without it and never notice; if you care about reproducibility (you should, for science and for debugging), set it.

## The DistributedSampler for multi-GPU

For DDP training (Module 16), each rank needs its own _disjoint_ slice of the dataset. `DistributedSampler` does this:
    
    
    from torch.utils.data.distributed import DistributedSampler
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4)
    
    # Important: call set_epoch each epoch so the shuffle pattern changes
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)             # otherwise every epoch sees the SAME shuffle
        for x, y in loader:
            ...

Two gotchas: (1) `shuffle` goes on the _sampler_ , not the DataLoader; passing `shuffle=True` to DataLoader with a custom sampler errors out. (2) The `set_epoch` call is mandatory — otherwise every epoch shuffles the same way and the model sees the same effective ordering. Forgetting this is a real bug.

## Code Magnets: build the production training DataLoader

You're setting up the data pipeline for transformer training. Variable-length sequences (so you need a pad collate), 4 workers, pinned memory, persistent workers across epochs, deterministic seeding.

Arrange the magnets into a working DataLoader setup. Three are red herrings.

def seed_worker(worker_id): worker_seed = torch.initial_seed() % 2**32 np.random.seed(worker_seed); random.seed(worker_seed) g = torch.Generator(); g.manual_seed(42) loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True, persistent_workers=True, collate_fn=pad_collate, shuffle=True, worker_init_fn=seed_worker, generator=g) sampler=RandomSampler(dataset), prefetch_factor=16,

show solution
    
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed); random.seed(worker_seed)
    
    g = torch.Generator(); g.manual_seed(42)
    
    loader = DataLoader(dataset, batch_size=32, num_workers=4,
        pin_memory=True, persistent_workers=True,
        collate_fn=pad_collate,
        shuffle=True,
        worker_init_fn=seed_worker, generator=g)

The traps:

  * `sampler=RandomSampler(dataset)`: redundant with `shuffle=True`, and passing both errors. Pick one. For single-GPU shuffle, `shuffle=True` is enough.
  * `prefetch_factor=16`: way too high. Default 2 is almost always right; 4-8 if you have variable-latency I/O. 16 mostly just wastes RAM.

The minimum reproducible production setup is exactly what's left: pin, persistent, collate, shuffle, seed. Everything else is tuning.

## Who does what?

Match each DataLoader knob/concept to its real job.

Knob / concept

Real job

num_workers

A. Pads variable-length samples or otherwise turns a list of samples into a batch.

prefetch_factor

B. Decides which indices to fetch in what order.

persistent_workers

C. Number of subprocesses running __getitem__ in parallel.

pin_memory

D. Page-locks the batch in CPU memory so GPU DMA can read async.

collate_fn

E. How many batches each worker keeps "ready in advance" — buffer depth.

Sampler

F. Keeps workers alive across epochs (avoids re-running expensive __init__).

worker_init_fn

G. Per-worker hook to seed RNGs and ensure each worker has unique-but-deterministic state.

show solution

**num_workers** → C  
**prefetch_factor** → E  
**persistent_workers** → F  
**pin_memory** → D  
**collate_fn** → A  
**Sampler** → B  
**worker_init_fn** → G 

The mental shortcut: _num_workers = parallelism, prefetch = buffer depth, persistent = lifetime across epochs, pin = async DMA, collate = batching, sampler = ordering, worker_init = RNG hygiene_.

## Exercises

> **✎ SHARPEN YOUR PENCIL** > > 

**1.** A team's training loop is GPU-bound for the first 1000 steps and then suddenly becomes data-bound. What's the most likely cause?

show answer

The first epoch's `__getitem__` calls populated the OS file cache, so I/O was fast. After warmup, something else changed — most likely the workers died (no `persistent_workers=True`) and the next epoch had to respawn them, re-loading the Dataset's `__init__` in each worker. If `__init__` reads big files or builds big indexes, this is a per-epoch tax. Set `persistent_workers=True` and the issue typically disappears.

> **✎ SHARPEN YOUR PENCIL** > > 

**2.** You're training a vision model and notice that with `num_workers=8` the GPU sits at ~70% utilization. Doubling to 16 doesn't help. What should you check?

show answer

Likely a slow `__getitem__` bottleneck that's not parallel. Time a single `__getitem__` call. If it's slow because of CPU augmentation (e.g., heavy PIL transforms), more workers eventually saturate the CPU; consider GPU-side augmentation (kornia, NVIDIA DALI). If it's slow because of disk I/O, more workers don't help past disk bandwidth — switch to a faster format (webdataset, parquet) or pre-decode to a faster representation. The next-step diagnosis tool is `torch.profiler` from M13.

> **✎ SHARPEN YOUR PENCIL** > > 

**3.** Implement a custom collate that handles a dataset where `__getitem__` returns `{'image': tensor, 'caption': str, 'metadata': dict}`. Tensors should stack, strings should become a list, dicts should remain a list of dicts.

show answer
    
    
    def multimodal_collate(samples):
        images = torch.stack([s['image'] for s in samples])
        captions = [s['caption'] for s in samples]
        metadata = [s['metadata'] for s in samples]
        return {'image': images, 'caption': captions, 'metadata': metadata}

The default collate would _try_ to stack the captions and choke. The pattern: explicitly stack the things that should be tensors, and pass through the rest as lists. For more complex cases (variable-shape tensors), pad in the collate.

> **✎ SHARPEN YOUR PENCIL** > > 

**4.** A team's distributed training run is wasting 30% of every epoch on a synchronization stall. They use `DistributedSampler` but don't call `set_epoch`. What's happening, and is the stall related?

show answer

The stall is probably unrelated (likely a slow `__getitem__` on some ranks or imbalanced data shards), but _not calling`set_epoch`_ is its own bug. Without it, every epoch's shuffle pattern is the same — the model sees the same effective sample order every epoch, defeating the point of shuffling. This doesn't directly cause stalls but makes training less effective. Always call `sampler.set_epoch(epoch)` at the start of each training epoch.

### What just happened?

  * A `DataLoader` is a **producer-consumer pipeline** : Sampler emits indices, Workers run `__getitem__`, collate_fn batches, pin_memory enables async GPU transfer.
  * **Map-style datasets** (`__len__` \+ `__getitem__`) are the default; **iterable-style** for streaming/infinite/sharded data.
  * **The`__getitem__` trap**: anything that doesn't depend on `idx` belongs in `__init__`, not in `__getitem__`. Especially: don't read big files or do full DataFrame scans per call.
  * Index DataFrames by position (`iloc`) or by pre-built dict, not by value-equality scans.
  * `num_workers` = 4-8 for most setups. More than that usually saturates the next bottleneck (disk, CPU). Each worker has a full Dataset copy — memory matters.
  * `prefetch_factor=2` default is fine. Bump to 4-8 only for variable-latency I/O.
  * `persistent_workers=True` avoids the per-epoch worker respawn tax. Almost always the right setting.
  * `collate_fn` turns a list of samples into a batch. Default works for fixed shapes; write a custom one for padding, multimodal data, or anything weird.
  * **Bucketing by length** via `batch_sampler` is the standard NLP trick — 2-3× throughput by avoiding wasted padding compute.
  * `pin_memory=True` \+ `.to('cuda', non_blocking=True)` = async DMA hand-off. Free 5-15% in data-bound training.
  * **Worker RNG is the silent reproducibility bug.** Set `worker_init_fn` \+ `generator` on the DataLoader. Otherwise seeds in the main process don't propagate predictably.
  * **DistributedSampler** : each rank sees a disjoint slice. Always call `sampler.set_epoch(epoch)` before each epoch — otherwise shuffling is identical every epoch.
  * The reflex: when training is slow, time `next(iter(loader))` in isolation. If that's slow, your data pipeline is bound; the diagnosis tree branches into worker count, `__getitem__` hot lines, and collate cost.

Module 11 closes Part IV with the training loop itself: how to wire all of this together into something you can resume from a checkpoint, debug with overfit-a-batch, and run for a million steps without it falling apart.
