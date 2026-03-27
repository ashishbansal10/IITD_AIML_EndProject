"""
data_splitter.py
================
Few-Shot Learning Data Pipeline — Split, Sample, Load.

Classes
-------
OffsetFetcher               Lazy global-index fetch via prebuilt offset table
StreamSplitWrapper          Streaming virtual splits via MD5 hashing
StaticSampleSplitter        Sample-level slicing for all dstypes
FewShotClassSplitter        THE splitter — class-level + sample-level split
TransformedPoolDataset      Lazy dataset wrapper — map-style and streaming
EpisodicBatchSampler        N-way K-shot episode sampler
TaskCollator                Reshapes flat episode → {support, query, target}
SmartDataLoaderFactory      Builds DataLoaders for any pool + mode

Pool Names (produced by FewShotClassSplitter)
---------------------------------------------
pretrain    64 base classes, sample subset A  — batch, both paradigms
train       64 base classes, sample subset B  — batch (standard) / episodic (fewshot)
val_seen    64 base classes, sample subset C  — batch validation (standard)
test        64 base classes, sample subset D  — seen-class accuracy, both paradigms
val_unseen  16 val classes,  all samples      — episodic meta-val (fewshot)
novel       20 novel classes, all samples     — unseen-class accuracy, both paradigms

Default Split Dict
------------------
split = {
    'base': {
        'pretrain': 0.4,
        'train':    0.3,
        'val_seen': 0.2,
        'test':     0.1
    },
    'val_unseen': 0.16,
    'novel':      0.20
}

High Level Usage
----------------
# 1. Load data
ds = DataSource.load('hf', 'mini-imagenet', ...)

# 2. Define transforms
transformers = {'train': train_transform, 'eval': eval_transform}

# 3. Split
splitter = FewShotClassSplitter()
splitter.apply(data_source=ds, split={...}, random_state=42)
print(splitter)

# 4. Factory
factory = SmartDataLoaderFactory(splitter, transformers, device=device)

# 5. Get loaders
pretrain_loader  = factory.get_loader('pretrain',    mode='batch',    batch_size=64)
train_loader     = factory.get_loader('train',       mode='batch',    batch_size=64)    # standard
train_loader     = factory.get_loader('train',       mode='episodic', n=5, k=5, q=15)  # fewshot
val_seen_loader  = factory.get_loader('val_seen',    mode='batch',    batch_size=64)    # standard
val_unseen_loader= factory.get_loader('val_unseen',  mode='episodic', n=5, k=5, q=15)  # fewshot
test_loader      = factory.get_loader('test',        mode='batch',    batch_size=64)
novel_loader     = factory.get_loader('novel',       mode='episodic', n=5, k=5, q=15)
"""

import hashlib
import random
import math
from collections import defaultdict
from collections.abc import Mapping
import torch
from torch.utils.data import Sampler, Dataset, DataLoader


# ==============================================================================
# OffsetFetcher
# ==============================================================================

class OffsetFetcher:
    """
    Lazy fetch function using a prebuilt offset table.
    Built once from DataSource resources at splitter apply() time.
    Used by TransformedPoolDataset for O(n_splits) global index lookup.

    Completely independent of splitting logic — can be passed around freely.

    Args:
        data_source : Initialized DataSource object with populated resources.

    Usage:
        fetcher = OffsetFetcher(ds)
        raw_sample = fetcher(global_idx)   # lazy — no data loaded at construction
    """

    def __init__(self, data_source):
        self._table = []
        offset = 0
        for split_name, dataset in data_source.resources.items():
            if dataset is None:
                continue
            length = len(dataset)
            self._table.append((offset, offset + length, dataset))
            offset += length

        if not self._table:
            raise RuntimeError(
                "OffsetFetcher: DataSource has no loaded resources. "
                "Ensure DataSource.load() was called successfully."
            )

    def __call__(self, global_idx):
        """
        Fetches raw sample for global_idx.
        Returns None if index is out of range.
        """
        for start, end, dataset in self._table:
            if start <= global_idx < end:
                return dataset[global_idx - start]
        return None

    @property
    def total_samples(self):
        """Total number of samples across all resource splits."""
        return self._table[-1][1] if self._table else 0

    def __repr__(self):
        splits = len(self._table)
        total  = self.total_samples
        return f"OffsetFetcher(splits={splits}, total_samples={total})"


# ==============================================================================
# StreamSplitWrapper
# ==============================================================================

class StreamSplitWrapper:
    """
    Lazy virtual split for streaming datasources.
    Routes samples deterministically into named splits via MD5 hashing.
    Boundary always reset — independent of logger presence.

    Args:
        source_iterable : Iterable streaming dataset.
        target_name     : Name of the split this wrapper represents.
        split_cfg       : {split_name: ratio} covering all splits.
        random_state    : Seed for deterministic routing.
        logger          : Optional logger.
    """

    def __init__(self, source_iterable, target_name, split_cfg,
                 random_state=None, logger=None):
        self.source_iterable = source_iterable
        self.target_name     = target_name
        self.seed            = str(random_state if random_state is not None else 42)

        self.low  = 0.0
        self.high = 0.0
        found     = False
        curr      = 0.0

        for name, prob in split_cfg.items():
            if name == target_name:
                self.low  = curr
                self.high = curr + prob
                found     = True
                break
            curr += prob

        if not found:
            self.low  = 0.0
            self.high = 0.0
            if logger:
                logger.warning(
                    f"StreamSplitWrapper: '{target_name}' not in split_cfg. "
                    f"Results will be empty."
                )

    def _get_hash_score(self, identifier):
        combined = f"{self.seed}_{identifier}".encode('utf-8')
        hash_hex = hashlib.md5(combined).hexdigest()
        return int(hash_hex[:16], 16) / 0xFFFFFFFFFFFFFFFF

    def __iter__(self):
        for i, sample in enumerate(self.source_iterable):
            uid = i
            if isinstance(sample, dict):
                uid = sample.get('id') or sample.get('guid') or sample.get('path') or i
            elif hasattr(sample, 'id'):
                uid = sample.id
            if self.low <= self._get_hash_score(uid) < self.high:
                yield sample

    def __len__(self):
        if hasattr(self.source_iterable, '__len__'):
            return int(len(self.source_iterable) * (self.high - self.low))
        raise TypeError(
            f"Source {type(self.source_iterable)} does not support __len__."
        )

    def __repr__(self):
        return (f"StreamSplitWrapper(target='{self.target_name}', "
                f"range=[{self.low:.3f}, {self.high:.3f}))")


# ==============================================================================
# StaticSampleSplitter
# ==============================================================================

class StaticSampleSplitter:
    """
    Sample-level slicing for in-memory / map-style datasets.
    Dispatches to framework-specific logic per dstype.

    Supported dstypes:
        hfdataset, torchdataset, pygdataset, sklearndataset,
        fsdataset, pandasdf, numpy, generic list/array.
    """

    @staticmethod
    def split(data, split_cfg, dstype=None, random_state=None, logger=None):
        """
        Splits data into named subsets by ratio.

        Args:
            data         : Dataset object.
            split_cfg    : {name: ratio} — ratios must sum to 1.0.
            dstype       : DataSource dstype string.
            random_state : Shuffle seed before slicing.
            logger       : Optional logger.
        Returns:
            dict of {name: data_subset}
        """
        if logger:
            logger.info(
                f"StaticSampleSplitter: dstype={dstype or type(data).__name__}, "
                f"split={split_cfg}"
            )

        if dstype == "hfdataset" or hasattr(data, 'select'):
            if random_state is not None:
                data = data.shuffle(seed=random_state)
            return StaticSampleSplitter._slice_generic(data, split_cfg, method="hf")

        if dstype in ["torchdataset", "pygdataset"] or 'torch' in type(data).__module__:
            return StaticSampleSplitter._split_torch(data, split_cfg, random_state, logger)

        if dstype == "sklearndataset":
            from sklearn.utils import resample
            if random_state is not None:
                data = resample(data, replace=False, n_samples=len(data),
                                random_state=random_state)
            return StaticSampleSplitter._slice_generic(data, split_cfg, method="sklearn")

        if dstype == "fsdataset" or (isinstance(data, list) and dstype is None):
            data_list = list(data)
            if random_state is not None:
                random.Random(random_state).shuffle(data_list)
            return StaticSampleSplitter._slice_generic(data_list, split_cfg, method="standard")

        if dstype == "pandasdf" or hasattr(data, 'iloc'):
            if random_state is not None:
                data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
            return StaticSampleSplitter._slice_generic(data, split_cfg, method="pandas")

        if dstype == "numpy" or type(data).__module__.startswith('numpy'):
            import numpy as np
            if random_state is not None:
                data = np.random.default_rng(random_state).permutation(data)
            return StaticSampleSplitter._slice_generic(data, split_cfg, method="numpy")

        # Fallback
        return StaticSampleSplitter._slice_generic(data, split_cfg, method="standard")

    @staticmethod
    def _slice_generic(data, split_cfg, method="standard"):
        results = {}
        total   = (len(data) if not isinstance(data, Mapping)
                   else len(next(iter(data.values()))))
        curr    = 0
        items   = list(split_cfg.items())

        for i, (name, p) in enumerate(items):
            end = total if i == len(items) - 1 else curr + int(round(total * p))

            if method == "hf":
                results[name] = data.select(range(curr, end))
            elif method == "pandas":
                results[name] = data.iloc[curr:end]
            elif isinstance(data, Mapping):
                sliced = {k: v[curr:end] for k, v in data.items()}
                if type(data).__name__ == "Bunch":
                    from sklearn.utils import Bunch
                    results[name] = Bunch(**sliced)
                else:
                    results[name] = sliced
            else:
                results[name] = data[curr:end]

            curr = end
        return results

    @staticmethod
    def _split_torch(data, split_cfg, random_state, logger):
        from torch.utils.data import random_split
        total   = len(data)
        names   = list(split_cfg.keys())
        lengths = []

        for i, (name, p) in enumerate(split_cfg.items()):
            lengths.append(
                max(0, total - sum(lengths)) if i == len(split_cfg) - 1
                else int(round(total * p))
            )

        if sum(lengths) != total:
            lengths[-1] += total - sum(lengths)

        generator = (torch.Generator().manual_seed(random_state)
                     if random_state is not None else None)
        subsets   = random_split(data, lengths, generator=generator)
        return {names[i]: s for i, s in enumerate(subsets)}


# ==============================================================================
# FewShotClassSplitter
# ==============================================================================

class FewShotClassSplitter:
    """
    Single splitter serving both Standard and FewShot paradigms.

    Split pipeline:
        1. Validate split dict       — 2-level mixed dict rules
        2. Detect DataSource type    — class_wide vs sample_wide
        3. Build OffsetFetcher       — lazy fetch, built once
        4. Build global index map    — {class_id: [global_indices]}
        5. Filter low-count classes  — remove classes below min_samples_per_class
        6. Assign class-level pools  — base / val_unseen / novel
        7. Split base samples        — pretrain / train / val_seen / test

    Exposes for SmartDataLoaderFactory:
        get_indices(pool_name)           → flat list of global indices
        get_class_to_indices(pool_name)  → {class_id: [indices]} for episodic
        get_fetch_fn()                   → OffsetFetcher instance
        pool_names()                     → list of valid pool names

    Split dict format (2-level mixed):
        {
            'base': {                   ← nested dict = base classes, sample split
                'pretrain': 0.4,        ← sample ratios within base, must sum to 1.0
                'train':    0.3,
                'val_seen': 0.2,
                'test':     0.1
            },
            'val_unseen': 0.16,         ← float = class-level ratio
            'novel':      0.20          ← float = class-level ratio
        }
        # 'base' class ratio = 1.0 - 0.16 - 0.20 = 0.64 (implicit)

    Note: Streaming DataSources not supported — episodic sampling requires
          random access which streaming cannot provide.
    """

    DEFAULT_SPLIT = {
        'base': {
            'pretrain': 0.4,
            'train':    0.3,
            'val_seen': 0.2,
            'test':     0.1
        },
        'val_unseen': 0.16,
        'novel':      0.20
    }

    def __init__(self):
        self.ds                    = None
        self.split                 = None
        self.random_state          = None
        self.logger                = None
        self.min_samples_per_class = 20

        # OffsetFetcher — built at apply() time
        self._fetcher              = None

        # Global map: class_id → [global_indices] across all HF resource splits
        self.class_to_indices      = defaultdict(list)

        # Class-level pools: 'base' / 'val_unseen' / 'novel' → [class_ids]
        self._class_pools          = {}

        # Sample-level pools within base:
        # pool_name → {class_id: [global_indices]}
        # Keys: 'pretrain', 'train', 'val_seen', 'test'
        self._sample_pools         = {}

    # ------------------------------------------------------------------
    # apply — main entry point
    # ------------------------------------------------------------------

    def apply(self, data_source, split=None, random_state=None, logger=None,
              min_samples_per_class=20, **config):
        """
        Performs the full split pipeline.

        Args:
            data_source           : Initialized DataSource object.
            split                 : 2-level mixed split dict.
                                    Default: FewShotClassSplitter.DEFAULT_SPLIT
            random_state          : Reproducibility seed.
            logger                : Optional logger.
            min_samples_per_class : Classes with fewer samples filtered out.
            **config              : Reserved for future use.
        Returns:
            self
        """
        self.ds                    = data_source
        self.random_state          = random_state
        self.logger                = logger
        self.min_samples_per_class = min_samples_per_class

        # Reset — safe for re-entrant calls
        self._fetcher           = None
        self.class_to_indices   = defaultdict(list)
        self._class_pools       = {}
        self._sample_pools      = {}

        # 1. Validate split dict
        self.split = self._validate_split(split or self.DEFAULT_SPLIT)

        # 2. Reject streaming — incompatible with episodic
        if data_source.is_stream:
            raise NotImplementedError(
                "FewShotClassSplitter does not support streaming DataSources. "
                "Episodic sampling requires random index access."
            )

        # 3. Build OffsetFetcher — lazy fetch, O(n_splits) per call
        self._fetcher = OffsetFetcher(data_source)
        self._log(f"OffsetFetcher built: {self._fetcher}")

        # 4. Build global class → indices map
        self._build_global_map()

        # 5. Filter classes with insufficient samples
        self._filter_low_count_classes()

        # 6. Assign classes to base / val_unseen / novel
        ds_structure = self._detect_datasource_structure()
        self._log(f"DataSource structure detected: {ds_structure}")

        if ds_structure == 'class_wide':
            self._assign_class_pools_from_hf()
        else:
            self._assign_class_pools_by_ratio()

        # 7. Sample-split base classes into pretrain / train / val_seen / test
        self._split_base_samples()

        self._log(
            f"Split complete. Pools: {self.pool_names()} | "
            f"Classes — base: {len(self._class_pools.get('base', []))}, "
            f"val_unseen: {len(self._class_pools.get('val_unseen', []))}, "
            f"novel: {len(self._class_pools.get('novel', []))}"
        )
        return self

    # ------------------------------------------------------------------
    # Split dict validation
    # ------------------------------------------------------------------

    def _validate_split(self, split):
        """
        Validates 2-level mixed split dict.

        Rules:
            - Exactly ONE key has a nested dict → base classes with sample split
            - All other keys must be floats → class-level ratios
            - Nested dict values must sum to 1.0 → sample ratios within base
            - Explicit float ratios must sum to < 1.0 → room for implicit base ratio
            - Implicit base ratio = 1.0 - sum(explicit float ratios)

        Returns validated split with '_class_ratio' injected into nested dict.
        """
        if not isinstance(split, dict) or len(split) == 0:
            raise ValueError("split must be a non-empty dict.")

        dict_keys  = [k for k, v in split.items() if isinstance(v, dict)]
        float_keys = [k for k, v in split.items() if isinstance(v, float)]

        if len(dict_keys) == 0:
            raise ValueError(
                "split must contain exactly one nested dict (base class sample split). "
                "Got all float values. "
                "Example: split={'base': {'pretrain':0.4,'train':0.3,'val_seen':0.2,'test':0.1}, "
                "'val_unseen':0.16, 'novel':0.20}"
            )

        if len(dict_keys) > 1:
            raise ValueError(
                f"split must contain exactly ONE nested dict. "
                f"Got {len(dict_keys)}: {dict_keys}"
            )

        # Validate nested sample ratios sum to 1.0
        nested_key  = dict_keys[0]
        nested_dict = split[nested_key]
        sample_sum  = sum(nested_dict.values())
        if abs(sample_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Sample ratios under '{nested_key}' must sum to 1.0. "
                f"Got {sample_sum:.6f}."
            )

        # Validate explicit class ratios leave room for implicit base
        explicit_sum = sum(split[k] for k in float_keys)
        if explicit_sum >= 1.0:
            raise ValueError(
                f"Explicit class ratios sum to {explicit_sum:.4f} — "
                f"no room for implicit '{nested_key}' class ratio. "
                f"Reduce explicit ratios so they sum to < 1.0."
            )

        # Inject implicit class ratio into nested dict
        implicit_ratio = round(1.0 - explicit_sum, 6)
        validated      = {}
        for k, v in split.items():
            if isinstance(v, dict):
                validated[k] = {'_class_ratio': implicit_ratio, **v}
            else:
                validated[k] = v

        self._log(
            f"Split validated. Class ratios — "
            + ", ".join(
                f"{k}: {v['_class_ratio']:.4f}" if isinstance(v, dict) else f"{k}: {v:.4f}"
                for k, v in validated.items()
            )
        )
        return validated

    # ------------------------------------------------------------------
    # DataSource structure detection
    # ------------------------------------------------------------------

    def _detect_datasource_structure(self):
        """
        Inspects DataSource resources to determine split structure.

        Returns:
            'class_wide'  : each resource split has completely different classes
                            (correct Mini-ImageNet benchmark — 64/16/20 split)
            'sample_wide' : same classes appear across resource splits
                            (timm/mini-imagenet style — overlapping classes)
        """
        try:
            mapper_func = self.ds.get_dataset_label_mapper()
        except RuntimeError:
            self._log("No label mapper found — assuming sample_wide.")
            return 'sample_wide'

        split_class_sets = []
        for split_name, dataset in self.ds.resources.items():
            if dataset is None:
                continue
            local_map = mapper_func(dataset, self.ds.class_to_idx)
            split_class_sets.append(set(local_map.keys()))

        if not split_class_sets:
            return 'sample_wide'

        all_ids    = [c for s in split_class_sets for c in s]
        is_disjoint = len(set(all_ids)) == len(all_ids)
        return 'class_wide' if is_disjoint else 'sample_wide'

    # ------------------------------------------------------------------
    # Global map construction
    # ------------------------------------------------------------------

    def _build_global_map(self):
        """
        Builds {class_id: [global_indices]} across ALL resource splits.
        Global index = local index + cumulative offset.
        Called after OffsetFetcher is built (offset table shared via ds resources).
        """
        mapper_func = self.ds.get_dataset_label_mapper()
        offset      = 0

        for split_name, dataset in self.ds.resources.items():
            if dataset is None:
                continue
            local_map = mapper_func(dataset, self.ds.class_to_idx)
            for class_id, local_indices in local_map.items():
                self.class_to_indices[class_id].extend(
                    [i + offset for i in local_indices]
                )
            offset += len(dataset)

        self._log(
            f"Global map built: {len(self.class_to_indices)} classes, "
            f"{self._fetcher.total_samples} total samples."
        )

    def _filter_low_count_classes(self):
        """Removes classes with fewer than min_samples_per_class samples."""
        initial = len(self.class_to_indices)
        self.class_to_indices = {
            cid: idxs for cid, idxs in self.class_to_indices.items()
            if len(idxs) >= self.min_samples_per_class
        }
        removed = initial - len(self.class_to_indices)
        if removed > 0:
            self._log(
                f"Filtered {removed} classes with < {self.min_samples_per_class} samples. "
                f"Remaining: {len(self.class_to_indices)} classes."
            )

    # ------------------------------------------------------------------
    # Class-level pool assignment
    # ------------------------------------------------------------------

    def _assign_class_pools_from_hf(self):
        """
        Uses HF resource split membership for class assignment.
            resources['train'] → base
            resources['val']   → val_unseen
            resources['test']  → novel
        Falls back to ratio-based if test split missing.
        """
        mapper_func      = self.ds.get_dataset_label_mapper()
        resource_to_pool = {'train': 'base', 'val': 'val_unseen', 'test': 'novel'}
        pool_class_ids   = defaultdict(list)

        for res_name, dataset in self.ds.resources.items():
            if dataset is None:
                continue
            pool_name = resource_to_pool.get(res_name, 'base')
            local_map = mapper_func(dataset, self.ds.class_to_idx)
            for class_id in local_map.keys():
                if class_id in self.class_to_indices:
                    pool_class_ids[pool_name].append(class_id)

        # Fallback if test split missing
        if 'novel' not in pool_class_ids:
            self._log("No test split found in DataSource — falling back to ratio-based class split.")
            return self._assign_class_pools_by_ratio()

        self._class_pools = dict(pool_class_ids)
        self._log(
            f"Class pools (HF membership): "
            f"base={len(self._class_pools.get('base', []))}, "
            f"val_unseen={len(self._class_pools.get('val_unseen', []))}, "
            f"novel={len(self._class_pools.get('novel', []))}"
        )

    def _assign_class_pools_by_ratio(self):
        """
        Random class-level split by ratio for sample_wide sources.
        Uses '_class_ratio' from validated split dict.
        """
        all_class_ids = list(self.class_to_indices.keys())
        rng           = random.Random(self.random_state)
        rng.shuffle(all_class_ids)

        total = len(all_class_ids)

        # Extract class ratios from validated split dict
        class_ratios = {}
        for key, val in self.split.items():
            if isinstance(val, dict):
                class_ratios[key] = val['_class_ratio']
            else:
                class_ratios[key] = val

        # Assign classes in order: novel, val_unseen, base (remainder)
        curr = 0
        items = list(class_ratios.items())
        assigned = {}

        for i, (pool_name, ratio) in enumerate(items):
            n = round(ratio * total)
            end = curr + n
            if i == len(items) - 1:
                assigned[pool_name] = all_class_ids[curr:]
            else:
                assigned[pool_name] = all_class_ids[curr:end]
            curr = end

        self._class_pools = assigned
        self._log(
            f"Class pools (ratio-based): "
            + ", ".join(f"{k}={len(v)}" for k, v in self._class_pools.items())
        )

    # ------------------------------------------------------------------
    # Sample-level split within base
    # ------------------------------------------------------------------

    def _split_base_samples(self):
        """
        Sample-level split of base class indices into pretrain/train/val_seen/test.
        Each class split independently to preserve class balance across pools.
        """
        # Extract sample ratios from nested dict
        sample_cfg = {}
        for key, val in self.split.items():
            if isinstance(val, dict):
                sample_cfg = {k: v for k, v in val.items() if k != '_class_ratio'}
                break

        if not sample_cfg:
            # No sample split — put all base samples in 'train'
            self._sample_pools['train'] = {
                cid: self.class_to_indices[cid]
                for cid in self._class_pools.get('base', [])
            }
            self._log("No sample split config found — all base samples assigned to 'train'.")
            return

        # Initialize pool dicts
        for pool_name in sample_cfg:
            self._sample_pools[pool_name] = {}

        rng = random.Random(self.random_state)

        for cid in self._class_pools.get('base', []):
            indices = self.class_to_indices[cid][:]
            rng.shuffle(indices)

            total = len(indices)
            curr  = 0
            items = list(sample_cfg.items())

            for i, (pool_name, ratio) in enumerate(items):
                end = total if i == len(items) - 1 else curr + round(total * ratio)
                self._sample_pools[pool_name][cid] = indices[curr:end]
                curr = end

        self._log(
            "Base sample pools: " +
            ", ".join(
                f"{k}={sum(len(v) for v in d.values())} samples"
                for k, d in self._sample_pools.items()
            )
        )

    # ------------------------------------------------------------------
    # Public API — for SmartDataLoaderFactory
    # ------------------------------------------------------------------

    def get_indices(self, pool_name):
        """
        Returns flat list of global indices for a pool.
        Used by TransformedPoolDataset as its index list.

        Args:
            pool_name : One of pool_names()
        Returns:
            list of int (global indices)
        """
        c2i = self.get_class_to_indices(pool_name)
        return [idx for indices in c2i.values() for idx in indices]

    def get_class_to_indices(self, pool_name):
        """
        Returns {class_id: [global_indices]} for a pool.
        Used by EpisodicBatchSampler for episodic sampling.

        Base sub-pools (pretrain/train/val_seen/test) → carved sample indices.
        Val_unseen / novel → full class_to_indices for those class sets.

        Args:
            pool_name : One of pool_names()
        Returns:
            dict {class_id: [global_indices]}
        """
        # Base sample-level sub-pools
        if pool_name in self._sample_pools:
            return self._sample_pools[pool_name]

        # Class-level pools (val_unseen, novel)
        if pool_name in self._class_pools and pool_name != 'base':
            return {
                cid: self.class_to_indices[cid]
                for cid in self._class_pools[pool_name]
            }

        raise ValueError(
            f"Pool '{pool_name}' not found. "
            f"Available pools: {self.pool_names()}"
        )

    def get_fetch_fn(self):
        """
        Returns the OffsetFetcher instance for this splitter.
        Used by SmartDataLoaderFactory to build TransformedPoolDataset.

        Returns:
            OffsetFetcher callable — fetcher(global_idx) → raw sample
        """
        if self._fetcher is None:
            raise RuntimeError(
                "OffsetFetcher not built. Call apply() before get_fetch_fn()."
            )
        return self._fetcher

    def pool_names(self):
        """
        Returns list of all valid pool names.
        Base sub-pools first, then class-level pools.
        """
        names = list(self._sample_pools.keys())           # pretrain, train, val_seen, test
        names += [k for k in self._class_pools if k != 'base']  # val_unseen, novel
        return names

    # ------------------------------------------------------------------
    # Debug access
    # ------------------------------------------------------------------

    def __getitem__(self, pool_name):
        """splitter['train'] → pool info dict for debug."""
        return self._pool_info(pool_name)

    def _pool_info(self, pool_name):
        try:
            c2i = self.get_class_to_indices(pool_name)
        except ValueError:
            return {'error': f"pool '{pool_name}' not found"}
        return {
            'pool':    pool_name,
            'classes': len(c2i),
            'samples': sum(len(v) for v in c2i.values()),
        }

    def to_dict(self):
        """Returns debug dict for all pools."""
        return {name: self._pool_info(name) for name in self.pool_names()}

    def __str__(self):
        lines = [f"FewShotClassSplitter — {len(self.pool_names())} pools:"]
        for name in self.pool_names():
            info = self._pool_info(name)
            lines.append(
                f"  {name:12s} → classes: {info.get('classes', '?'):3d}, "
                f"samples: {info.get('samples', '?'):6d}"
            )
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    # ------------------------------------------------------------------
    # Internal logging
    # ------------------------------------------------------------------

    def _log(self, message):
        if self.logger:
            self.logger.info(f"[FewShotClassSplitter] {message}")


# ==============================================================================
# TransformedPoolDataset
# ==============================================================================

class TransformedPoolDataset(Dataset):
    """
    Unified lazy-loading dataset wrapper for both paradigms and both modes.

    Map-style path (is_stream=False):
        __getitem__(idx) → fetch_fn(global_idx) → _extract_by_backend
                        → transform → (img, label)

    Streaming path (is_stream=True):
        __iter__() → stream_source → _extract_by_backend
                  → transform → (img, label)
        __getitem__ raises TypeError for streaming.

    class_to_indices:
        None → batch mode only (StandardSplitStrategy pools)
        dict → episodic mode supported (FewShotClassSplitter pools)

    Args:
        fetch_fn         : OffsetFetcher or callable(global_idx) → raw sample.
                           None for streaming.
        indices          : Flat list of global indices. None for streaming.
        transform        : torchvision transform or None.
        dstype           : DataSource dstype string.
        is_stream        : True for streaming datasource.
        class_to_indices : {class_id: [indices]} for episodic. None for batch.
        stream_source    : Iterable for streaming path. None for map-style.
        logger           : Optional logger.
    """

    def __init__(self, fetch_fn, indices, transform, dstype, is_stream,
                 class_to_indices, stream_source=None, logger=None):
        self.fetch_fn         = fetch_fn
        self.indices          = indices
        self.transform        = transform
        self.dstype           = dstype
        self.is_stream        = is_stream
        self.class_to_indices = class_to_indices
        self.stream_source    = stream_source
        self.logger           = logger

    def __len__(self):
        if self.is_stream:
            raise TypeError(
                "Streaming dataset has no fixed length. "
                "DataLoader will iterate until stream exhausted."
            )
        return len(self.indices)

    def __getitem__(self, idx):
        if self.is_stream:
            raise TypeError(
                "Streaming dataset does not support random access. "
                "Use DataLoader with shuffle=False and num_workers=0."
            )
        global_idx = self.indices[idx]
        raw_sample = self.fetch_fn(global_idx)

        if raw_sample is None:
            raise ValueError(
                f"fetch_fn returned None for global_idx={global_idx}. "
                f"Index may be out of range or DataSource resource missing."
            )

        img, label = self._extract_by_backend(raw_sample)

        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"[TransformedPoolDataset] Transform error at "
                        f"global_idx={global_idx}: {e}"
                    )
                raise
        return img, label

    def __iter__(self):
        """Streaming path — sequential only."""
        if not self.is_stream:
            raise TypeError(
                "Map-style dataset should be iterated via DataLoader, not __iter__ directly."
            )
        for raw_sample in self.stream_source:
            img, label = self._extract_by_backend(raw_sample)
            if self.transform:
                img = self.transform(img)
            yield img, label

    def _extract_by_backend(self, sample):
        """Extracts (image, label) tuple for all supported dstypes."""
        if self.dstype == "hfdataset":
            return sample["image"], sample["label"]

        elif self.dstype == "webdataset":
            return (sample.get("jpg") or sample.get("png")), sample.get("cls")

        elif self.dstype == "pandasdf":
            return sample["image"], sample["label"]

        elif self.dstype == "pygdataset":
            # PyG Data object — return whole object and y label
            return sample, sample.y

        elif self.dstype == "sklearndataset":
            return sample[0], sample[1]

        elif self.dstype == "fsdataset":
            return sample[0], sample[1]

        else:
            # torchdataset, timmdataset, default tuple
            return sample[0], sample[1]


# ==============================================================================
# EpisodicBatchSampler
# ==============================================================================

class EpisodicBatchSampler(Sampler):
    """
    N-way K-shot episode sampler for episodic training and evaluation.

    Each __iter__ call yields flat lists of N*(K+Q) local indices:
        [class1_s1..sK+Q, class2_s1..sK+Q, ...]

    RNG reset at start of each __iter__ — deterministic for fixed seed+epoch.
    Training:   call set_epoch(epoch) before each epoch to vary episodes.
    Evaluation: do NOT call set_epoch — keeps episodes identical across runs.

    Args:
        class_to_indices : {class_id: [local_indices]} — from TransformedPoolDataset.
        n_way            : N classes per episode.
        k_shot           : K support samples per class.
        q_query          : Q query samples per class.
        iterations       : Episodes per epoch.
        random_state     : Base seed.
        logger           : Optional logger.
    """

    def __init__(self, class_to_indices, n_way, k_shot, q_query,
                 iterations, random_state=None, logger=None):
        super().__init__()
        self.class_to_indices = class_to_indices
        self.n_way            = n_way
        self.k_shot           = k_shot
        self.q_query          = q_query
        self.iterations       = iterations
        self.seed             = random_state
        self.epoch            = 0
        self.logger           = logger

        self.eligible_classes = self._validate_pool()

        if len(self.eligible_classes) < self.n_way:
            raise ValueError(
                f"Requested {self.n_way}-way but only "
                f"{len(self.eligible_classes)} classes have >= "
                f"{k_shot + q_query} samples."
            )

    def _validate_pool(self):
        """Filters classes with enough samples for K+Q per episode."""
        required = self.k_shot + self.q_query
        eligible = []
        for cid, indices in self.class_to_indices.items():
            if len(indices) >= required:
                eligible.append(cid)
            elif self.logger:
                self.logger.warning(
                    f"[EpisodicBatchSampler] Class {cid} skipped: "
                    f"{len(indices)} samples < required {required}."
                )
        return eligible

    def set_epoch(self, epoch):
        """
        Vary episode sampling across training epochs.
        Call before each training epoch.
        Do NOT call for evaluation loaders.
        """
        self.epoch = epoch

    def __iter__(self):
        """
        Yields flat list of N*(K+Q) local indices per episode.
        Fresh RNG per call — deterministic for fixed (seed + epoch).
        """
        effective_seed = (self.seed + self.epoch) if self.seed is not None else None
        rng            = random.Random(effective_seed)

        for _ in range(self.iterations):
            batch            = []
            selected_classes = rng.sample(self.eligible_classes, self.n_way)
            for cid in selected_classes:
                sampled = rng.sample(self.class_to_indices[cid],
                                     self.k_shot + self.q_query)
                batch.extend(sampled)
            yield batch

    def __len__(self):
        return self.iterations


# ==============================================================================
# TaskCollator
# ==============================================================================

class TaskCollator:
    """
    Reshapes flat episode batch into structured dict for model consumption.

    Input  : list of N*(K+Q) (img_tensor, label) tuples
    Output : {
        'support' : Tensor[N, K, C, H, W],
        'query'   : Tensor[N, Q, C, H, W],
        'target'  : LongTensor[N*Q]  — task-relative labels 0..N-1
    }

    Args:
        n_way   : N classes per episode.
        k_shot  : K support samples per class.
        q_query : Q query samples per class.
        device  : Target device for tensors. None = keep on CPU.
    """

    def __init__(self, n_way, k_shot, q_query):
        self.n_way   = n_way
        self.k_shot  = k_shot
        self.q_query = q_query

    def __call__(self, batch):
        imgs     = torch.stack([item[0] for item in batch])
        reshaped = imgs.view(self.n_way, self.k_shot + self.q_query, *imgs.shape[1:])
        support  = reshaped[:, :self.k_shot]
        query    = reshaped[:, self.k_shot:]
        target   = torch.arange(self.n_way).repeat_interleave(self.q_query)

        return {
            "support": support,
            "query":   query,
            "target":  target.long()
        }


# ==============================================================================
# SmartDataLoaderFactory
# ==============================================================================

class SmartDataLoaderFactory:
    """
    Builds DataLoaders for any pool + mode combination.
    Works with FewShotClassSplitter for both Standard and FewShot paradigms.

    Responsibilities:
        - Retrieves indices and class_to_indices from splitter
        - Creates TransformedPoolDataset (lazy — no data loaded at creation)
        - Creates DataLoader in batch or episodic mode
        - Handles local index remapping for episodic sampler
        - Selects correct transform per pool
        - Handles streaming batch path

    Modes:
        'batch'    : regular DataLoader — yields (img, label) batches
                     works for all pools, streaming + map-style
        'episodic' : EpisodicBatchSampler + TaskCollator
                     requires class_to_indices (fewshot pools only)
                     does not support streaming

    Transform selection (override via transform_key param):
        'pretrain', 'train' pools → 'train' transform
        all other pools           → 'eval'  transform

    Usage:
        factory = SmartDataLoaderFactory(splitter, transformers, device=device)

        # Both paradigms — pretrain
        pretrain_loader = factory.get_loader('pretrain', mode='batch', batch_size=64)

        # Standard paradigm
        train_loader    = factory.get_loader('train',    mode='batch', batch_size=64)
        val_loader      = factory.get_loader('val_seen', mode='batch', batch_size=64)
        test_loader     = factory.get_loader('test',     mode='batch', batch_size=64)

        # FewShot paradigm
        meta_loader     = factory.get_loader('train',      mode='episodic', n=5, k=5, q=15, iterations=600)
        metaval_loader  = factory.get_loader('val_unseen', mode='episodic', n=5, k=5, q=15, iterations=200)
        seen_loader     = factory.get_loader('val_seen',   mode='episodic', n=5, k=5, q=15, iterations=200)
        novel_loader    = factory.get_loader('novel',      mode='episodic', n=5, k=5, q=15, iterations=600)

        # Evaluation — both paradigms
        test_loader     = factory.get_loader('test',       mode='episodic', n=5, k=5, q=15, iterations=600)
        novel_loader    = factory.get_loader('novel',      mode='episodic', n=5, k=5, q=15, iterations=600)
    """

    # Pools that use training transforms
    TRAIN_TRANSFORM_POOLS = {'pretrain', 'train'}

    def __init__(self, splitter, transformers, device=None,
                 random_state=None, logger=None):
        """
        Args:
            splitter     : FewShotClassSplitter instance after apply().
            transformers : {'train': transform, 'eval': transform}
            device       : torch.device or None.
            random_state : Seed for episodic samplers.
            logger       : Optional logger.
        """
        if splitter.ds is None:
            raise RuntimeError(
                "Splitter has no DataSource. "
                "Call splitter.apply(data_source=...) before factory init."
            )
        self.splitter     = splitter
        self.ds           = splitter.ds
        self.transformers = transformers
        self.device       = device
        self.random_state = random_state
        self.logger       = logger

    def get_loader(self, pool_name, mode='batch',
                   n=5, k=5, q=15, iterations=100,
                   batch_size=64, num_workers=2,
                   shuffle=None, transform_key=None):
        """
        Builds and returns a DataLoader for the specified pool and mode.

        Args:
            pool_name     : Pool name — must be in splitter.pool_names().
            mode          : 'batch' or 'episodic'.
            n             : N-way per episode (episodic only).
            k             : K-shot per episode (episodic only).
            q             : Q-query per episode (episodic only).
            iterations    : Episodes per epoch (episodic only).
            batch_size    : Batch size (batch mode only).
            num_workers   : DataLoader workers.
            shuffle       : Override shuffle for batch mode.
                            Default: True for train/pretrain, False otherwise.
            transform_key : Override transform selection — 'train' or 'eval'.
        Returns:
            torch.utils.data.DataLoader
        """
        # Validate pool
        if pool_name not in self.splitter.pool_names():
            raise ValueError(
                f"Pool '{pool_name}' not available. "
                f"Available: {self.splitter.pool_names()}"
            )

        # Validate mode
        if mode not in ('batch', 'episodic'):
            raise ValueError(f"mode must be 'batch' or 'episodic'. Got '{mode}'.")

        if mode == 'episodic' and self.ds.is_stream:
            raise NotImplementedError(
                "Episodic mode requires random access. "
                "Streaming datasources are not supported for episodic loading."
            )

        # Select transform
        if transform_key is None:
            transform_key = ('train' if pool_name in self.TRAIN_TRANSFORM_POOLS
                             else 'eval')
        transform = self.transformers.get(transform_key)

        # Get data from splitter — lazy, no data loaded yet
        indices          = self.splitter.get_indices(pool_name)
        class_to_indices = self.splitter.get_class_to_indices(pool_name)
        fetch_fn         = self.splitter.get_fetch_fn()

        # Build TransformedPoolDataset
        dataset = TransformedPoolDataset(
            fetch_fn         = fetch_fn,
            indices          = indices,
            transform        = transform,
            dstype           = self.ds.dstype,
            is_stream        = self.ds.is_stream,
            class_to_indices = class_to_indices,
            stream_source    = None,
            logger           = self.logger
        )

        is_accelerated = (
            self.device is not None and
            self.device.type != 'cpu' and
            not self.ds.is_stream
        )

        if mode == 'batch':
            return self._make_batch_loader(
                dataset, batch_size, num_workers,
                is_accelerated, pool_name, shuffle
            )
        else:
            return self._make_episodic_loader(
                dataset, n, k, q, iterations,
                num_workers, is_accelerated
            )

    # ------------------------------------------------------------------
    # Loader builders
    # ------------------------------------------------------------------

    def _make_batch_loader(self, dataset, batch_size, num_workers,
                           pin_memory, pool_name, shuffle_override):
        """Standard DataLoader for batch mode."""
        if dataset.is_stream:
            # Streaming — no shuffle, no pin_memory
            return DataLoader(
                dataset,
                batch_size  = batch_size,
                num_workers = num_workers,
                persistent_workers = num_workers > 0,
                prefetch_factor    = 2 if num_workers > 0 else None
            )

        should_shuffle = (
            pool_name in self.TRAIN_TRANSFORM_POOLS
            if shuffle_override is None else shuffle_override
        )

        return DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = should_shuffle,
            num_workers = num_workers,
            pin_memory  = pin_memory,
            persistent_workers = num_workers > 0,
            prefetch_factor    = 2 if num_workers > 0 else None,
            drop_last   = False
        )

    def _make_episodic_loader(self, dataset, n, k, q,
                              iterations, num_workers, pin_memory):
        """
        Episodic DataLoader using EpisodicBatchSampler + TaskCollator.

        Index remapping:
            EpisodicBatchSampler works in local index space (0..len(dataset)-1).
            TransformedPoolDataset maps local → global via self.indices[local_idx].
            So sampler yields local indices → dataset fetches correct global sample.
        """
        if dataset.class_to_indices is None:
            raise ValueError(
                "Episodic mode requires class_to_indices. "
                "Ensure pool comes from FewShotClassSplitter."
            )

        # Remap global indices → local positions in dataset.indices
        global_to_local = {
            g_idx: l_idx
            for l_idx, g_idx in enumerate(dataset.indices)
        }

        local_class_to_indices = {}
        for cid, global_idxs in dataset.class_to_indices.items():
            local_idxs = [
                global_to_local[gi]
                for gi in global_idxs
                if gi in global_to_local
            ]
            if local_idxs:
                local_class_to_indices[cid] = local_idxs

        sampler = EpisodicBatchSampler(
            class_to_indices = local_class_to_indices,
            n_way            = n,
            k_shot           = k,
            q_query          = q,
            iterations       = iterations,
            random_state     = self.random_state,
            logger           = self.logger
        )

        collator = TaskCollator(
            n_way   = n,
            k_shot  = k,
            q_query = q
        )

        return DataLoader(
            dataset,
            batch_sampler = sampler,
            collate_fn    = collator,
            num_workers   = num_workers,
            pin_memory    = pin_memory,
            persistent_workers = num_workers > 0,
            prefetch_factor    = 2 if num_workers > 0 else None
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def valid_pools(self):
        """Returns valid pool names for this factory's splitter."""
        return self.splitter.pool_names()

    def __repr__(self):
        return (f"SmartDataLoaderFactory("
                f"paradigm=fewshot, "
                f"pools={self.splitter.pool_names()}, "
                f"device={self.device})")
