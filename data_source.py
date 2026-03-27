import os
import shutil
import importlib
import hashlib
from collections import defaultdict
from abc import ABC, abstractmethod

# ==========================================
# class DataSourceRegister (Bookkeeping for DataSource)
# ==========================================

class DataSourceRegister:
    _source_types = {}
    _dataset_types = {}
    _dataset_label_mappers = {}

    @classmethod
    def register_source_type(cls, source_type_name, handler=None):
        """
        Works as both a manual call and a decorator.
        Manual: DataSourceRegister.register_source_type("disk", DiskClass)
        Decorator: @DataSourceRegister.register_source_type("disk")
        """
        def wrapper(handler_class):
            cls._source_types[source_type_name] = handler_class
            return handler_class

        # If handler is passed, it's a manual registration
        if handler is not None:
            return wrapper(handler)
        
        # Otherwise, return the wrapper to be used as a decorator
        return wrapper

    @classmethod
    def register_dataset_type(cls, dstype_name, handler=None):
        """
        Works as both a manual call and a decorator for inspectors.
        """
        def wrapper(handler_func):
            cls._dataset_types[dstype_name] = handler_func
            return handler_func

        if handler is not None:
            return wrapper(handler)
        
        return wrapper

    @classmethod
    def register_dataset_label_mapper(cls, dstype_name, handler=None):
        """
        Register a label mapper for a specific dataset type.
        """
        def wrapper(handler_func):
            cls._dataset_label_mappers[dstype_name] = handler_func
            return handler_func

        if handler is not None:
            return wrapper(handler)
        
        return wrapper

    @classmethod
    def get_source_handler(cls, source_type_name):
        if source_type_name not in cls._source_types:
            raise ValueError(f"Unsupported source_type: {source_type_name}. Valid options are: {list(cls._source_types.keys())}")
        return cls._source_types[source_type_name]
    
    @classmethod
    def get_dataset_handler(cls, dstype_name):
        if dstype_name not in cls._dataset_types:
            raise ValueError(f"Unsupported dataset_type: {dstype_name}. Valid options are: {list(cls._dataset_types.keys())}")
        return cls._dataset_types[dstype_name]

    @classmethod
    def get_dataset_label_mapper(cls, dstype_name):
        if dstype_name not in cls._dataset_label_mappers:
            raise ValueError(
                f"No label mapper registered for dstype '{dstype_name}'. "
                f"Registered mappers: {list(cls._dataset_label_mappers.keys())}"
            )
        return cls._dataset_label_mappers[dstype_name]

    @classmethod
    def list_source_types(cls):
        return list(cls._source_types.keys())

    @classmethod
    def list_dataset_types(cls):
        return list(cls._dataset_types.keys())

    @classmethod
    def list_dataset_label_mappers(cls):
        return list(cls._dataset_label_mappers.keys())

    @classmethod
    def clear(cls):
        cls._source_types.clear()
        cls._dataset_types.clear()
        cls._dataset_label_mappers.clear()


# ==========================================
# class DatasetLabelMapper (dataset-specific metadata extraction logic)
# ==========================================

class DatasetLabelMapper:
    """
    Metadata Registry for translating any physical data format into 
    a logical {class_id: [indices]} mapping.
    """

    @staticmethod
    def _get_id(label, class_to_idx):
        """Helper to safely map label string/int to our master ID."""
        l_str = str(label)
        return class_to_idx.get(l_str)

    @staticmethod
    def default_map(dataset, class_to_idx):
        """Fallback for any Pythonic dataset (Slow Path)."""
        mapping = defaultdict(list)
        for i in range(len(dataset)):
            _, label = dataset[i]
            c_idx = DatasetLabelMapper._get_id(label, class_to_idx)
            if c_idx is not None:
                mapping[c_idx].append(i)
        return mapping

    # 1. HuggingFace Dataset
    @staticmethod
    @DataSourceRegister.register_dataset_label_mapper("hfdataset")
    def map_hf(dataset, class_to_idx):
        mapping = defaultdict(list)
        
        # 1. Identify label column and feature
        target_keys = [k for k, v in dataset.features.items() if hasattr(v, 'names')]
        if not target_keys: return mapping
        
        label_col = target_keys[0]
        local_feature = dataset.features[label_col]

        # 2. Local Helper: The "Bridge" logic
        # Takes (index, local_label_value), updates mapping
        def _process_entry(i, local_idx):
            try:
                # Local Int -> Name -> Global Int
                class_name = local_feature.int2str(local_idx)
                global_idx = class_to_idx.get(class_name)
                if global_idx is not None:
                    mapping[global_idx].append(i)
            except (ValueError, IndexError):
                pass

        # 3. Choose path based on dataset capability
        if hasattr(dataset, "__getitem__") and not getattr(dataset, "is_streaming", False):
            # FAST PATH: Columnar access (Map-style)
            # Pulling one column is significantly faster than row-by-row dict conversion
            labels = dataset[label_col]
            for i, val in enumerate(labels):
                _process_entry(i, val)
        else:
            # SLOW PATH: Iterator access (Streaming/IterableDataset)
            for i, example in enumerate(dataset):
                _process_entry(i, example[label_col])
                
        return mapping

    # 2. WebDataset
    @staticmethod
    @DataSourceRegister.register_dataset_label_mapper("webdataset")
    def map_wds(dataset, class_to_idx):
        """Sequential scan for WebDataset tar-streams."""
        mapping = defaultdict(list)
        for i, sample in enumerate(dataset):
            label = sample.get('cls', sample.get('label'))
            c_idx = DatasetLabelMapper._get_id(label, class_to_idx)
            if c_idx is not None:
                mapping[c_idx].append(i)
        return mapping

    # 3. Torch / Timm
    @staticmethod
    @DataSourceRegister.register_dataset_label_mapper("torchdataset")
    @DataSourceRegister.register_dataset_label_mapper("timmdataset")
    def map_torch(dataset, class_to_idx):
        """Optimized for torchvision.datasets and timm datasets."""
        mapping = defaultdict(list)
        targets = getattr(dataset, 'targets', getattr(dataset, 'labels', None))
        if targets is not None:
            for i, l in enumerate(targets):
                c_idx = DatasetLabelMapper._get_id(l, class_to_idx)
                if c_idx is not None: mapping[c_idx].append(i)
        else:
            return DatasetLabelMapper.default_map(dataset, class_to_idx)
        return mapping

    # 4. PyTorch Geometric (PyG)
    @staticmethod
    @DataSourceRegister.register_dataset_label_mapper("pygdataset")
    def map_pyg(dataset, class_to_idx):
        """Optimized for PyG graph y-tensors."""
        mapping = defaultdict(list)
        y = None
        if hasattr(dataset, 'y'): y = dataset.y
        elif hasattr(dataset, 'data') and hasattr(dataset.data, 'y'): y = dataset.data.y
        
        if y is not None:
            y_list = y.view(-1).tolist()
            for i, l in enumerate(y_list):
                c_idx = DatasetLabelMapper._get_id(l, class_to_idx)
                if c_idx is not None: mapping[c_idx].append(i)
        return mapping

    # 5. Scikit-Learn
    @staticmethod
    @DataSourceRegister.register_dataset_label_mapper("sklearndataset")
    def map_sklearn(dataset, class_to_idx):
        """Handles sklearn.utils.Bunch (like load_iris())."""
        mapping = defaultdict(list)
        labels = getattr(dataset, 'target', None)
        if labels is not None:
            for i, l in enumerate(labels):
                c_idx = DatasetLabelMapper._get_id(l, class_to_idx)
                if c_idx is not None: mapping[c_idx].append(i)
        return mapping

    # 6. fsspec (Remote/Abstract Filesystems)
    @staticmethod
    @DataSourceRegister.register_dataset_label_mapper("fsdataset")
    def map_fsspec(dataset, class_to_idx):
        """Handles remote datasets via fsspec (S3/GCS/Azure)."""
        mapping = defaultdict(list)
        samples = getattr(dataset, 'samples', getattr(dataset, 'info', []))
        for i, entry in enumerate(samples):
            label = entry[1] if isinstance(entry, (tuple, list)) else entry.get('label')
            c_idx = DatasetLabelMapper._get_id(label, class_to_idx)
            if c_idx is not None: mapping[c_idx].append(i)
        return mapping

    # 7. Pandas
    @staticmethod
    @DataSourceRegister.register_dataset_label_mapper("pandasdf")
    def map_pandas(dataset, class_to_idx):
        """Fast vectorized path for DataFrames."""
        mapping = defaultdict(list)
        target_col = next((c for c in dataset.columns if c.lower() in 
                         ['label', 'target', 'class', 'category']), None)
        if target_col:
            groups = dataset.groupby(target_col).indices
            for label_val, idxs in groups.items():
                c_idx = DatasetLabelMapper._get_id(label_val, class_to_idx)
                if c_idx is not None: mapping[c_idx].extend(idxs.tolist())
        return mapping


# ==========================================
# class DatasetTypeTraits (dataset-specific metadata extraction logic)
# ==========================================

class DatasetTypeTraits:
    """
    Unified container for logical metadata extraction. 
    Each method corresponds to a 'dstype' and implements 
    specific discovery strategies without peeking at the data.
    """
    
    STANDARD_NAMES = {'label', 'labels', 'class', 'classes', 'target', 'targets', 'category', 'categories'}

    @staticmethod
    @DataSourceRegister.register_dataset_type("hfdataset")
    def inspect_hfds(resources, identifier, manual_classes, **kwargs):
        if manual_classes: return sorted(list(set(manual_classes)))
            
        union_classes = set()
        for res in [r for r in resources.values() if r is not None]:
            features = getattr(res, 'features', {})
            target_key = next((k for k, v in features.items() 
                               if k.lower() in DatasetTypeTraits.STANDARD_NAMES 
                               and hasattr(v, 'names')), None)
            if target_key:
                union_classes.update(features[target_key].names)
            else:
                for feat in features.values():
                    if hasattr(feat, 'names'):
                        union_classes.update(feat.names)
                        break
        return sorted(list(union_classes))

    @staticmethod
    @DataSourceRegister.register_dataset_type("webdataset")
    def inspect_wds(resources, identifier, manual_classes, **kwargs):
        if manual_classes: return sorted(list(set(manual_classes)))
            
        try:
            import fsspec, json
            base_path = identifier.split('{')[0].rsplit('/', 1)[0]
            fs, _ = fsspec.core.url_to_fs(base_path)
            for meta_file in ["stats.json", "meta.json", "label_map.json"]:
                full_path = f"{base_path}/{meta_file}"
                if fs.exists(full_path):
                    with fs.open(full_path, 'r') as f:
                        meta_data = json.load(f)
                        for key in ['classes', 'labels', 'label_names', 'names']:
                            if key in meta_data:
                                return sorted(list(set(meta_data[key])))
        except Exception:
            pass 
        return []

    @staticmethod
    @DataSourceRegister.register_dataset_type("torchdataset")
    @DataSourceRegister.register_dataset_type("timmdataset")
    def inspect_torchds(resources, identifier, manual_classes, **kwargs):
        if manual_classes: return sorted(list(set(manual_classes)))
            
        union_classes = set()
        for res in [r for r in resources.values() if r is not None]:
            classes = getattr(res, 'classes', [])
            if not classes and hasattr(res, 'class_to_idx'):
                classes = list(res.class_to_idx.keys())
            union_classes.update(classes)
        return sorted(list(union_classes))

    @staticmethod
    @DataSourceRegister.register_dataset_type("pygdataset")
    def inspect_pygds(resources, identifier, manual_classes, **kwargs):
        """PyG metadata property for graph classification/node tasks."""
        if manual_classes: return sorted(list(set(manual_classes)))
            
        union_classes = set()
        for res in [r for r in resources.values() if r is not None]:
            n = getattr(res, 'num_classes', 0)
            if n:
                # PyG usually provides a count; we generate string labels for the union
                union_classes.update([str(i) for i in range(n)])
        return sorted(list(union_classes))

    @staticmethod
    @DataSourceRegister.register_dataset_type("sklearndataset")
    def inspect_sklearnds(resources, identifier, manual_classes, **kwargs):
        """Handles Scikit-Learn Bunch objects or fitted estimators."""
        if manual_classes: return sorted(list(set(manual_classes)))
            
        union_classes = set()
        for res in [r for r in resources.values() if r is not None]:
            # Priority: target_names (Bunch), Fallback: classes_ (Estimator)
            names = getattr(res, 'target_names', getattr(res, 'classes_', []))
            union_classes.update(names)
        return sorted(list(union_classes))

    @staticmethod
    @DataSourceRegister.register_dataset_type("fsdataset")
    def inspect_fsds(resources, identifier, manual_classes, **kwargs):
        if manual_classes: return sorted(list(set(manual_classes)))
            
        try:
            import fsspec
            fs, path = fsspec.core.url_to_fs(identifier)
            contents = fs.ls(path, detail=True)
            return sorted([os.path.basename(c['name'].rstrip('/')) 
                          for c in contents if c['type'] == 'directory'])
        except Exception:
            return []

    @staticmethod
    @DataSourceRegister.register_dataset_type("pandasdf")
    def inspect_pandasdf(resources, identifier, manual_classes, **kwargs):
        if manual_classes: return sorted(list(set(manual_classes)))
        
        union_classes = set()
        for res in [r for r in resources.values() if r is not None]:
            target_col = next((c for c in res.columns 
                               if c.lower() in DatasetTypeTraits.STANDARD_NAMES), None)
            if target_col:
                # Assuming the column exists, we extract unique values
                # or treat the column name as the identified class structure
                union_classes.update(res[target_col].unique().tolist())

        return sorted(list(union_classes))


# ==========================================
# class DataSource (PHYSICAL LAYER)
# ==========================================

class DataSource(ABC):
    """
Physical Layer: Responsible for locating and accessing raw data.

    This base class standardizes the discovery of data paths and the retrieval 
    of raw assets into a unified resource inventory. While the discovery logic 
    is execution-agnostic, the output is optimized for the PyTorch ecosystem, 
    ensuring 'resources' can be directly wrapped by DataLoaders or Samplers.

    Design Principles:
    ------------------
    1. PyTorch-Native Integration: Designed to bridge physical storage (S3, 
       Local, Hub) with Torch-specific data structures (Datasets, Iterables).
    2. Standardized Discovery: Normalizes how disparate assets (folders, 
       archives, or remote streams) are identified and indexed.
    3. Resource Mapping: Automatically maps physical storage structures into 
       a consistent 'resources' dictionary containing 'train', 'test', and 'val' 
       objects, ready for logical splitting and transformation.

    Attributes:
        identifier (str)    : The physical path, URL, or repository ID of the data.
        sub_type (str)      : The data modality (image, audio, video) or 
                                library-specific dataset name (e.g., 'CIFAR10').
        resources (dict)    : A standardized inventory of discovered data pools:
                                {'train': <Dataset>, 'test': <Dataset>, 'val': <Dataset>}
        logger (callable)   : Optional logging function for verbose output.
        engine_kwargs       : Additional backend-specific options (e.g., md5 for URL verification, library-specific args).
        cache_root (str)    : Local directory for persistent storage and extraction.
        force_download(bool): If True, bypasses local cache to re-fetch assets.
    
    Usage Pattern:
        driver = DataSource.load(source_type, identifier, sub_type, **kwargs)

    Args:
        - source_type       : The origin type (e.g., 'disk', 'url', 'hf', 'stream', 'library', 'cloud')
        - identifier        : The primary location (e.g., directory path, URL, or dataset name).
        - sub_type          : The specific type of data within the source (e.g., 'image', 'text', 'audio') 
                                or library-specific dataset name (e.g., 'CIFAR10' for torchvision).
        - logger            : A logging function (e.g., print) for verbose output.
        - cache_root        : Directory for caching downloads or extracted data.
        - force_download    : Whether to re-download or re-extract even if cached data exists.
        - clean_on_exit     : Whether to clear cached data after use (if cache_root is not specified).
        - num_proc          : Number of processes for parallel operations (e.g., extraction, indexing).
        **kwargs            : Other source type specific options for fine-tuning behavior
                                (e.g., md5 for URL verification, library-specific args like load_dataset parameters).

    Usage Examples:
        ds = DataSource.load("disk", identifier="./my_images.zip", sub_type="image")
        ds = DataSource.load('url', identifier='https://data.com/dataset.zip', cache_root='./cache', force_download=True)
        ds = DataSource.load("hf", identifier="glue", sub_type="mrpc")
        ds = DataSource.load("stream", identifier="laion/laion2B-en", buffer_size=10000)
                # Will keep 10k samples in RAM to shuffle them on-the-fly
        ds = DataSource.load("library", identifier="torchvision", sub_type="CIFAR10", cache_root="./data")
    """

    def __init__(self, *args, **kwargs):
        """Standard constructor is disabled to enforce use of .load()"""
        raise RuntimeError(
            f"Direct initialization of {self.__class__.__name__} is disabled. "
            f"Please use DataSource.load(source_type, identifier, sub_type=<None, sub_type>, logger=None, **kwargs) instead."
        )

    def _init_internal(self, identifier, sub_type=None, logger=None, **kwargs):
        """
        Args:
            identifier (str): Path, URL, or Module name.
            logger (callable, optional): A function like 'print' for verbose output.
            **kwargs: Source-specific options (e.g., download=True).
        """
        self.identifier = identifier
        self.sub_type = sub_type
        self.logger = logger

        # Orchestration Parameters
        self.cache_root = kwargs.pop('cache_root', os.path.expanduser('~/.cache/ai_data'))
        # Also pop 'cache_dir' in case the user accidentally passed it, 
        # to prevent duplicate argument errors later.
        kwargs.pop('cache_dir', None)
    
        self.force_download = kwargs.pop('force_download', False)
        self.clean_on_exit = kwargs.pop('clean_on_exit', False)
        self.num_proc = kwargs.pop('num_proc', None)
        self.engine_kwargs = kwargs  # Remaining args passed to specific drivers

        # Unified Resource Inventory
        self.resources = {'train': None, 'val': None, 'test': None}
        self.dstype = None          # type of datasource object like hf dataset, torch dataset, etc. This is set by the specific driver after fetch()
        self.classes = []           # Sorted list of class names
        self.class_to_idx = {}      # Reverse lookup: {"classname": 0}

        # For drivers that create a local cache (e.g., CloudDataSource)
        # This will be populated by the specific driver during fetch()
        self._local_cache_path = self._get_local_cache_path()

    def __enter__(self):
        """Allows usage: with DataSource.load(...) as ds:"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Guarantees cleanup even if an error occurs inside the 'with' block."""
        if self.clean_on_exit:
            self.clean()

    def __del__(self):
        """
        Automatic cleanup on object destruction, governed by clean_on_exit.
        """
        try:
            if getattr(self, 'clean_on_exit', False):
                self.clean()
        except Exception:
            pass  # Silently ignore during interpreter shutdown

    def clean(self):
        """
        Explicitly triggers the removal of dataset-specific cached assets.
        Useful for manual memory/disk management during long-running sessions.
        """
        if self._local_cache_path and os.path.exists(self._local_cache_path):
            try:
                self._log(f"Cleaning cached assets: {self._local_cache_path}")
                if os.path.isdir(self._local_cache_path):
                    shutil.rmtree(self._local_cache_path)
                else:
                    os.remove(self._local_cache_path)
                
                # Reset path after successful deletion
                self._local_cache_path = None
            except Exception as e:
                self._log(f"Cleanup failed for {self._local_cache_path}: {e}")

    def __getitem__(self, split):
        """Returns the raw list for a split: src['train']"""
        return self.resources.get(split)

    def __len__(self):
        """Returns total count of discovered items."""
        return self.get_len()
    
    def get_len(self, split=None):
        """
        Returns length of a specific split, or total length if split is None.
        """
        if split is not None:
            pool = self.resources.get(split)
            return len(pool) if pool is not None else 0
        
        return sum(len(v) for v in self.resources.values() if v is not None)

    def get_dataset_label_mapper(self):
        """
        Returns the specific mapping function based on the current dstype.
        """
        # We look up the function in the Registry using our own dstype
        try:
            return DataSourceRegister.get_dataset_label_mapper(self.dstype)
        except ValueError as e:
            raise RuntimeError(f"DataSource has no label mapper for dstype='{self.dstype}'. "
                               f"Was fetch() called? Original error: {e}")

    @property
    @abstractmethod
    def is_stream(self) -> bool:
        """Determines if the DataSplitter should use random access or streaming filters."""
        pass

    @abstractmethod
    def fetch(self):
        """
        Scans the source to populate self.resources. 
        Must be implemented by specialized subclasses.
        """
        pass

    @staticmethod
    def load(source_type, identifier, sub_type=None, logger=None, **kwargs):
        """
        Factory method to initialize a specific DataSource.

        Args:
            source_type (str): ['disk', 'url', 'hf', 'stream', 'library', 'cloud']
                - 'disk': Local folder or file structure.
                - 'url': Remote URL pointing to a file or archive (zip, tar.gz).
                - 'hf': HuggingFace Datasets, accessed via the datasets library, using dataset names (e.g., 'cifar10', 'imagenet-1k').
                - 'stream': A streaming API endpoint that provides image data on-the-fly.
                - 'library': Datasets from external modules/libraries (torchvision, torchaudio, etc).
                - 'cloud': Data stored in cloud services (e.g., AWS S3, Google Cloud Storage) accessed via their respective SDKs.
        
        Returns:
            DataSource: An initialized instance of a specialized subclass.
        
        Rest of the args & usage are explained in the class docstring.
        """
        # Fetch Physical Driver from DataSourceRegister
        cls_source_type = DataSourceRegister.get_source_handler(source_type)

        # 2. Bypass __init__ by using __new__ and calling the internal init
        # We create the object without triggering the RuntimeError in __init__
        instance = cls_source_type.__new__(cls_source_type)
        instance._init_internal(identifier, sub_type=sub_type, logger=logger, **kwargs)
    
        # Run physical fetch (which sets self.dstype)
        instance.fetch() 
        
        # 3. Get the logical inspector (Dataset Type)
        if instance.dstype and any(instance.resources.values()):
            try:
                # Looks up the static method in DatasetTypeTraits via DataSourceRegister
                inspector = DataSourceRegister.get_dataset_handler(instance.dstype)
            
                # The 'Standard Trio' call:
                # We pass resources, the original path, and any user-provided class list
                instance.classes = inspector(
                    resources=instance.resources,
                    identifier=instance.identifier,
                    manual_classes=instance.engine_kwargs.get("classes", [])
                )
            
                # 3. FINALIZATION: Populate Metadata properties
                if instance.classes:
                    instance.class_to_idx = {name: i for i, name in enumerate(instance.classes)}

            except ValueError as e:
                # Handle cases where dstype was set but no inspector was registered
                print(f"Warning: Metadata extraction skipped. {e}")

        return instance

    
    def _get_common_params(self):
        """Helper to inject global params into backend calls."""
        params = {
            "cache_dir": self._local_cache_path,
            "download_mode": "force_redownload" if self.force_download else "reuse_dataset_if_exists",
            "trust_remote_code": self.engine_kwargs.get("trust_remote_code", False)
        }
        if self.num_proc is not None:
            params["num_proc"] = self.num_proc

        return params

    def _get_local_cache_path(self):
        """
        Generates a deterministic filesystem path. 
        Same inputs always yield the same path, allowing for cache reuse.
        """
        # Create a safe slug for the folder name
        ident_slug = "".join([c if c.isalnum() else "_" for c in str(self.identifier)])
        ident_slug = ident_slug.strip("_")[-20:]
        
        # Hash identifier + sub_type to ensure uniqueness
        seed = f"{self.identifier}_{self.sub_type}"
        unique_hash = hashlib.md5(seed.encode()).hexdigest()[:8]
        
        sub_slug = f"_{self.sub_type}" if self.sub_type else ""
        return os.path.join(self.cache_root, f"{ident_slug}{sub_slug}_{unique_hash}")

    def _log(self, message):
        """Internal helper for optional logging."""
        if self.logger:
            self.logger.info(f"[DataSource] {message}")


@DataSourceRegister.register_source_type("disk")
class DiskDataSource(DataSource):
    @property
    def is_stream(self): return False

    def fetch(self):
        if self.force_download:
            self.clean()
            self._local_cache_path = self._get_local_cache_path()  # regenerate after clean

        os.makedirs(self._local_cache_path, exist_ok=True)

        # 1. Physical Existence Check (Fail fast)
        if not os.path.exists(self.identifier):
            raise FileNotFoundError(f"Path not found: {self.identifier}")

        try:
            self._fetch()
            self._finalize_report()
        except Exception as e:
            # Catch backend specific errors (Corrupt zip, unsupported CSV format, etc.)
            raise RuntimeError(f"Engine failed to index {self.identifier}. Error: {str(e)}")

    def _fetch(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Library 'datasets' not found. Please install using 'pip install datasets'.")

        self._log(f"Searching local directory: {self.identifier}")

        builder_map = {
            "image": "imagefolder",
            "audio": "audiofolder",
            "video": "videofolder"
        }

        # 1. Determine Builder & Entry Point
        is_dir = os.path.isdir(self.identifier)
        if is_dir:
            builder = builder_map.get(self.sub_type, "imagefolder")
            data_kwargs = {"data_dir": self.identifier}
        else:
            ext = os.path.splitext(self.identifier)[1][1:].lower()
            
            if self.sub_type == "tabular":
                # ext is 'csv', 'json', 'parquet' etc.
                builder = ext 
            else:
                # It is a media file OR an archive (zip/tar) containing media
                # The 'folder' builders in HF natively handle local archives
                builder = builder_map.get(self.sub_type, "imagefolder")
            
            data_kwargs = {"data_files": self.identifier}

        # 2. Execute Backend Engine
        # This handles nested splits automatically (train/test/val subfolders)
        common = self._get_common_params()
        # Strictly filter out keys that we've already handled in 'common'
        # or that might clash with Hugging Face's API
        forbidden_keys = list(common.keys()) + ['cache_root', 'cache_dir']
        clean_kwargs = {k: v for k, v in self.engine_kwargs.items() if k not in forbidden_keys}

        ds = load_dataset(
            builder,
            **data_kwargs,
            **common,
            **clean_kwargs
        )

        # 3. Map to Resources Dictionary
        if isinstance(ds, dict):
            # Map detected splits to our standardized keys
            self.resources['train'] = ds.get('train')
            self.resources['test'] = ds.get('test')
            # Normalize 'validation' to 'val'
            self.resources['val'] = ds.get('validation') or ds.get('val')
            
            # Case: User pointed directly to a 'test' folder
            # HF might return {'train': data} even if the folder name was 'test'
            # if only one split is found, we ensure it's at least in 'train'
            if not self.resources['train'] and len(ds) == 1:
                self.resources['train'] = list(ds.values())[0]
        else:
            # Case: Single file or flat directory -> Everything is train
            self.resources['train'] = ds
        self.dstype = "hfdataset"  #type(self.resources['train']).__name__

    def _finalize_report(self):
        msg = " | ".join([f"{k}: {len(v) if v else 'Streaming'}" 
                  for k, v in self.resources.items()])

        self._log(f"Fetch complete. Resources indexed -> {msg}")


@DataSourceRegister.register_source_type("url")
class UrlDataSource(DataSource):
    """
    Unified handler for remote URLs (ZIP, TAR, CSV, pickle, etc.) using HF backend.

    Flow:
        - Tabular / single flat file  → load_dataset(builder, data_files=url)  [HF native]
        - Archive (zip/tar)           → DownloadManager.download+extract → inspect contents:
              image/audio/video files → load_dataset(builder, data_dir=local_dir)
              pickle files            → manual load → HF DatasetDict (lazy Image feature)
              other                   → imagefolder best-effort fallback
    """

    ARCHIVE_EXTENSIONS = ('.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2')
    TABULAR_EXTENSIONS  = ('csv', 'json', 'parquet', 'jsonl')

    IMAGE_EXTS  = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    AUDIO_EXTS  = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    VIDEO_EXTS  = {'.mp4', '.avi', '.mov', '.mkv'}
    PICKLE_EXTS = {'.pickle', '.pkl'}
    TABULAR_EXTS = {'.csv', '.json', '.jsonl', '.parquet'}

    # Maps filename keywords → canonical split names (order matters — more specific first)
    PICKLE_SPLIT_KEYWORDS = [
        ('train_phase_train', 'train'),
        ('train_phase_test',  None),
        ('train_phase_val',   None),
        ('_train',            'train'),
        ('_validation',       'val'),
        ('_val',              'val'),
        ('_test',             'test'),
    ]

    @property
    def is_stream(self): return False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fetch(self):
        if self.force_download:
            self.clean()
            self._local_cache_path = self._get_local_cache_path()  # regenerate after clean

        os.makedirs(self._local_cache_path, exist_ok=True)

        if not self.identifier.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL: {self.identifier}")

        try:
            self._fetch()
            self._finalize_report()
        except Exception as e:
            raise RuntimeError(f"Engine failed to fetch/index URL {self.identifier}. Error: {str(e)}") from e

    # ------------------------------------------------------------------
    # Core fetch logic
    # ------------------------------------------------------------------

    def _fetch(self):
        try:
            from datasets import load_dataset
            from datasets.download import DownloadManager, DownloadConfig
        except ImportError:
            raise ImportError("Library 'datasets' not found. Please install using 'pip install datasets'.")

        self._log(f"Fetching remote resource: {self.identifier}")

        common = self._get_common_params()
        forbidden_keys = list(common.keys()) + ['cache_root', 'cache_dir']
        clean_kwargs = {k: v for k, v in self.engine_kwargs.items() if k not in forbidden_keys}

        url_clean = self.identifier.split('?')[0].lower()

        # ── Case 1: Tabular flat file — HF handles URL natively ──────────────
        if self.sub_type == "tabular":
            ext     = url_clean.split('.')[-1]
            builder = ext if ext in self.TABULAR_EXTENSIONS else 'csv'
            ds      = load_dataset(builder, data_files=self.identifier, **common, **clean_kwargs)

        # ── Case 2: Archive — download+extract, then inspect contents ─────────
        elif any(url_clean.endswith(ext) for ext in self.ARCHIVE_EXTENSIONS):
            dl_manager = DownloadManager(download_config=DownloadConfig(
                cache_dir=self._local_cache_path,
                force_download=self.force_download,
            ))
            local_archive  = dl_manager.download(self.identifier)
            local_data_dir = dl_manager.extract(local_archive)
            self._log(f"Extracted to: {local_data_dir}")

            # Unwrap single top-level subdirectory (common zip pattern)
            subdirs = [
                os.path.join(local_data_dir, d)
                for d in os.listdir(local_data_dir)
                if os.path.isdir(os.path.join(local_data_dir, d))
            ]
            if len(subdirs) == 1:
                local_data_dir = subdirs[0]
                self._log(f"Unwrapped to: {local_data_dir}")

            ds = self._load_extracted(local_data_dir, common, clean_kwargs, load_dataset)

        # ── Case 3: Single non-archive file — data_files URL directly ────────
        else:
            sub_type_builder = {"image": "imagefolder", "audio": "audiofolder", "video": "videofolder"}
            builder = sub_type_builder.get(self.sub_type, "imagefolder")
            ds      = load_dataset(builder, data_files=self.identifier, **common, **clean_kwargs)

        # ── Map to standardized resources ─────────────────────────────────────
        if isinstance(ds, dict):
            self.resources['train'] = ds.get('train')
            self.resources['test']  = ds.get('test')
            self.resources['val']   = ds.get('validation') or ds.get('val')

            # Fallback: single split with non-standard name → treat as train
            if not self.resources['train'] and len(ds) == 1:
                self.resources['train'] = list(ds.values())[0]
        else:
            self.resources['train'] = ds

        self.dstype = "hfdataset"

    # ------------------------------------------------------------------
    # Extracted directory loader — inspects content, picks right strategy
    # ------------------------------------------------------------------

    def _load_extracted(self, local_data_dir, common, clean_kwargs, load_dataset):
        """
        Inspects the extracted directory's dominant file type
        and routes to the appropriate loading strategy.
        """
        from collections import Counter

        all_files = [
            (root, f)
            for root, _, files in os.walk(local_data_dir)
            for f in files
        ]

        if not all_files:
            raise RuntimeError(f"Extracted directory is empty: {local_data_dir}")

        ext_counts = Counter(os.path.splitext(f)[1].lower() for _, f in all_files)
        dominant   = ext_counts.most_common(1)[0][0]
        self._log(f"Extracted content — file types: {dict(ext_counts)}")

        # Image / Audio / Video folder structure
        if dominant in self.IMAGE_EXTS:
            return load_dataset("imagefolder", data_dir=local_data_dir, split=None, **common, **clean_kwargs)
        elif dominant in self.AUDIO_EXTS:
            return load_dataset("audiofolder", data_dir=local_data_dir, split=None, **common, **clean_kwargs)
        elif dominant in self.VIDEO_EXTS:
            return load_dataset("videofolder", data_dir=local_data_dir, split=None, **common, **clean_kwargs)

        # Tabular files
        elif dominant in self.TABULAR_EXTS:
            builder = dominant.lstrip('.')
            return load_dataset(builder, data_dir=local_data_dir, split=None, **common, **clean_kwargs)

        # Pickle files — load manually, wrap as lazy HF DatasetDict
        elif dominant in self.PICKLE_EXTS:
            return self._load_pickles(local_data_dir, all_files)

        # Unknown — best-effort imagefolder fallback
        else:
            self._log(f"Warning: Unknown file type '{dominant}', attempting imagefolder fallback.")
            return load_dataset("imagefolder", data_dir=local_data_dir, split=None, **common, **clean_kwargs)

    # ------------------------------------------------------------------
    # Pickle loader — generic {data, labels} format → lazy HF DatasetDict
    # ------------------------------------------------------------------

    def _load_pickles(self, local_data_dir, all_files):
        import pickle
        import numpy as np
        from collections import Counter
        from datasets import Dataset, DatasetDict, Features, ClassLabel
        from datasets import Image as HFImage

        split_data = {}

        pickle_files = sorted(
            os.path.join(root, f)
            for root, f in all_files
            if os.path.splitext(f)[1].lower() in self.PICKLE_EXTS
        )

        for pkl_path in pickle_files:
            fname = os.path.basename(pkl_path).lower()
            self._log(f"Reading pickle: {fname}")

            with open(pkl_path, 'rb') as pf:
                obj = pickle.load(pf, encoding='bytes')

            # Normalize bytes keys
            obj = {k.decode() if isinstance(k, bytes) else k: v
                   for k, v in obj.items()}

            images = next((obj[k] for k in ('data', 'images') if k in obj), None)
            labels = next((obj[k] for k in ('labels', 'label', 'targets') if k in obj), None)

            if images is None or labels is None:
                self._log(f"  Skipping {fname} — keys: {list(obj.keys())}")
                continue

            # Reshape flat uint8 → (N, H, W, C)
            if isinstance(images, np.ndarray) and images.ndim == 2:
                n, side = images.shape[0], int((images.shape[1] // 3) ** 0.5)
                images  = images.reshape(n, 3, side, side).transpose(0, 2, 3, 1)

            label_list = labels.tolist() if hasattr(labels, 'tolist') else list(labels)

            # ── Compute stats on RAW file before any remapping ────────
            counts    = Counter(label_list)
            n_cls     = len(counts)
            min_c     = min(counts.values())
            max_c     = max(counts.values())
            balance   = 'balanced' if min_c == max_c else f'UNBALANCED min={min_c} max={max_c}'
            img_shape = images[0].shape if hasattr(images[0], 'shape') else 'unknown'
            raw_name  = os.path.splitext(fname)[0]   # filename without extension

            # ── Log BEFORE remap ──────────────────────────────────────
            self._log(f"  [before remap] '{raw_name}'")
            self._log(f"       samples={len(label_list):,} | classes={n_cls} | "
                      f"ids=[{min(counts.keys())}..{max(counts.keys())}] | "
                      f"{balance} | img_shape={img_shape}")

            # ── Resolve split key via PICKLE_SPLIT_KEYWORDS ───────────
            split_key = next(
                (v for k, v in self.PICKLE_SPLIT_KEYWORDS if k in fname),
                'train'
            )

            if split_key is None:
                self._log(f"       '{raw_name}' → skipped")
                continue

            # ── Log AFTER remap ───────────────────────────────────────
            self._log(f"       → remap : '{raw_name}' → '{split_key}'  (ds.resources['{split_key}'])")

            # ── Accumulate into split_data ────────────────────────────
            if split_key not in split_data:
                split_data[split_key] = {'image': [], 'label': []}
            split_data[split_key]['image'].extend(images)
            split_data[split_key]['label'].extend(label_list)

        if not split_data:
            raise RuntimeError(
                f"No usable data found in pickle files under {local_data_dir}"
            )

        # ── Cross-split summary after ALL files loaded ─────────────────
        self._log(f"{'─'*55}")
        self._log(f"ds.resources summary (after remap + merge):")
        self._log(f"  {'split':<12} {'samples':>8}  {'classes':>8}  {'avg/class':>10}  {'id range'}")
        self._log(f"  {'─'*52}")

        all_split_ids = {}
        for skey, content in split_data.items():
            c       = Counter(content['label'])
            n_samp  = len(content['label'])
            n_cls   = len(c)
            id_min  = min(c.keys())
            id_max  = max(c.keys())
            self._log(f"  {skey:<12} {n_samp:>8,}  {n_cls:>8}  "
                      f"{n_samp/n_cls:>10.1f}  [{id_min}..{id_max}]")
            all_split_ids[skey] = set(c.keys())

        # Cross-split overlap
        self._log(f"  {'─'*52}")
        split_keys = list(all_split_ids.keys())
        for i in range(len(split_keys)):
            for j in range(i + 1, len(split_keys)):
                a, b    = split_keys[i], split_keys[j]
                overlap = all_split_ids[a] & all_split_ids[b]
                status  = '✓ disjoint' if len(overlap) == 0 \
                          else f'WARNING: {len(overlap)} shared classes {sorted(overlap)[:5]}{"..." if len(overlap)>5 else ""}'
                self._log(f"  overlap {a} ∩ {b} : {status}")

        all_unique = set(l for c in split_data.values() for l in c['label'])
        self._log(f"  Total unique label IDs : {len(all_unique)}")
        self._log(f"  Label ID range         : {min(all_unique)} – {max(all_unique)}")
        self._log(f"{'─'*55}")

        # ── Wrap as HF DatasetDict ─────────────────────────────────────
        all_labels  = sorted(all_unique)
        label_names = [str(l) for l in all_labels]

        features = Features({
            'image': HFImage(),
            'label': ClassLabel(names=label_names)
        })

        return DatasetDict({
            split: Dataset.from_dict(content, features=features)
            for split, content in split_data.items()
        })

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _finalize_report(self):
        msg = " | ".join([
            f"{k}: {len(v) if v is not None else 'None'}"
            for k, v in self.resources.items()
        ])
        self._log(f"Remote fetch complete. Resources indexed -> {msg}")


@DataSourceRegister.register_source_type("hf")
class HFDataSource(DataSource):
    """
    Handler for Hugging Face Hub datasets.
    Supports both standard downloads and streaming mode for massive datasets.
    """
    @property
    def is_stream(self): return False

    def fetch(self):
        try:
            self._fetch()
            self._finalize_report()
        except Exception as e:
            raise RuntimeError(f"Failed to load HF dataset '{self.identifier}'. Error: {str(e)}")

    def _fetch(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Library 'datasets' not found. Please install using 'pip install datasets'.")

        self._log(f"Connecting to Hugging Face Hub: {self.identifier}")

        # 1. Execute Backend Engine
        # 'sub_type' here acts as the 'name' or 'config' of the dataset 
        # (e.g., 'en' for multilingual datasets)
        common = self._get_common_params()
        clean_kwargs = {k: v for k, v in self.engine_kwargs.items() if k not in common}

        ds = load_dataset(
            path=self.identifier,
            name=self.sub_type,
            **common,                   # cache_root, force_download, num_proc
            **clean_kwargs              # revision, token, trust_remote_code, etc.
        )

        # 2. Map to Resources Dictionary
        if isinstance(ds, dict):
            # Hub datasets usually follow standard naming
            self.resources['train'] = ds.get('train')
            self.resources['test'] = ds.get('test')
            # Hub standard is 'validation', our standard is 'val'
            self.resources['val'] = ds.get('validation') or ds.get('val')
            
            # Fallback for datasets with non-standard single splits
            if not self.resources['train'] and len(ds) == 1:
                self.resources['train'] = list(ds.values())[0]
        else:
            # If the Hub returns a single Dataset object instead of a Dict
            self.resources['train'] = ds
        self.dstype = "hfdataset"  #type(self.resources['train']).__name__

    def _finalize_report(self):
        msg = " | ".join([f"{k}: {len(v) if v else 'Streaming'}" for k, v in self.resources.items()])
        self._log(f"HF Hub Sync complete. Resources -> {msg}")


@DataSourceRegister.register_source_type("stream")
class StreamDataSource(DataSource):
    """
    Physical Layer: High-performance streaming engine for remote or sharded data.
    
    This driver enables 'zero-disk' data access by iterating over remote assets 
    (S3, GCS, HF Hub) without requiring a full local download. It is specifically 
    tuned for Large Language Models (LLMs) and web-scale vision datasets where 
    data volume exceeds local storage capacity.

    Key Features:
    -------------
    1. Dual-Engine Support: Automatically switches between Hugging Face Streaming 
       and WebDataset (.tar shards) based on the identifier pattern.
    2. Zero-Latency Start: Training begins immediately as the first data chunk 
       hits the network buffer.
    3. PyTorch Integration: Returns 'IterableDataset' objects which are 
       native to the torch.utils.data ecosystem.
    """
    @property
    def is_stream(self): return True

    def fetch(self):
        """
        Entry point for establishing the network stream. 
        Decides between WebDataset shards and Hugging Face streaming.
        """
        # Logic: If the identifier contains shard patterns or .tar, use WebDataset
        is_sharded = any(char in self.identifier for char in ["{", ".tar"])
        
        try:
            if is_sharded:
                self._fetch_webdataset()
            else:
                self._fetch_hf_stream()
            self._finalize_report()
        except Exception as e:
            raise RuntimeError(f"Streaming connection failed for {self.identifier}: {str(e)}")

    def _fetch_webdataset(self):
        """
        Logic for Sharded WebDatasets (.tar). 
        Optimized for high-throughput image-text training (e.g., CLIP, LAION).
        """
        try:
            import webdataset as wds
        except ImportError:
            raise ImportError("Library 'webdataset' not found. Please install using 'pip install webdataset'.")

        self._log(f"Opening WebDataset stream: {self.identifier}")
        
        # We wrap the URL/Path in a WebDataset object
        # engine_kwargs can include: shardshuffle, handler, nodesplitter
        dataset = wds.WebDataset(self.identifier, **self.engine_kwargs)
        
        # WebDatasets are typically treated as a continuous training stream
        self.resources['train'] = dataset
        self.dstype = "webdataset"  #type(self.resources['train']).__name__

    def _fetch_hf_stream(self):
        """
        Logic for Hugging Face 'streaming' mode.
        Converts remote Parquet/JSONL/CSV files into a pipe-able stream.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Library 'datasets' not found. Please install using 'pip install datasets'.")
        
        self._log(f"Initializing HF Remote Stream: {self.identifier}")

        # streaming=True is the 'magic' flag that prevents downloading
        safe_kwargs = {k: v for k, v in self.engine_kwargs.items() 
               if k not in ('trust_remote_code', 'num_proc', 'cache_root', 'download_mode')}

        ds_dict = load_dataset(
            path=self.identifier,
            name=self.sub_type,
            streaming=True,
            **safe_kwargs
        )

        # Map splits from the DatasetDict to our Resource Inventory
        if isinstance(ds_dict, dict):
            self.resources['train'] = ds_dict.get('train')
            self.resources['test'] = ds_dict.get('test')
            self.resources['val'] = ds_dict.get('validation') or ds_dict.get('val')
        else:
            self.resources['train'] = ds_dict
        self.dstype = "hfdataset"  #type(self.resources['train']).__name__

    def _finalize_report(self):
        """
        Logs the established streams. 
        Note: We avoid calling len() as streams do not have a fixed size.
        """
        active_splits = [k for k, v in self.resources.items() if v is not None]
        self._log(f"Stream Sync Complete [{self.identifier}]. "
                  f"Active Iterators: {', '.join(active_splits)}")
        

@DataSourceRegister.register_source_type("library")
class LibraryDataSource(DataSource):
    """
    Universal bridge for popular AI libraries: timm, torchvision, torchaudio, torchtext, torch_geometric and sklearn.
    Uses a probe-based split configuration to handle inconsistent library APIs.

    This driver maps various library-specific data formats into a standardized 
    Resource Inventory (train, test, val). It handles:
    - torchvision: split='train' or train=True
    - torchaudio: subset='training'
    - timm: split='train'/'validation'
    - torch_geometric: Mask-based graph datasets
    - sklearn: Bunch-based tabular datasets
    
    Options (via kwargs):
        val_split (float): If > 0 and no native 'val' exists, carves 'val' out of 'train'.
        seed (int): Random seed for the auto-split logic.

    """
    
    # Standardized probe map: (argument_name, value)
    SPLIT_CONFIGS = {
        'train': [('train', True), ('split', 'train'), ('subset', 'training')],
        'test':  [('train', False), ('split', 'test'), ('subset', 'testing')],
        'val':   [('split', 'val'), ('split', 'validation'), ('subset', 'validation')]
    }

    @property
    def is_stream(self): return False

    def fetch(self):
        # Dispatcher mapping
        dispatch_map = {
            'timm': self._fetch_timm,
            'torchvision': self._fetch_torch,
            'torchaudio': self._fetch_torch,
            'torchtext': self._fetch_torch,
            'torch_geometric': self._fetch_pyg,
            'sklearn': self._fetch_sklearn
        }

        fetcher = dispatch_map.get(self.identifier)
        if not fetcher:
            raise ValueError(f"Unsupported library identifier: {self.identifier}")

        try:
            fetcher()
            self._finalize_report()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch {self.identifier}/{self.sub_type}: {e}")

    # --- Library Specific Fetchers ---

    def _fetch_timm(self):
        """timm-specific loading using its factory method."""
        try:
            import timm
        except ImportError:
            raise ImportError("Library 'timm' not found. Please install using 'pip install timm'.")

        for split_key in self.SPLIT_CONFIGS.keys():
            # timm is uniquely 'split' centric
            for arg_name, arg_val in self.SPLIT_CONFIGS[split_key]:
                if arg_name == 'split':
                    try:
                        self.resources[split_key] = timm.data.create_dataset(
                            self.sub_type, root=self._local_cache_path, split=arg_val, **self.engine_kwargs
                        )
                        break # Found a working split for this resource
                    except:
                        continue
        self.dstype = "timmdataset"  #type(self.resources['train']).__name__

    def _fetch_torch(self):
        """Handles standard torchvision, torchaudio, and torchtext datasets."""
        try:
            lib_datasets = importlib.import_module(f"{self.identifier}.datasets")
        except ImportError:
            raise ImportError(f"Library '{self.identifier}' not found. Please install using 'pip install {self.identifier}'.")
        
        ds_class = getattr(lib_datasets, self.sub_type)

        for split_key in self.SPLIT_CONFIGS.keys():
            # Probe using the SPLIT_CONFIGS registry
            for arg_name, arg_val in self.SPLIT_CONFIGS[split_key]:
                try:
                    kwargs = {arg_name: arg_val, 'root': self._local_cache_path, 'download': True}
                    kwargs.update(self.engine_kwargs)
                    self.resources[split_key] = ds_class(**kwargs)
                    break 
                except (TypeError, ValueError):
                    continue # Try next naming convention

        if self.resources.get('train') is None:
            raise RuntimeError(
                f"Failed to load '{self.sub_type}' from '{self.identifier}'. "
                f"All split config probes failed. Check library API compatibility."
            )

        self.dstype = "torchdataset"  #type(self.resources['train']).__name__

    def _fetch_pyg(self):
        """Special handling for Graph data which often uses masks."""
        try:
            pyg_lib = importlib.import_module("torch_geometric.datasets")
        except ImportError:
            raise ImportError("Library 'torch_geometric' not found. Please install using 'pip install torch_geometric'.")
        
        ds_class = getattr(pyg_lib, self.sub_type)
        # PyG usually takes root as a positional or keyword
        self.resources['train'] = ds_class(root=self._local_cache_path, **self.engine_kwargs)
        self.dstype = "pygdataset"  #type(self.resources['train']).__name__

    def _fetch_sklearn(self):
        """Handles scikit-learn 'load_x' style functions."""
        try:
            from sklearn import datasets as sk_dsets
        except ImportError:
            raise ImportError("Library 'scikit-learn' not found. Please install using 'pip install scikit-learn'.")

        loader_name = f"load_{self.sub_type.lower()}"
        loader = getattr(sk_dsets, loader_name, None)
        if not loader:
            raise AttributeError(f"Sklearn has no loader named {loader_name}")

        # Sklearn loaders return a 'Bunch' object (dict-like)
        self.resources['train'] = loader(**self.engine_kwargs)
        self.dstype = "sklearndataset"  #type(self.resources['train']).__name__

    def _finalize_report(self):
        counts = {k: len(v) if v else 0 for k, v in self.resources.items()}
        self._log(f"Synced {self.identifier}: " + " | ".join(f"{k}:{v}" for k, v in counts.items()))


@DataSourceRegister.register_source_type("cloud")
class CloudDataSource(DataSource):
    """
    Physical Layer: Remote cloud storage driver (S3, GCS, Azure).
    
    This driver synchronizes or streams files from cloud buckets into the 
    local cache. It uses 'fsspec' to provide a unified filesystem interface, 
    allowing the 'DiskDataSource' logic to be applied to remote cloud objects.
    
    Config Requirements:
    --------------------
    - identifier: The bucket URI (e.g., 's3://my-bucket/dataset-v1')
    - sub_type: Data modality (e.g., 'image', 'video')
    
    Credentials:
    ------------
    Managed via environment variables (AWS_ACCESS_KEY_ID, etc.) or 
    passed via 'storage_options' in engine_kwargs.
    """

    @property
    def is_stream(self):
        # Cloud is only a stream if specific streaming options are enabled in engine_kwargs
        return self.engine_kwargs.get('stream_from_cloud', False)

    def fetch(self):
        """
        Discovers and retrieves data from the cloud provider.
        """
        try:
            import fsspec
        except ImportError:
            raise ImportError("Library 'fsspec' not found. Please install using 'pip install fsspec'. CloudDataSource requires 'fsspec' and relevant backends (s3fs, gcsfs).")

        self._log(f"Connecting to Cloud Provider: {self.identifier}")

        # 1. Establish Filesystem Connection
        # storage_options handles credentials (key, secret, token)
        storage_options = self.engine_kwargs.pop('storage_options', {})
        fs, path = fsspec.core.url_to_fs(self.identifier, **storage_options)

        # 2. Local Sync / Download Logic
        # We determine if we should sync the whole bucket or treat it as a stream
        is_streaming = self.engine_kwargs.get('stream_from_cloud', False)

        if not is_streaming:
            self._sync_to_local(fs, path)
            # Once synced, we delegate the split discovery to Disk-style logic
            self._discover_local_splits()
        else:
            self._log("Cloud Streaming mode enabled. Bypassing local cache.")
            self._fetch_remote_references(fs, path)

        self._finalize_report()

    def _sync_to_local(self, fs, remote_path):
        """
        Downloads remote bucket contents to the _local_cache_path.
        Uses fsspec.get() which is optimized for recursive directory sync.
        """
        local_dest = os.path.join(self._local_cache_path, self.identifier.replace("://", "_"))
        
        if not os.path.exists(local_dest) or self.force_download:
            self._log(f"Syncing cloud assets to: {local_dest}")
            fs.get(remote_path, local_dest, recursive=True)
        else:
            self._log("Using existing cloud cache.")
            
        # Update the identifier to the local path for split discovery
        self.identifier = local_dest

    def _discover_local_splits(self):
        """
        Reuses the logic from DiskDataSource to find train/test/val folders.
        This runs after the Cloud Sync is complete.
        """
        from datasets import load_dataset
        
        # We use the HF engine to index the newly downloaded cloud files
        builder_map = {"image": "imagefolder", "audio": "audiofolder", "video": "videofolder"}
        builder = builder_map.get(self.sub_type, "imagefolder")

        common = self._get_common_params()
        clean_kwargs = {k: v for k, v in self.engine_kwargs.items() if k not in common}

        ds_dict = load_dataset(
            builder, 
            data_dir=self.identifier, 
            **common,
            **clean_kwargs
        )
        
        self.resources['train'] = ds_dict.get('train')
        self.resources['test'] = ds_dict.get('test')
        self.resources['val'] = ds_dict.get('validation') or ds_dict.get('val')
        self.dstype = "hfdataset"  #type(self.resources['train']).__name__

    def _fetch_remote_references(self, fs, path):
        """
        Advanced: Collects file pointers without downloading the actual bytes.
        Useful for custom PyTorch Dataset implementations that read direct from S3.
        """
        # Collect all file paths in the bucket
        files = fs.find(path)
        self.resources['train'] = files # Raw list of remote pointers
        self.dstype = "fsdataset"  #type(self.resources['train']).__name__
        self._log(f"Discovered {len(files)} remote objects.")

    def _finalize_report(self):
        summary = " | ".join([f"{k}: {len(v) if v else 0}" for k, v in self.resources.items() if v])
        self._log(f"Cloud Sync Complete. Resources -> {summary}")


