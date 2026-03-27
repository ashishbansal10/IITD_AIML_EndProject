"""
model_factory.py
================
Modular, config-driven model composition for Few-Shot Learning.

Design
------
ComponentModel      — abstract base: neural / mathematical / geometric
                      3-level HP API: DEFAULT_HP → config dict → hp tuner
ComponentRegistry   — register & instantiate by name (easyfsl + timm + custom)
SubChain            — sequential chain of ComponentModels as single node
                      supports 'repeat' for Nx stacked layers (e.g. Transformer)
ModelConfig         — dict-based connectivity (Hydra-replaceable via from_yaml)
                      components can be single dict OR list of dicts (SubChain)
CompositeModel      — builds & runs model from config (torch.fx-replaceable via _execute_graph)
ModelFactory        — single entry point: create / save / load

Required Libraries
------------------
# requirements.txt
# torch>=2.0.0
# torchvision>=0.15.0
# easyfsl>=1.4.0              # ResNet12, Conv4 — pip install easyfsl
# torch_geometric>=2.3.0      # GATRelationalLayer, GNNBackbone — pip install torch_geometric
# pyyaml>=6.0                 # ModelConfig.from_yaml() — pip install pyyaml
# optuna>=3.0.0               # HP tuning (tuner.py) — pip install optuna
# pytorch-lightning>=2.0.0    # trainer.py — pip install pytorch-lightning
# hydra-core>=1.3.0           # future: ModelConfig.from_hydra() — pip install hydra-core
# timm>=0.9.0                 # optional fallback for other architectures — pip install timm

Registered Components
---------------------
'basic_cnn'     — 2-block CNN for quick pipeline testing (no external dep, no HP)
'conv4'         — Conv-4 via easyfsl (fallback backbone, output_dim=64)
'resnet12'      — ResNet-12 via easyfsl (primary backbone, output_dim=640)
'linear'        — thin wrapper over nn.Linear (raw logits, learnable)
'softmax'       — thin wrapper over nn.Softmax (probabilities, mathematical)
'prototypical'  — PrototypicalNet nearest-centroid (distances, mathematical)
                  name from Snell et al. 2017 "Prototypical Networks"
'gat_layer'     — Graph Attention via PyG (placeholder — Runs 5,6)
'gnn_backbone'  — Pure GNN backbone via PyG (placeholder — Runs 3,4)

Forward Modes
-------------
mode='embedding'    backbone only          → raw embedding [B, D]
mode='linear'       backbone → linear      → raw logits [B, n_classes]
mode='softmax'      backbone → linear → softmax → probabilities [B, n_classes]
mode='prototypical' prototypical(support_emb, query_emb) → distances [N*Q, N]

Trainer uses:
    mode='linear'      — standard train/val loss (CrossEntropyLoss needs raw logits)
    mode='embedding'   — compute support/query embeddings for episodic
    mode='prototypical'— episodic train/val loss (CrossEntropyLoss on distances)

Evaluator uses:
    mode='linear'      — raw logits → argmax → accuracy (no F.softmax needed)
    mode='softmax'     — direct probabilities for Score reporting
    mode='embedding'   — backbone quality analysis
    mode='prototypical'— direct distances → argmax → accuracy

Why NOT use mode='softmax' for training loss:
    F.cross_entropy(softmax_output, labels) = double softmax = wrong gradients
    F.cross_entropy(raw_logits, labels) applies softmax internally = correct

HP Configuration — 3 Levels Per ComponentModel
-----------------------------------------------
Level 1 — Class DEFAULT_HP  : hardcoded defaults, always available
Level 2 — ModelConfig dict  : override at config definition time
           All keys except 'name', 'role' and 'repeat' in component dict
           are passed as **kwargs to ComponentModel.__init__
           e.g. 'backbone': {'name':'resnet12','role':'backbone','dropout_rate':0.1}
Level 3 — HP Tuner (Optuna) : override at trial time via component.set_hp(**kwargs)
           tuner.py: model.get_component('backbone').set_hp(dropout_rate=0.2)

HP params per component:
    BasicCNN          : no HP config — fixed for testing only
    Conv4             : dropout_rate(0.0)  [easyfsl fixed arch — dropout as wrapper]
    ResNet12          : dropout_rate(0.0)  [easyfsl fixed arch — dropout as wrapper]
    Linear            : embed_dim(*), n_classes(*), dropout_rate(0.0)
    PrototypicalNet   : n_way(5), k_shot(5), distance_metric('euclidean')
    GATRelationalLayer: embed_dim(640), n_heads(4), dropout_rate(0.1),
                        attention_dropout(0.1), k_neighbours(5)
    GNNBackbone       : embed_dim(640), n_layers(3), dropout_rate(0.1),
                        k_neighbours(5), n_heads(4)
    (* = required, no default)

Component Config Format
-----------------------
Single component (dict):
    'backbone': {'name': 'resnet12', 'role': 'backbone', 'dropout_rate': 0.1}

SubChain (list of dicts) — executed sequentially, acts as single node:
    'backbone': [
        {'name': 'resnet12',  'role': 'backbone'},
        {'name': 'gat_layer', 'role': 'backbone', 'n_heads': 4},
    ]

SubChain with repeat — Nx stacked layers (each instance independent weights):
    'encoder': {'name': 'encoder_block', 'role': 'backbone', 'repeat': 6}

Predefined Configs (convenience presets — all accept **hp_overrides)
--------------------------------------------------------------------
ModelConfig.test_config()    — Quick test with BasicCNN (no external deps)
ModelConfig.cnn_config()     — Runs 1 & 2 (CNN Standard / FewShot)
ModelConfig.gnn_config()     — Runs 3 & 4 (GNN Standard / FewShot)
ModelConfig.hybrid_config()  — Runs 5 & 6 (Hybrid Standard / FewShot)
All accept **hp_overrides.

Extensibility
-------------
Replace Hydra   : ModelConfig.from_yaml(path)         — zero CompositeModel change
Replace torch.fx: CompositeModel._execute_graph()     — one method swap
Add component   : @ComponentRegistry.register('name') — one decorator
Swap backbone   : ComponentRegistry.unregister('resnet12')
                  ComponentRegistry.register('resnet12', NewImpl)

Freeze control (works on ComponentModel and SubChain equally)
-------------------------------------------------------------
model.freeze('linear')                # by alias name
model.freeze_by_role('head')          # by role — all head components
model.freeze_all_except('backbone')   # by exclusion
model.unfreeze_by_role('backbone')    # unfreeze by role

-------------------------------------------
Notes — Design Decisions & Status
-------------------------------------------

[DONE] Conv4, ResNet12 — random init only, no pretrained weights.
       dropout_rate applied as nn.Dropout wrapper post-backbone.

[DONE] PrototypicalNet — n_way/k_shot set via ModelConfig, not trainer.
       Trainer computes embeddings, passes to mode='prototypical'.

[DONE] GATRelationalLayer — k-NN graph + GATConv + residual connection.
       Episode-aware: trainer/evaluator pass support+query together.
       Requires: torch-cluster>=1.6.3

[DONE] GNNBackbone — CNN stem (no global pool) → 100 nodes → GCNConv×3
       → global mean pool → [B, 640]. Grid edges precomputed as buffer.

[DONE] TrainingState — pretrain_best_val_acc / pretrain_best_val_loss
       preserved separately before train phase resets best metrics.

[DONE] validate_and_stamp() — deepcopies train/eval configs before stamping.
       Safe for shared configs across ExperimentConfigs.

[DONE] PICKLE_SPLIT_KEYWORDS — train_phase_test/val skipped (None).
       Clean 60,000 sample standard benchmark split preserved.

[DONE] TuneConfig.proxy_epochs — overrides hardcoded max(10,...) in tuner.

[PENDING] ComponentModel._apply_hp() — re-applies HP after set_hp().
          Currently no-op in most subclasses.

[PENDING] ModelFactory.load() — add device consistency check.

[FUTURE] ModelConfig.from_hydra(cfg) — when Hydra integrated.

[FUTURE] Replace _execute_graph() with torch.fx GraphModule.

[FUTURE] PrototypicalNet distance_metric='cosine' — stub present, not implemented.

[FUTURE] detach()/attach() — component hot-swap for novel class fine-tuning.
"""

import os
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import knn_graph
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any


# ==============================================================================
# ComponentModel — Abstract Base
# ==============================================================================

class ComponentModel(nn.Module, ABC):
    """
    Abstract base for all model components.

    Types:
        Neural      : Linear, BasicCNN, Conv4, ResNet12 — learnable parameters
        Mathematical: Softmax, PrototypicalNet — no learnable parameters
        Geometric   : GATRelationalLayer, GNNBackbone — graph neural

    HP Configuration (3 levels):
        Level 1 — DEFAULT_HP class dict  : hardcoded class defaults
        Level 2 — __init__ **kwargs      : from ModelConfig dict/yaml
        Level 3 — set_hp(**kwargs)       : from Optuna tuner at trial time

    Subclasses must implement:
        output_dim : int property — output feature dimensionality
        forward()  : computation

    Inherits full nn.Module:
        .parameters(), .requires_grad_(), .to(device), .state_dict()
    """

    # Level 1 — class-level defaults. Override in each subclass.
    DEFAULT_HP: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        super().__init__()
        # Merge class defaults with any kwargs from ModelConfig
        self._hp = {**self.DEFAULT_HP, **kwargs}

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output feature dimensionality of this component."""
        pass

    # ------------------------------------------------------------------
    # HP API
    # ------------------------------------------------------------------

    def get_hp(self) -> Dict[str, Any]:
        """
        Returns current HP configuration.
        Used by Optuna tuner to inspect and override values.
        """
        return self._hp.copy()

    def set_hp(self, **kwargs):
        """
        Override HP values at runtime. Called by Optuna tuner at trial time.
        Calls _apply_hp() after update to propagate to internal layers.
        """
        self._hp.update(kwargs)
        self._apply_hp()

    def _apply_hp(self):
        """
        Re-applies HP values to internal layers after set_hp().
        Default: no-op. Override in subclass for runtime HP change support.

        [PENDING] Implement where needed:
            Conv4, ResNet12    : update self._dropout.p
            Linear             : update self._dropout.p
            GATRelationalLayer : update dropout rates
        """
        pass

    # ------------------------------------------------------------------
    # Freeze control
    # ------------------------------------------------------------------

    @property
    def is_mathematical(self) -> bool:
        """True if component has no learnable parameters."""
        return sum(p.numel() for p in self.parameters()) == 0

    def freeze(self):
        """Freeze all parameters. No-op for mathematical components."""
        self.requires_grad_(False)

    def unfreeze(self):
        """Unfreeze all learnable parameters."""
        if not self.is_mathematical:
            self.requires_grad_(True)

    def is_frozen(self) -> bool:
        """True if all parameters are frozen or no parameters exist."""
        params = list(self.parameters())
        return (not params) or (not any(p.requires_grad for p in params))

    # ------------------------------------------------------------------
    # Parameter Stats
    # ------------------------------------------------------------------

    def param_count(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        frozen = 'frozen' if self.is_frozen() else 'trainable'
        math   = ', mathematical' if self.is_mathematical else ''
        return (f"{self.__class__.__name__}"
                f"(output_dim={self.output_dim}, {frozen}{math}, "
                f"params={self.param_count():,})")


# ==============================================================================
# SubChain — sequential chain of ComponentModels as single node
# ==============================================================================

class SubChain(ComponentModel):
    """
    Sequential chain of ComponentModels — acts as a single node in CompositeModel.

    Created automatically when a component config entry is a list of dicts:
        'backbone': [
            {'name': 'resnet12',  'role': 'backbone'},
            {'name': 'gat_layer', 'role': 'backbone', 'n_heads': 4},
        ]

    Also handles 'repeat' for Nx stacked layers:
        'encoder': {'name': 'encoder_block', 'role': 'backbone', 'repeat': 6}
        → 6 independent encoder instances (each with own weights) chained sequentially

    Behaviour:
        Single input, single output — sequential execution
        freeze/unfreeze propagates to ALL members
        HP access per member by index
        output_dim = last member's output_dim
        Internally uses nn.ModuleList for PyTorch param tracking

    Individual member control (if needed — prefer alias naming instead):
        chain.freeze_member(0)
        chain.set_hp(index=1, dropout_rate=0.2)
    """

    def __init__(self, components: List[ComponentModel], **kwargs):
        # Don't pass kwargs to ComponentModel.__init__ — SubChain has no own HP
        nn.Module.__init__(self)
        self._hp    = {}
        if not components:
            raise ValueError("SubChain requires at least one ComponentModel.")
        self._chain   = nn.ModuleList(components)
        self._out_dim = components[-1].output_dim

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for component in self._chain:
            x = component(x)
        return x

    # ------------------------------------------------------------------
    # Freeze — propagates to all members
    # ------------------------------------------------------------------

    def freeze(self):
        for comp in self._chain:
            comp.freeze()

    def unfreeze(self):
        for comp in self._chain:
            comp.unfreeze()

    def is_frozen(self) -> bool:
        """True if ALL members are frozen."""
        return all(c.is_frozen() for c in self._chain)

    def freeze_member(self, index: int):
        """Freeze specific member by index."""
        self._chain[index].freeze()

    def unfreeze_member(self, index: int):
        """Unfreeze specific member by index."""
        self._chain[index].unfreeze()

    # ------------------------------------------------------------------
    # HP — per member access
    # ------------------------------------------------------------------

    def get_hp(self, index: int = None) -> Dict:
        """
        Get HP config.
        index=None → {idx: hp_dict} for all members.
        index=N    → hp_dict for member N.
        """
        if index is not None:
            return self._chain[index].get_hp()
        return {i: c.get_hp() for i, c in enumerate(self._chain)}

    def set_hp(self, index: int = None, **kwargs):
        """
        Set HP on member(s).
        index=None → applies to ALL members.
        index=N    → applies to member N only.
        """
        if index is not None:
            self._chain[index].set_hp(**kwargs)
        else:
            for comp in self._chain:
                comp.set_hp(**kwargs)

    def param_count(self) -> int:
        return sum(c.param_count() for c in self._chain)

    def trainable_param_count(self) -> int:
        return sum(c.trainable_param_count() for c in self._chain)

    def __len__(self):
        return len(self._chain)

    def __repr__(self):
        members = [c.__class__.__name__ for c in self._chain]
        return (f"SubChain(members={members}, "
                f"output_dim={self.output_dim}, "
                f"frozen={self.is_frozen()})")


# ==============================================================================
# ComponentRegistry
# ==============================================================================

class ComponentRegistry:
    """
    Registry for ComponentModels — instantiate by name.

    Supports:
        Decorator  : @ComponentRegistry.register('name')
        Manual     : ComponentRegistry.register('name', MyClass)
        timm       : any timm model name as fallback (pretrained=False always)
        easyfsl    : resnet12, conv4 via registered wrappers

    SubChain and repeat handling done in create_from_cfg().

    Usage:
        component = ComponentRegistry.create('resnet12')
        component = ComponentRegistry.create('linear', embed_dim=640, n_classes=64)
        component = ComponentRegistry.create('resnet18')   # timm fallback
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, handler=None):
        """
        Register a ComponentModel by name.
        Works as decorator or manual call.
        """
        def wrapper(component_cls):
            if name in cls._registry:
                raise ValueError(
                    f"ComponentRegistry: '{name}' already registered. "
                    f"Call unregister('{name}') first."
                )
            cls._registry[name] = component_cls
            return component_cls

        if handler is not None:
            return wrapper(handler)
        return wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> ComponentModel:
        """
        Instantiate ComponentModel by name.
        Handles 'repeat' kwarg — creates SubChain of N independent instances.
        Falls back to timm for unregistered names (pretrained=False).

        Args:
            name     : Registered name or timm model name.
            **kwargs : Constructor args. 'repeat' consumed here if present.
        Returns:
            ComponentModel or SubChain.
        """
        repeat = kwargs.pop('repeat', 1)

        if name in cls._registry:
            if repeat > 1:
                # N independent instances — each has own separate weights
                components = [
                    cls._registry[name](**copy.deepcopy(kwargs))
                    for _ in range(repeat)
                ]
                return SubChain(components)
            return cls._registry[name](**kwargs)

        # Fallback — try timm (pretrained=False always)
        try:
            import timm
            timm_model = timm.create_model(name, pretrained=False, num_classes=0)
            return _TimmWrapper(timm_model, **kwargs)
        except Exception:
            pass

        raise ValueError(
            f"ComponentRegistry: '{name}' not found in registry or timm.\n"
            f"Registered names: {cls.list()}\n"
            f"Tip: pip install easyfsl — for resnet12/conv4\n"
            f"Tip: pip install timm    — for other standard architectures"
        )

    @classmethod
    def create_from_cfg(cls, comp_cfg: Union[dict, list]) -> ComponentModel:
        """
        Create ComponentModel from component config entry.
        Handles single dict or list of dicts (→ SubChain).

        Args:
            comp_cfg : dict (single component) or list of dicts (SubChain)
        Returns:
            ComponentModel or SubChain
        """
        if isinstance(comp_cfg, list):
            # List → SubChain
            components = []
            for cfg in comp_cfg:
                name   = cfg['name']
                kwargs = {k: v for k, v in cfg.items()
                          if k not in ('name', 'role')}
                components.append(cls.create(name, **kwargs))
            return SubChain(components)

        elif isinstance(comp_cfg, dict):
            name   = comp_cfg['name']
            kwargs = {k: v for k, v in comp_cfg.items()
                      if k not in ('name', 'role')}
            return cls.create(name, **kwargs)

        raise ValueError(
            f"Component config must be dict or list of dicts. "
            f"Got: {type(comp_cfg)}"
        )

    @classmethod
    def unregister(cls, name: str):
        """
        Remove registered component.
        Use to swap implementations:
            ComponentRegistry.unregister('resnet12')
            ComponentRegistry.register('resnet12', NewResNet12)
        """
        cls._registry.pop(name, None)

    @classmethod
    def list(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def clear(cls):
        cls._registry.clear()


# ==============================================================================
# Timm Wrapper
# ==============================================================================

class _TimmWrapper(ComponentModel):
    """
    Wraps any timm model as a ComponentModel.
    Created automatically by ComponentRegistry for unregistered names.
    Always trained from scratch — pretrained=False explicitly.

    [CONFIRMED] Not used by any current registered component.
    Acts as fallback for future timm model names via registry.
    If timm models added later: verify pretrained=False and num_classes=0.
    """

    DEFAULT_HP: Dict[str, Any] = {'embed_dim': None}

    def __init__(self, timm_model: nn.Module, embed_dim: int = None, **kwargs):
        super().__init__(embed_dim=embed_dim, **kwargs)
        self._model   = timm_model
        self._out_dim = embed_dim or getattr(timm_model, 'num_features', 512)

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


# ==============================================================================
# Built-in ComponentModels
# ==============================================================================

# ------------------------------------------------------------------
# BasicCNN — 2-block CNN for quick pipeline testing only
# ------------------------------------------------------------------

@ComponentRegistry.register('basic_cnn')
class BasicCNN(ComponentModel):
    """
    Minimal 2-block CNN for pipeline testing only.
    No external dependency — pure PyTorch.
    No HP configuration — fixed architecture for testing.
    NOT for real experiments.

    Architecture:
        Conv2d(3,32,3) → BN → ReLU → MaxPool2d(2)
        Conv2d(32,64,3) → BN → ReLU → MaxPool2d(2)
        AdaptiveAvgPool2d(1,1) → Flatten
    Input : [B, 3, H, W]  any size ≥ 8×8
    Output: [B, 64]
    """

    DEFAULT_HP: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._out_dim = 64
        self.features = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


# ------------------------------------------------------------------
# Conv4 — standard few-shot backbone via easyfsl
# ------------------------------------------------------------------

@ComponentRegistry.register('conv4')
class Conv4(ComponentModel):
    """
    Conv-4 backbone — 4 conv blocks, 64-dim embedding.
    Original backbone from Prototypical Networks (Snell et al. 2017).
    Sourced from easyfsl — industry-standard few-shot library.
    Trained from scratch — no pretrained weights.

    Architecture:
        4 × [Conv2d(→64, 3x3) → BN → ReLU → MaxPool2d(2)]
    Input : [B, 3, 84, 84]
    Output: [B, 64]

    HP config (Level 2 via ModelConfig, Level 3 via Optuna):
        dropout_rate (float, 0.0) : post-backbone dropout
                                    easyfsl arch is fixed — added as wrapper

    [CONFIRMED] easyfsl conv4() accepts no kwargs — fixed architecture.
    [CONFIRMED] No pretrained weights — random init.

    Install: pip install easyfsl
    """

    DEFAULT_HP: Dict[str, Any] = {'dropout_rate': 0.0}

    def __init__(self, dropout_rate: float = 0.0, **kwargs):
        super().__init__(dropout_rate=dropout_rate, **kwargs)
        try:
            from easyfsl.modules import conv4
            self._backbone = conv4()
            self._out_dim  = 64
        except ImportError:
            raise ImportError(
                "easyfsl is required for Conv4.\n"
                "Install: pip install easyfsl"
            )
        self._dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._dropout(self._backbone(x))

    def _apply_hp(self):
        """Update dropout wrapper after set_hp(dropout_rate=x)."""
        rate          = self._hp.get('dropout_rate', 0.0)
        self._dropout = nn.Dropout(rate) if rate > 0 else nn.Identity()


# ------------------------------------------------------------------
# ResNet12 — primary few-shot backbone via easyfsl
# ------------------------------------------------------------------

@ComponentRegistry.register('resnet12')
class ResNet12(ComponentModel):
    """
    ResNet-12 backbone — 640-dim embedding.
    Standard backbone for few-shot benchmarks from 2019 onwards.
    Sourced from easyfsl — industry-standard few-shot library.
    Trained from scratch — no pretrained weights.

    Architecture:
        4 residual blocks, channels: 64 → 160 → 320 → 640
        DropBlock regularization (easyfsl internal default)
    Input : [B, 3, 84, 84]
    Output: [B, 640]

    HP config (Level 2 via ModelConfig, Level 3 via Optuna):
        dropout_rate (float, 0.0) : post-backbone dropout
                                    easyfsl arch is fixed — added as wrapper

    [CONFIRMED] easyfsl resnet12() accepts no kwargs — fixed architecture.
    [CONFIRMED] drop_rate (DropBlock) not configurable — easyfsl internal.
    [CONFIRMED] No pretrained weights — random init.

    Install: pip install easyfsl
    """

    DEFAULT_HP: Dict[str, Any] = {'dropout_rate': 0.0}

    def __init__(self, dropout_rate: float = 0.0, **kwargs):
        super().__init__(dropout_rate=dropout_rate, **kwargs)
        try:
            from easyfsl.modules import resnet12
            self._backbone = resnet12()
            self._out_dim  = 640
        except ImportError:
            raise ImportError(
                "easyfsl is required for ResNet12.\n"
                "Install: pip install easyfsl"
            )
        self._dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._dropout(self._backbone(x))

    def _apply_hp(self):
        """Update dropout wrapper after set_hp(dropout_rate=x)."""
        rate          = self._hp.get('dropout_rate', 0.0)
        self._dropout = nn.Dropout(rate) if rate > 0 else nn.Identity()


# ------------------------------------------------------------------
# Linear — thin wrapper over nn.Linear, needed for classifier
# ------------------------------------------------------------------

@ComponentRegistry.register('linear')
class Linear(ComponentModel):
    """
    Linear classification layer — thin wrapper over nn.Linear.
    Always returns RAW LOGITS — no softmax applied.

    Use for training loss:
        logits = model(imgs, mode='linear')
        loss   = F.cross_entropy(logits, labels)   # has softmax internally
        ← NEVER pass softmax output to CrossEntropyLoss — double softmax

    Use for accuracy:
        acc = (logits.argmax(1) == labels).float().mean()
        ← argmax(logits) == argmax(softmax(logits)) — no softmax needed

    For probabilities: use mode='softmax' — Softmax component after this.

    [CONFIRMED] bias=True fixed — final classifier, no BN after.

    HP: embed_dim(*), n_classes(*), dropout_rate(0.0)
    (* = required)
    """

    DEFAULT_HP: Dict[str, Any] = {'dropout_rate': 0.0}

    def __init__(self, embed_dim: int, n_classes: int,
                 dropout_rate: float = 0.0, **kwargs):
        super().__init__(embed_dim=embed_dim, n_classes=n_classes,
                         dropout_rate=dropout_rate, **kwargs)
        self._out_dim = n_classes
        self._dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc       = nn.Linear(embed_dim, n_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self._dropout(x))   # always raw logits

    def _apply_hp(self):
        """Update dropout wrapper after set_hp(dropout_rate=x)."""
        rate          = self._hp.get('dropout_rate', 0.0)
        self._dropout = nn.Dropout(rate) if rate > 0 else nn.Identity()


# ------------------------------------------------------------------
# Softmax — thin wrapper over nn.Softmax
# ------------------------------------------------------------------

@ComponentRegistry.register('softmax')
class Softmax(ComponentModel):
    """
    Softmax layer — thin wrapper over nn.Softmax.
    Mathematical — no learnable parameters.

    Used for evaluation probability reporting only (mode='softmax').
    NEVER used for training loss — CrossEntropyLoss has softmax internally.

    Sits after Linear in graph:
        backbone → linear → softmax

    output_dim = same as input (n_classes)

    HP: dim(1) — softmax dimension
    """

    DEFAULT_HP: Dict[str, Any] = {'dim': 1}

    def __init__(self, dim: int = 1, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self._softmax = nn.Softmax(dim=dim)
        self._out_dim = -1   # same as input — set dynamically

    @property
    def output_dim(self) -> int:
        return self._out_dim   # dynamic — same as input dim

    @property
    def is_mathematical(self) -> bool:
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._softmax(x)


# ------------------------------------------------------------------
# PrototypicalNet — nearest centroid classifier
# ------------------------------------------------------------------

@ComponentRegistry.register('prototypical')
class PrototypicalNet(ComponentModel):
    """
    Prototypical Networks nearest-centroid classifier.
    Prototypical classification head — no learnable parameters.
    Implements Snell et al. 2017 — Prototypical Networks.

    Design note:
        n_way/k_shot set via ModelConfig — protocol config, not HP tunable.
        Trainer passes pre-computed embeddings via mode='embedding'.
        CompositeModel has zero knowledge of N/K/Q.
        Trainer computes embeddings via mode='embedding', passes to mode='prototypical'.

    Algorithm:
        1. prototype[c] = mean(support_emb for class c)   [N, D]
        2. distance[q,c] = euclidean(query_emb[q], prototype[c])
        3. logits = -distance   (nearest = highest score)

    Used for:
        Episodic training loss:
            dists = model(s_emb, q_emb, mode='prototypical')
            loss  = F.cross_entropy(dists, target)   # CrossEntropyLoss has softmax

        Prototype accuracy evaluation:
            dists = model(s_emb, q_emb, mode='prototypical')
            acc   = (dists.argmax(1) == target).float().mean()


    HP (protocol config — not Optuna tunable):
        n_way           (int, 5)           : classes per episode
        k_shot          (int, 5)           : support samples per class
        distance_metric (str, 'euclidean') : 'euclidean' (cosine: [FUTURE])

    Input:
        support_emb : [N*K, D]  — pre-computed via mode='embedding'
        query_emb   : [N*Q, D]  — pre-computed via mode='embedding'
    Output:
        logits      : [N*Q, N]  — negative distances
    """

    DEFAULT_HP: Dict[str, Any] = {
        'n_way':           5,
        'k_shot':          5,
        'distance_metric': 'euclidean',
    }

    def __init__(self, n_way: int = 5, k_shot: int = 5,
                 distance_metric: str = 'euclidean', **kwargs):
        super().__init__(n_way=n_way, k_shot=k_shot,
                         distance_metric=distance_metric, **kwargs)
        self.n_way           = n_way
        self.k_shot          = k_shot
        self.distance_metric = distance_metric
        # Anchor so .to(device) propagates correctly
        self._anchor         = nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def output_dim(self) -> int:
        return -1   # dynamic — depends on n_way at runtime

    @property
    def is_mathematical(self) -> bool:
        return True

    def forward(self,
                support_emb: torch.Tensor,
                query_emb:   torch.Tensor) -> torch.Tensor:
        """
        Args:
            support_emb : [N*K, D]
            query_emb   : [N*Q, D]
        Returns:
            logits      : [N*Q, N_way] — negative Euclidean distances
        """
        D  = support_emb.shape[-1]

        # [N*K, D] → [N, K, D] → mean → [N, D]
        prototypes = (support_emb
                      .view(self.n_way, self.k_shot, D)
                      .mean(dim=1))                          # [N, D]

        if self.distance_metric == 'euclidean':
            dists = torch.cdist(query_emb, prototypes, p=2)
            return -dists   # [N*Q, N]

        elif self.distance_metric == 'cosine':
            # [FUTURE] cosine similarity
            raise NotImplementedError(
                "distance_metric='cosine' not yet implemented. Use 'euclidean'."
            )
        else:
            raise ValueError(
                f"Unknown distance_metric: '{self.distance_metric}'. "
                f"Use 'euclidean' or 'cosine'."
            )


# ------------------------------------------------------------------
# GATRelationalLayer — Graph Attention (placeholder, Runs 5 & 6)
# ------------------------------------------------------------------


@ComponentRegistry.register('gat_layer')
class GATRelationalLayer(ComponentModel):
    """
    Graph Attention Network relational reasoning layer.
    Part of Hybrid CNN-GNN backbone (Runs 5 & 6).
 
    Sits after ResNet12 in a SubChain — takes per-image embeddings,
    builds a k-NN graph over them, runs one round of GAT message passing,
    and returns refined embeddings of the same shape.
 
    Key design decisions:
        - Graph built internally from embeddings — caller passes [B, D] only.
          SubChain compatibility: single tensor in, single tensor out.
        - knn_graph uses detached embeddings for graph construction.
          Gradients flow through GATConv weights, not through graph topology.
        - Residual connection: output = GAT(x) + x
          Ensures stable training — if GAT attention is unhelpful, model
          can learn to zero out the GAT contribution via weights.
        - Safe fallback when B <= k_neighbours (e.g. small last batch).
 
    In episodic training (Run 6):
        Trainer must pass support + query TOGETHER through backbone so the
        GAT sees the full episode graph — see trainer.py note at bottom.
 
    HP config:
        embed_dim         (int,   640) : feature dimensionality
        n_heads           (int,   4)   : GAT attention heads
        dropout_rate      (float, 0.1) : post-GAT node dropout
        attention_dropout (float, 0.1) : attention coefficient dropout
        k_neighbours      (int,   5)   : k-NN graph construction
 
    Input : [B, embed_dim]
    Output: [B, embed_dim]
 
    Requires: pip install torch_geometric
    """
 
    DEFAULT_HP: Dict[str, Any] = {
        'embed_dim':         640,
        'n_heads':           4,
        'dropout_rate':      0.1,
        'attention_dropout': 0.1,
        'k_neighbours':      5,
    }
 
    def __init__(self, embed_dim: int = 640, n_heads: int = 4,
                 dropout_rate: float = 0.1, attention_dropout: float = 0.1,
                 k_neighbours: int = 5, **kwargs):
        super().__init__(embed_dim=embed_dim, n_heads=n_heads,
                         dropout_rate=dropout_rate,
                         attention_dropout=attention_dropout,
                         k_neighbours=k_neighbours, **kwargs)
        self._out_dim     = embed_dim
        self.embed_dim    = embed_dim
        self.n_heads      = n_heads
        self.k_neighbours = k_neighbours
 
        try:
            # out_channels per head = embed_dim // n_heads
            # concat=True → output = out_channels * n_heads = embed_dim
            # This preserves output_dim = embed_dim for the residual to work.
            self.gat = GATConv(
                in_channels  = embed_dim,
                out_channels = embed_dim // n_heads,
                heads        = n_heads,
                concat       = True,
                dropout      = attention_dropout,
                add_self_loops = True,   # each node attends to itself too
            )
            self._dropout        = (nn.Dropout(dropout_rate)
                                    if dropout_rate > 0 else nn.Identity())
            self._available      = True
        except ImportError:
            # PyG not installed — fall back to identity (pipeline still runs)
            self.gat             = nn.Identity()
            self._dropout        = nn.Identity()
            self._available      = False
            print("Warning: torch_geometric not found. "
                  "GATRelationalLayer running as identity. "
                  "pip install torch_geometric")
 
    @property
    def output_dim(self) -> int:
        return self._out_dim
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, embed_dim]  — per-image embeddings from ResNet12
        Returns:
            x : [B, embed_dim]  — refined embeddings (same shape)
        """
        if not self._available:
            return x   # identity fallback if PyG not installed
  
        B = x.size(0)
 
        # Safe k — can't have more neighbours than other nodes
        k = min(self.k_neighbours, B - 1)
        if k < 1:
            # Only 1 node — nothing to aggregate from
            return x
 
        # Build k-NN graph from current embeddings.
        # Detach so graph construction doesn't affect gradients —
        # only GATConv weights are trained, not the graph topology.
        with torch.no_grad():
            edge_index = knn_graph(
                x.detach(),
                k    = k,
                loop = False,    # self-loops handled by add_self_loops in GATConv
            )
        # edge_index: [2, B*k]  — on same device as x
 
        # GAT message passing — gradients flow through gat weights
        x_gat = self.gat(x, edge_index)      # [B, embed_dim]
        x_gat = self._dropout(x_gat)
 
        # Residual: preserves pretrain backbone information.
        # If GAT adds noise early in training, residual keeps the
        # original embedding dominant until GAT learns to help.
        return x + x_gat
 

# ------------------------------------------------------------------
# GNNBackbone — Pure GNN backbone (placeholder, Runs 3 & 4)
# ------------------------------------------------------------------

@ComponentRegistry.register('gnn_backbone')
class GNNBackbone(ComponentModel):
    """
    Pure GNN feature extractor — replaces CNN backbone entirely (Runs 3 & 4).
 
    Architecture:
        CNN stem (3 conv blocks, NO global pool) → spatial feature map
        Flatten spatial positions → graph nodes
        GCN message passing over spatial grid
        Global mean pool over nodes → single embedding per image
 
    Why CNN stem + GNN (not pure pixel GNN):
        Raw pixels give poor node features (RGB only).
        CNN stem learns local texture/edge features first, then GNN
        reasons over spatial relationships between those richer features.
        This matches the report description: "node features initialised
        from pixel-level representations".
 
    CNN stem design:
        Input:  [B, 3,   84, 84]
        Block1: [B, 64,  42, 42]  (conv3x3 + BN + ReLU + MaxPool2)
        Block2: [B, 128, 21, 21]  (conv3x3 + BN + ReLU + MaxPool2)
        Block3: [B, 256, 10, 10]  (conv3x3 + BN + ReLU + MaxPool2)
        → 100 nodes per image, each with 256-dim feature
 
    Graph construction:
        Grid edges — each spatial position connects to its 8 neighbours.
        Precomputed once in __init__ as edge template, expanded per batch.
        Grid edges are natural for spatial reasoning (locality preserved).
        No hyperparameter needed (unlike k-NN).
 
    GNN layers:
        Linear projection: 256 → embed_dim (640)
        N × GCNConv(embed_dim, embed_dim) + ReLU + Dropout
 
    Global pooling:
        Mean over all 100 nodes per image → [B, embed_dim]
        Same output shape as ResNet12 — downstream head unchanged.
 
    HP config:
        embed_dim     (int,   640) : output feature dimensionality
        n_layers      (int,   3)   : number of GCN message passing layers
        dropout_rate  (float, 0.1) : between GCN layers
        k_neighbours  (int,   5)   : unused (grid edges used instead)
        n_heads       (int,   4)   : unused (GCN used instead of GAT)
 
    Input : [B, 3, H, W]  — H=W=84
    Output: [B, embed_dim]
 
    Requires: pip install torch_geometric
    """
 
    DEFAULT_HP: Dict[str, Any] = {
        'embed_dim':    640,
        'n_layers':     3,
        'dropout_rate': 0.1,
        'k_neighbours': 5,    # unused — grid edges used
        'n_heads':      4,    # unused — GCN used
    }
 
    # CNN stem output: 3 blocks of (conv+bn+relu+maxpool2) on 84x84
    # 84 → 42 → 21 → 10 (floor division at each pool)
    STEM_CHANNELS  = [64, 128, 256]   # output channels per block
    SPATIAL_SIZE   = 10               # H' = W' = 10 after 3 pools
    STEM_OUT_DIM   = 256              # channels after stem
 
    def __init__(self, embed_dim: int = 640, n_layers: int = 3,
                 dropout_rate: float = 0.1, k_neighbours: int = 5,
                 n_heads: int = 4, **kwargs):
        super().__init__(embed_dim=embed_dim, n_layers=n_layers,
                         dropout_rate=dropout_rate,
                         k_neighbours=k_neighbours,
                         n_heads=n_heads, **kwargs)
        self._out_dim    = embed_dim
        self.embed_dim   = embed_dim
        self.n_layers    = n_layers
        self.dropout_rate = dropout_rate
 
        # ── CNN stem (no global pool) ──────────────────────────────────
        stem_layers = []
        in_ch = 3
        for out_ch in self.STEM_CHANNELS:
            stem_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ]
            in_ch = out_ch
        self.stem = nn.Sequential(*stem_layers)
        # stem output: [B, 256, 10, 10]
 
        # ── Project stem features to embed_dim ────────────────────────
        # Linear projection per node: 256 → embed_dim
        # Applied as Conv1d over node dimension for efficiency
        self.node_proj = nn.Linear(self.STEM_OUT_DIM, embed_dim)
 
        # ── GCN message passing layers ─────────────────────────────────
        try:
            self.convs = nn.ModuleList([
                GCNConv(embed_dim, embed_dim) for _ in range(n_layers)
            ])
            self._dropout  = (nn.Dropout(dropout_rate)
                              if dropout_rate > 0 else nn.Identity())
            self._available = True
        except ImportError:
            self.convs      = nn.ModuleList()
            self._dropout   = nn.Identity()
            self._available = False
            print("Warning: torch_geometric not found. "
                  "GNNBackbone will use stem+pool only (no GCN). "
                  "pip install torch_geometric")
 
        # ── Precompute grid edge template for one image ────────────────
        # Reused across all batches — only node indices shift per image.
        # 8-connected grid on SPATIAL_SIZE x SPATIAL_SIZE.
        self.register_buffer(
            '_edge_template',
            self._build_grid_edges(self.SPATIAL_SIZE)
        )
        # _edge_template: [2, E] — edge pairs for one 10x10 grid
 
        self._nodes_per_image = self.SPATIAL_SIZE * self.SPATIAL_SIZE  # 100
 
    # ------------------------------------------------------------------
    # Grid edge construction — called once in __init__
    # ------------------------------------------------------------------
 
    @staticmethod
    def _build_grid_edges(size: int) -> torch.Tensor:
        """
        Build 8-connected grid edge_index for a size×size spatial grid.
        Returns undirected edges stored as bidirectional pairs.
        Shape: [2, E]  where E ≈ size*size*4 (avg ~4 edges per node)
 
        Node index for position (row, col) = row * size + col
        8 neighbours: all (row+dr, col+dc) for dr,dc in {-1,0,1}² except (0,0)
        """
        src, dst = [], []
        for r in range(size):
            for c in range(size):
                node = r * size + c
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < size and 0 <= nc < size:
                            nb = nr * size + nc
                            src.append(node)
                            dst.append(nb)
        return torch.tensor([src, dst], dtype=torch.long)

    @property
    def output_dim(self) -> int:
        return self._out_dim
 
    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, H, W]  — raw images, H=W=84
        Returns:
            x : [B, embed_dim] — graph-pooled embedding
        """
        B = x.size(0)
 
        # ── Step 1: CNN stem — extract spatial features ────────────────
        feat = self.stem(x)                    # [B, 256, 10, 10]
 
        # ── Step 2: Flatten spatial → nodes ───────────────────────────
        # [B, 256, 10, 10] → [B, 100, 256] → [B*100, 256]
        feat = feat.permute(0, 2, 3, 1)        # [B, 10, 10, 256]
        feat = feat.reshape(B * self._nodes_per_image, self.STEM_OUT_DIM)
 
        # ── Step 3: Project to embed_dim ──────────────────────────────
        feat = self.node_proj(feat)            # [B*100, embed_dim]
        feat = F.relu(feat)
 
        if not self._available or len(self.convs) == 0:
            # PyG not installed — skip GCN, just pool stem features
            feat = feat.view(B, self._nodes_per_image, self.embed_dim)
            return feat.mean(dim=1)            # [B, embed_dim]
 
        # ── Step 4: Build batch-expanded edge_index ───────────────────
        # _edge_template covers one image (nodes 0..99).
        # For B images, shift node indices by i*100 for image i.
        E      = self._edge_template.size(1)
        offset = torch.arange(B, device=x.device) * self._nodes_per_image
        # offset: [B]  →  broadcast to [B, E]
        src    = (self._edge_template[0].unsqueeze(0) +
                  offset.unsqueeze(1)).reshape(-1)        # [B*E]
        dst    = (self._edge_template[1].unsqueeze(0) +
                  offset.unsqueeze(1)).reshape(-1)        # [B*E]
        edge_index = torch.stack([src, dst], dim=0)       # [2, B*E]
 
        # ── Step 5: GCN message passing ───────────────────────────────
        for conv in self.convs:
            feat = conv(feat, edge_index)      # [B*100, embed_dim]
            feat = F.relu(feat)
            feat = self._dropout(feat)
 
        # ── Step 6: Global mean pool per image ────────────────────────
        feat = feat.view(B, self._nodes_per_image, self.embed_dim)
        return feat.mean(dim=1)                # [B, embed_dim]



# ==============================================================================
# ModelConfig — dict-based connectivity
# ==============================================================================

class ModelConfig:
    """
    Dict-based model connectivity configuration.

    Hydra replacement: ModelConfig.from_yaml(path) — same structure.
    [PENDING] ModelConfig.from_hydra(cfg) for OmegaConf.DictConfig.

    Structure:
        components : {alias: dict OR list_of_dicts}
            alias   : user-defined name used in graph edges and outputs
                      can be same as registry name (e.g. 'linear', 'softmax',
                      'prototypical') or different (e.g. 'backbone' for any
                      registered backbone like 'resnet12', 'conv4', 'gnn_backbone')
            name    : registered ComponentModel name in ComponentRegistry
            role    : 'backbone' or 'head'
            rest    : HP kwargs → passed as **kwargs to ComponentModel.__init__

        graph : [[src_alias, dst_alias], ...]
            'input' = reserved source node — not a component
            all other nodes must be aliases defined in components

        outputs : {mode: alias}
            Maps forward mode to component alias that produces that output.
            Order: embedding → linear → softmax → prototypical
            'embedding'    → backbone alias  (raw embedding)
            'linear'       → 'linear'        (raw logits — training loss)
            'softmax'      → 'softmax'       (probabilities — eval reporting)
            'prototypical' → 'prototypical'  (distances — episodic train+eval)

        frozen : [alias, ...] — optional, frozen at model init

    Example — CNN (Runs 1 & 2):
        {
            'components': {
                'backbone':     {'name': 'resnet12',     'role': 'backbone',
                                 'dropout_rate': 0.0},
                'linear':       {'name': 'linear',       'role': 'head',
                                 'embed_dim': 640, 'n_classes': 64},
                'softmax':      {'name': 'softmax',      'role': 'head'},
                'prototypical': {'name': 'prototypical', 'role': 'head',
                                 'n_way': 5, 'k_shot': 5},
            },
            'graph': [
                ['input',    'backbone'],
                ['backbone', 'linear'],
                ['linear',   'softmax'],
                ['backbone', 'prototypical'],
            ],
            'outputs': {
                'embedding':    'backbone',
                'linear':       'linear',
                'softmax':      'softmax',
                'prototypical': 'prototypical',
            }
        }
    """

    def __init__(self, config_dict: dict):
        self._cfg = config_dict
        self._validate()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelConfig':
        """Primary usage — inline dict in notebook."""
        return cls(copy.deepcopy(d))

    @classmethod
    def from_json(cls, path: str) -> 'ModelConfig':
        with open(path, 'r') as f:
            return cls(json.load(f))

    @classmethod
    def from_yaml(cls, path: str) -> 'ModelConfig':
        """
        Hydra-compatible YAML — same structure as from_dict.
        Zero migration needed when adding Hydra.
        pip install pyyaml
        """
        try:
            import yaml
            with open(path, 'r') as f:
                return cls(yaml.safe_load(f))
        except ImportError:
            raise ImportError(
                "pyyaml required for YAML config.\n"
                "Install: pip install pyyaml\n"
                "Or use ModelConfig.from_dict() with inline dict."
            )

    def to_dict(self) -> dict:
        return copy.deepcopy(self._cfg)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self._cfg, f, indent=2)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self):
        """Validates config structure and graph consistency."""
        required = ['components', 'graph', 'outputs']
        for key in required:
            if key not in self._cfg:
                raise ValueError(f"ModelConfig missing '{key}'. Required: {required}")

        valid_nodes = set(self._cfg['components'].keys()) | {'input'}

        for src, dst in self._cfg['graph']:
            if src not in valid_nodes:
                raise ValueError(f"Graph source '{src}' not in components or 'input'.")
            if dst not in self._cfg['components']:
                raise ValueError(f"Graph destination '{dst}' not in components.")

        for mode, node in self._cfg['outputs'].items():
            if node not in self._cfg['components']:
                raise ValueError(f"Output '{mode}' → '{node}' not in components.")

        # Validate each component entry has 'name' and 'role'
        for alias, comp_cfg in self._cfg['components'].items():
            cfgs = comp_cfg if isinstance(comp_cfg, list) else [comp_cfg]
            for cfg in cfgs:
                if 'name' not in cfg:
                    raise ValueError(f"Component '{alias}' missing 'name'.")
                if 'role' not in cfg:
                    raise ValueError(f"Component '{alias}' missing 'role'. Use 'backbone' or 'head'.")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def components(self) -> dict:
        return self._cfg['components']

    @property
    def graph(self) -> list:
        return self._cfg['graph']

    @property
    def outputs(self) -> dict:
        return self._cfg['outputs']

    @property
    def frozen_at_init(self) -> list:
        return self._cfg.get('frozen', [])

    def component_names(self) -> List[str]:
        """All component alias names."""
        return list(self._cfg['components'].keys())

    def components_by_role(self, role: str) -> List[str]:
        """Component aliases with given role."""
        result = []
        for alias, comp_cfg in self._cfg['components'].items():
            cfgs = comp_cfg if isinstance(comp_cfg, list) else [comp_cfg]
            if any(c.get('role') == role for c in cfgs):
                result.append(alias)
        return result

    def __repr__(self):
        return (f"ModelConfig(components={self.component_names()}, "
                f"outputs={list(self.outputs.keys())})")

    # ------------------------------------------------------------------
    # Predefined configs — convenience presets for all 6 runs
    # All accept **hp_overrides for any HP value.
    # Outputs always ordered: embedding → linear → softmax → prototypical
    # ------------------------------------------------------------------

    @classmethod
    def test_config(cls, n_classes: int = 64, **hp_overrides) -> 'ModelConfig':
        """
        Quick test config — BasicCNN, no external dependencies.
        Pipeline testing before plugging in real models.

        Args:
            n_classes     : base classes (default 64)
            **hp_overrides: n_way, k_shot
        """
        return cls.from_dict({
            'components': {
                'backbone':     {'name': 'basic_cnn',    'role': 'backbone'},
                'linear':       {'name': 'linear',       'role': 'head',
                                 'embed_dim': 64, 'n_classes': n_classes},
                'softmax':      {'name': 'softmax',      'role': 'head'},
                'prototypical': {'name': 'prototypical', 'role': 'head',
                                 'n_way':  hp_overrides.pop('n_way',  5),
                                 'k_shot': hp_overrides.pop('k_shot', 5)},
            },
            'graph': [
                ['input',    'backbone'],
                ['backbone', 'linear'],
                ['linear',   'softmax'],
                ['backbone', 'prototypical'],
            ],
            'outputs': {
                'embedding':    'backbone',
                'linear':       'linear',
                'softmax':      'softmax',
                'prototypical': 'prototypical',
            }
        })

    @classmethod
    def cnn_config(cls, backbone: str = 'resnet12', n_classes: int = 64, **hp_overrides) -> 'ModelConfig':
        """
        CNN config — Runs 1 (Standard) and 2 (FewShot).
        Architecture: ResNet12 → Linear → Softmax / Prototypical

        Primary backbone : resnet12 (output_dim=640)
        Fallback backbone: conv4    (output_dim=64)

        hp_overrides: backbone_dropout, head_dropout, n_way, k_shot, distance_metric
        """
        embed_dim = 640 if backbone == 'resnet12' else 64
        return cls.from_dict({
            'components': {
                'backbone': {
                    'name': backbone, 'role': 'backbone',
                    'dropout_rate': hp_overrides.pop('backbone_dropout', 0.0),
                },
                'linear': {
                    'name': 'linear', 'role': 'head',
                    'embed_dim': embed_dim,
                    'n_classes': n_classes,
                    'dropout_rate': hp_overrides.pop('head_dropout', 0.0),
                },
                'softmax': {'name': 'softmax', 'role': 'head'},
                'prototypical': {
                    'name': 'prototypical', 'role': 'head',
                    'n_way':           hp_overrides.pop('n_way',            5),
                    'k_shot':          hp_overrides.pop('k_shot',           5),
                    'distance_metric': hp_overrides.pop('distance_metric', 'euclidean'),
                },
            },
            'graph': [
                ['input',    'backbone'],
                ['backbone', 'linear'],
                ['linear',   'softmax'],
                ['backbone', 'prototypical'],
            ],
            'outputs': {
                'embedding':    'backbone',
                'linear':       'linear',
                'softmax':      'softmax',
                'prototypical': 'prototypical',
            }
        })

    @classmethod
    def gnn_config(cls, n_classes: int = 64,  embed_dim: int = 640, **hp_overrides) -> 'ModelConfig':
        """
        GNN config — Runs 3 (Standard) and 4 (FewShot).
        Architecture: GNNBackbone → Linear → Softmax / Prototypical
        PLACEHOLDER — full implementation Phase 6.

        hp_overrides: n_layers, backbone_dropout, k_neighbours, n_heads, head_dropout, n_way, k_shot
        """
        return cls.from_dict({
            'components': {
                'backbone': {
                    'name': 'gnn_backbone', 'role': 'backbone',
                    'embed_dim':    embed_dim,
                    'n_layers':     hp_overrides.pop('n_layers',     3),
                    'dropout_rate': hp_overrides.pop('backbone_dropout', 0.1),
                    'k_neighbours': hp_overrides.pop('k_neighbours', 5),
                    'n_heads':      hp_overrides.pop('n_heads',      4),
                },
                'linear': {
                    'name': 'linear', 'role': 'head',
                    'embed_dim': embed_dim,
                    'n_classes': n_classes,
                    'dropout_rate': hp_overrides.pop('head_dropout', 0.0),
                },
                'softmax': {'name': 'softmax', 'role': 'head'},
                'prototypical': {
                    'name': 'prototypical', 'role': 'head',
                    'n_way':  hp_overrides.pop('n_way',  5),
                    'k_shot': hp_overrides.pop('k_shot', 5),
                },
            },
            'graph': [
                ['input',    'backbone'],
                ['backbone', 'linear'],
                ['linear',   'softmax'],
                ['backbone', 'prototypical'],
            ],
            'outputs': {
                'embedding':    'backbone',
                'linear':       'linear',
                'softmax':      'softmax',
                'prototypical': 'prototypical',
            }
        })

    @classmethod
    def hybrid_config(cls, cnn_backbone: str = 'resnet12', n_classes: int = 64, **hp_overrides) -> 'ModelConfig':
        """
        Hybrid CNN-GNN config — Runs 5 (Standard) and 6 (FewShot).
        Architecture: [ResNet12 → GATRelationalLayer] → Linear → Softmax / Prototypical
        Backbone is SubChain — ResNet12 + GATRelationalLayer as single node.
        PLACEHOLDER — GATRelationalLayer full implementation Phase 6.

        hp_overrides: cnn_dropout, n_heads, gat_dropout, attention_dropout, k_neighbours, head_dropout, n_way, k_shot
        """
        embed_dim = 640 if cnn_backbone == 'resnet12' else 64
        return cls.from_dict({
            'components': {
                # SubChain — list of dicts
                'backbone': [
                    {
                        'name': cnn_backbone, 'role': 'backbone',
                        'dropout_rate': hp_overrides.pop('cnn_dropout', 0.0),
                    },
                    {
                        'name':              'gat_layer', 'role': 'backbone',
                        'embed_dim':         embed_dim,
                        'n_heads':           hp_overrides.pop('n_heads',           4),
                        'dropout_rate':      hp_overrides.pop('gat_dropout',       0.1),
                        'attention_dropout': hp_overrides.pop('attention_dropout', 0.1),
                        'k_neighbours':      hp_overrides.pop('k_neighbours',      5),
                    },
                ],
                'linear': {
                    'name': 'linear', 'role': 'head',
                    'embed_dim': embed_dim,
                    'n_classes': n_classes,
                    'dropout_rate': hp_overrides.pop('head_dropout', 0.0),
                },
                'softmax': {'name': 'softmax', 'role': 'head'},
                'prototypical': {
                    'name': 'prototypical', 'role': 'head',
                    'n_way':  hp_overrides.pop('n_way',  5),
                    'k_shot': hp_overrides.pop('k_shot', 5),
                },
            },
            'graph': [
                ['input',    'backbone'],
                ['backbone', 'linear'],
                ['linear',   'softmax'],
                ['backbone', 'prototypical'],
            ],
            'outputs': {
                'embedding':    'backbone',
                'linear':       'linear',
                'softmax':      'softmax',
                'prototypical': 'prototypical',
            }
        })


# ==============================================================================
# CompositeModel — builds and runs model from ModelConfig
# ==============================================================================

class CompositeModel(nn.Module):
    """
    Builds and runs a model from ModelConfig.

    Internals use nn.ModuleDict + manual graph execution.
    Designed for easy torch.fx swap:
        Only _execute_graph() changes — nothing else affected.

    Component building:
        Single dict config → ComponentModel registered directly
        List of dicts      → SubChain registered as single node
        'repeat' in config → SubChain of N independent instances

    Freeze control (works identically on ComponentModel and SubChain):
        freeze(name)              — by alias
        freeze_by_role(role)      — all with given role
        freeze_all_except(*names) — by exclusion
        unfreeze / unfreeze_by_role / unfreeze_all

    HP access per component:
        model.get_component('backbone').get_hp()
        model.get_component('backbone').set_hp(dropout_rate=0.2)
        # SubChain member: model.get_component('backbone').set_hp(index=0, dropout_rate=0.2)

    Forward modes:
        outputs dict defines available modes
            mode='embedding'    backbone only — raw embedding
            mode='linear'       backbone → linear — raw logits (TRAINING)
            mode='softmax'      backbone → linear → softmax — probabilities (EVAL)
            mode='prototypical' prototypical(support_emb, query_emb) — distances

        # Standard train / softmax eval:
        logits = model(imgs, mode='linear')

        # Backbone embedding (trainer uses this for proto):
        emb = model(imgs, mode='embedding')

        # Episodic train / proto eval (trainer pre-computes embeddings):
        support_emb = model(support, mode='embedding')   # [N*K, D]
        query_emb   = model(query,   mode='embedding')   # [N*Q, D]
        logits = model(support_emb=support_emb, query_emb=query_emb, mode='prototypical')
    """

    def __init__(self, config: ModelConfig, device: torch.device = None):
        super().__init__()
        self._config  = config
        self._device  = device
        self._roles   = {}

        # Build nn.ModuleDict — PyTorch tracks all params, .to(device) propagates
        self._components = nn.ModuleDict()

        for alias, comp_cfg in config.components.items():
            component               = ComponentRegistry.create_from_cfg(comp_cfg)
            self._components[alias] = component
            first_cfg               = (comp_cfg[0] if isinstance(comp_cfg, list)
                                       else comp_cfg)
            self._roles[alias]      = first_cfg.get('role', 'backbone')

        self._exec_order = list(config.graph)

        for alias in config.frozen_at_init:
            self.freeze(alias)

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Graph execution — isolated for torch.fx swap
    # ------------------------------------------------------------------

    def _execute_graph(self,
                       imgs:        Optional[torch.Tensor],
                       support_emb: Optional[torch.Tensor],
                       query_emb:   Optional[torch.Tensor],
                       mode:        str) -> torch.Tensor:
        """
        Executes computation graph and returns output for requested mode.

        Current: manual forward following edge list.
        torch.fx replacement: replace this method body only.

        Prototypical mode:
            Trainer pre-computes embeddings via mode='embedding'.
            Passed directly to PrototypicalNet — no backbone re-execution.
            N/K internal to PrototypicalNet — not passed here.
        """
        if mode == 'prototypical':
            if support_emb is None or query_emb is None:
                raise ValueError(
                    "mode='prototypical' requires support_emb and query_emb.\n"
                    "Compute via model(x, mode='embedding') in trainer first."
                )
            proto_alias = self._config.outputs['prototypical']
            return self._components[proto_alias](support_emb, query_emb)

        if imgs is None:
            raise ValueError(f"mode='{mode}' requires imgs tensor.")

        node_outputs: Dict[str, torch.Tensor] = {'input': imgs}
        target_alias = self._config.outputs[mode]

        for src, dst in self._exec_order:
            if dst in node_outputs:
                continue
            if src not in node_outputs:
                continue
            node_outputs[dst] = self._components[dst](node_outputs[src])
            if dst == target_alias:
                break

        if target_alias not in node_outputs:
            raise RuntimeError(
                f"Target '{target_alias}' not reached. "
                f"Check graph edges in ModelConfig."
            )

        return node_outputs[target_alias]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self,
                imgs:        Optional[torch.Tensor] = None,
                support_emb: Optional[torch.Tensor] = None,
                query_emb:   Optional[torch.Tensor] = None,
                mode:        str = 'linear') -> torch.Tensor:
        """
        Unified forward — mode selects output branch.

        Standard training:
            logits = model(imgs, mode='linear')
            loss   = F.cross_entropy(logits, labels)

        Episodic training:
            s_emb  = model(support, mode='embedding')
            q_emb  = model(query,   mode='embedding')
            dists  = model(s_emb, q_emb, mode='prototypical')
            loss   = F.cross_entropy(dists, target)

        Softmax eval (Score 1, 3, 5):
            probs = model(imgs, mode='softmax')

        Proto eval (Score 2, 4, 6, 7, 8, 9):
            dists = model(s_emb, q_emb, mode='prototypical')
            acc   = (dists.argmax(1) == target).float().mean()
        """
        if mode not in self._config.outputs:
            raise ValueError(
                f"mode='{mode}' not in outputs: {list(self._config.outputs.keys())}"
            )
        return self._execute_graph(imgs, support_emb, query_emb, mode)

    # ------------------------------------------------------------------
    # Freeze — by name
    # ------------------------------------------------------------------

    def freeze(self, name: str):
        self._get_component(name).freeze()

    def unfreeze(self, name: str):
        self._get_component(name).unfreeze()

    def is_frozen(self, name: str) -> bool:
        return self._get_component(name).is_frozen()

    def freeze_all(self):
        for comp in self._components.values(): comp.freeze()

    def unfreeze_all(self):
        for comp in self._components.values(): comp.unfreeze()

    def freeze_all_except(self, *names: str):
        for alias, comp in self._components.items():
            comp.freeze() if alias not in names else comp.unfreeze()

    # ------------------------------------------------------------------
    # Freeze — by role
    # ------------------------------------------------------------------

    def freeze_by_role(self, role: str):
        for alias, comp_role in self._roles.items():
            if comp_role == role:
                self._components[alias].freeze()

    def unfreeze_by_role(self, role: str):
        for alias, comp_role in self._roles.items():
            if comp_role == role:
                self._components[alias].unfreeze()

    def frozen_names(self) -> List[str]:
        return [n for n in self._components if self._components[n].is_frozen()]

    def trainable_names(self) -> List[str]:
        return [n for n in self._components if not self._components[n].is_frozen()]

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        """Parameters with requires_grad=True only."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_groups(self, lr_map: dict = None,
                               default_lr: float = 1e-3) -> list:
        """
        Per-component optimizer param groups.

        Usage:
            optimizer = Adam(model.trainable_param_groups(
                lr_map={'backbone': 1e-4, 'linear': 1e-3}
            ))

        [CONFIRMED] 'name' key for debugging — optimizer ignores unknown keys.
        """
        if lr_map is None:
            return [{'params': self.trainable_parameters(), 'lr': default_lr}]

        groups = []
        for alias, component in self._components.items():
            if component.is_frozen():
                continue
            trainable = [p for p in component.parameters() if p.requires_grad]
            if not trainable:
                continue
            lr = lr_map.get(alias, lr_map.get('default', default_lr))
            groups.append({'params': trainable, 'lr': lr, 'name': alias})
        return groups

    # ------------------------------------------------------------------
    # Component access
    # ------------------------------------------------------------------

    def get_component(self, name: str) -> ComponentModel:
        """Returns named ComponentModel or SubChain."""
        return self._get_component(name)

    def component_names(self) -> List[str]:
        return list(self._components.keys())

    def _get_component(self, name: str) -> ComponentModel:
        if name not in self._components:
            raise ValueError(
                f"Component '{name}' not found. Available: {self.component_names()}"
            )
        return self._components[name]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines            = [f"CompositeModel — {len(self._components)} components:"]
        total_params     = 0
        trainable_params = 0
        for alias, comp in self._components.items():
            role     = self._roles.get(alias, '?')
            frozen   = '❄ frozen'   if comp.is_frozen()     else '✓ trainable'
            math     = ' [math]'    if comp.is_mathematical  else ''
            p_count  = comp.param_count()
            t_count  = comp.trainable_param_count()
            total_params     += p_count
            trainable_params += t_count

            if isinstance(comp, SubChain):
                # Top-level SubChain row
                lines.append(
                    f"  {'SubChain':15s} [{role:8s}] {frozen:12s} "
                    f"params: {p_count:>8,}  trainable: {t_count:>8,}{math}"
                )
                # Each member indented 2 extra spaces
                for member in comp.components:
                    mname = member._hp.get('name', type(member).__name__)
                    mfroz = '❄ frozen' if member.is_frozen() else '✓ trainable'
                    lines.append(
                        f"    {mname:13s} [{role:8s}] {mfroz:12s} "
                        f"params: {member.param_count():>8,}  trainable: {member.trainable_param_count():>8,}{math}"
                    )
            else:
                name = comp._hp.get('name', type(comp).__name__)
                lines.append(
                    f"  {name:15s} [{role:8s}] {frozen:12s} "
                    f"params: {p_count:>8,}  trainable: {t_count:>8,}{math}"
                )

        lines.append(f"  {'─'*78}")
        lines.append(
            f"  {'Total':15s} {'':9s} {'':12s} "
            f"params: {total_params:>8,}  trainable: {trainable_params:>8,}"
        )
        return "\n".join(lines)

    def __str__(self):  return self.summary()
    def __repr__(self): return (f"CompositeModel(components={self.component_names()}, device={self._device})")


# ==============================================================================
# ModelFactory
# ==============================================================================

class ModelFactory:
    """
    Single entry point for model creation, saving, and loading.
    NOT nn.Module — pure static factory.

    Methods:
        create(config, device)         — ModelConfig → CompositeModel
        save(model, path, components)  — save named components
        load(model, path, components)  — load named components
        save_backbone(model, path)     — backbone only (after pretrain)
        load_backbone(model, path)     — backbone only (paradigm branch)
        from_checkpoint(path, device)  — full reconstruct from checkpoint

    Checkpoint format (.pt):
        {
            'config':       ModelConfig dict,
            'state_dicts':  {alias: state_dict},
            'frozen_names': [aliases frozen at save time],
            'device':       str(device),
        }

    [PENDING] load() — add device consistency check.
    """

    @staticmethod
    def create(config: ModelConfig,
               device: torch.device = None) -> CompositeModel:
        """
        Create CompositeModel from ModelConfig.
        device passed from notebook — never auto-detected.
        """
        model = CompositeModel(config, device=device)
        if device is not None:
            model = model.to(device)
        return model

    @staticmethod
    def save(model:      CompositeModel,
             path:       str,
             components: Optional[List[str]] = None):
        """
        Save checkpoint.
        Mathematical components (Softmax, PrototypicalNet) skipped — no state.
        components=None saves all non-mathematical components.
        """
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else '.',
            exist_ok=True
        )
        # Default: save all non-mathematical components
        if components is None:
            components = [
                alias for alias in model.component_names()
                if not model.get_component(alias).is_mathematical
            ]
        state_dicts = {
            alias: model.get_component(alias).state_dict()
            for alias in components
        }
        torch.save({
            'config':       model._config.to_dict(),
            'state_dicts':  state_dicts,
            'frozen_names': model.frozen_names(),
            'device':       str(model._device),
        }, path)

    @staticmethod
    def load(model:      CompositeModel,
             path:       str,
             components: Optional[List[str]] = None,
             strict:     bool = True):
        """
        Load checkpoint into existing model.
        components=None loads all saved in checkpoint.
        [PENDING] Add device consistency check.
        """
        checkpoint  = torch.load(path, map_location=model._device or 'cpu')
        state_dicts = checkpoint.get('state_dicts', {})
        comp_names  = components or list(state_dicts.keys())

        for alias in comp_names:
            if alias not in state_dicts:
                raise KeyError(
                    f"'{alias}' not in checkpoint. "
                    f"Available: {list(state_dicts.keys())}"
                )
            model.get_component(alias).load_state_dict(
                state_dicts[alias], strict=strict
            )

        for frozen_alias in checkpoint.get('frozen_names', []):
            if frozen_alias in model.component_names():
                model.freeze(frozen_alias)

    @staticmethod
    def save_backbone(model: CompositeModel, path: str):
        """
        Save backbone only — after pretrain before paradigm branch.
        Saves role='backbone' components only.
        """
        backbone_aliases = [
            alias for alias in model.component_names()
            if model._roles.get(alias) == 'backbone'
        ]
        ModelFactory.save(model, path, components=backbone_aliases)

    @staticmethod
    def load_backbone(model: CompositeModel, path: str, strict: bool = True):
        """Load backbone only — when branching after shared pretrain."""
        checkpoint       = torch.load(path, map_location=model._device or 'cpu')
        backbone_aliases = list(checkpoint.get('state_dicts', {}).keys())
        ModelFactory.load(model, path, components=backbone_aliases, strict=strict)

    @staticmethod
    def from_checkpoint(path: str,
                        device: torch.device = None) -> CompositeModel:
        """
        Reconstruct full model from checkpoint.
        Config saved inside — no prior architecture knowledge needed.
        device from notebook.
        """
        checkpoint = torch.load(path, map_location=device or 'cpu')
        config     = ModelConfig.from_dict(checkpoint['config'])
        model      = ModelFactory.create(config, device=device)
        ModelFactory.load(model, path)
        return model
