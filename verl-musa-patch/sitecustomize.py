"""Bootstrap the internal MUSA Migration before VERL is imported.

SGLang scheduler processes do not import ``verl_hardware_plugin``.  The import
hook below applies the SGLang IPC fix when the scheduler imports
``model_runner``; the regular Migration bootstrap remains shared by all
processes.
"""

import importlib.abc
import importlib.machinery
import os
import sys


def _log(message: str) -> None:
    """Emit a small, Ray-friendly bootstrap diagnostic."""
    print(f"[VERL_MUSA_SITE pid={os.getpid()}] {message}", flush=True)


def _has_megatron_runtime() -> bool:
    """Return whether this interpreter has Megatron/MCore available.

    ``sitecustomize`` is also inherited by SGLang and other helper processes.
    The external ``musa_patch`` package is a Megatron patch and must not be
    imported in a process that has no Megatron runtime.  Use ``PathFinder`` so
    this probe does not execute our own import hooks or import the package.
    """
    for module_name in ("megatron", "megatron.core", "mcore"):
        try:
            if importlib.machinery.PathFinder.find_spec(module_name, sys.path) is not None:
                return True
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
            # A partially initialized parent package can make find_spec raise;
            # that still means this process cannot safely use the base patch.
            continue
    return False


def _load_base_musa_patch() -> None:
    """Load the external base patch once, after a real workload starts."""
    migration_root = os.getenv("VERL_MUSA_PATCH", "").strip()
    if migration_root and migration_root not in sys.path:
        sys.path.insert(0, migration_root)

    # ``megatron`` may be installed globally even for an FSDP worker (for
    # example through another model package).  That is not sufficient reason
    # to load the external Megatron patch: FSDP has no MUSA_PATCH_PATH and
    # should leave the global Megatron installation untouched.  Megatron
    # launchers opt in by explicitly setting MUSA_PATCH_PATH.
    megatron_patch = os.getenv("MUSA_PATCH_PATH", "").strip()
    if not megatron_patch:
        # SGLang-only processes can inherit VERL_PLATFORM/PYTHONPATH but do
        # not have Megatron.  Do not import musa_patch there: it can install
        # Megatron/TE compatibility hooks and interfere with SGLang's native
        # MUSA capability detection.
        _log("skip megatron-lm-musa-patch: MUSA_PATCH_PATH is not set")
        return

    if not _has_megatron_runtime():
        _log(
            "skip megatron-lm-musa-patch: MUSA_PATCH_PATH is set but "
            "megatron/mcore is not importable"
        )
        return

    _log(f"loading megatron-lm-musa-patch from {megatron_patch}")
    if megatron_patch and megatron_patch not in sys.path:
        sys.path.insert(0, megatron_patch)

    # The base patch owns the general MUSA/Megatron integration.  The VERL
    # migration owns the VERL-specific hooks and its optional MATE wrapper.
    if "musa_patch" not in sys.modules:
        flash_env = "SLIME_PATCH_MUSA_FLASH_ATTN_MATE"
        old_flash_env = os.environ.get(flash_env)
        os.environ[flash_env] = "0"
        try:
            importlib.import_module("musa_patch")
            _log("loaded megatron-lm-musa-patch: musa_patch")
        finally:
            if old_flash_env is None:
                os.environ.pop(flash_env, None)
            else:
                os.environ[flash_env] = old_flash_env

    return


def _bootstrap_musa_migration() -> None:
    """Load the internal MUSA runtime before VERL or Megatron is imported."""
    if os.getenv("VERL_PLATFORM", "").strip().lower() != "musa":
        return

    _log(
        "starting MUSA bootstrap: "
        f"VERL_MUSA_PATCH={os.getenv('VERL_MUSA_PATCH', '')!r}, "
        f"MUSA_PATCH_PATH={os.getenv('MUSA_PATCH_PATH', '')!r}"
    )
    _load_base_musa_patch()
    migration = importlib.import_module("verl_musa_patch")
    migration.apply()
    _log("applied VERL MUSA patches: verl_musa_patch.apply()")


_BOOTSTRAP_STARTED = False


class _MusaMigrationFinder(importlib.abc.MetaPathFinder):
    """Bootstrap only when a workload imports VERL/Megatron.

    Ray runtime-env setup and other Python helpers inherit VERL_PLATFORM but
    must not import Transformer Engine just because ``sitecustomize`` exists.
    """

    # SGLang may import Transformer Engine for its own kernels.  Loading the
    # Megatron compatibility patch on that import would replace SGLang's real
    # MUSA capability (3, 1) with the CUDA-compatibility value (8, 3).
    _TARGETS = {"verl", "megatron"}

    def find_spec(self, fullname, path=None, target=None):
        global _BOOTSTRAP_STARTED
        if fullname not in self._TARGETS or _BOOTSTRAP_STARTED:
            return None
        _BOOTSTRAP_STARTED = True
        _bootstrap_musa_migration()
        return None


if os.getenv("VERL_PLATFORM", "").strip().lower() == "musa" and not any(
    isinstance(finder, _MusaMigrationFinder) for finder in sys.meta_path
):
    sys.meta_path.insert(0, _MusaMigrationFinder())


_TARGET = "sglang.srt.model_executor.model_runner"


class _MusaModelRunnerLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module):
        self._loader.exec_module(module)
        from verl_musa_patch.sglang import (
            _install_mtp_ipc_tensor_cache,
            _restore_musa_device_capability,
        )

        # The base Megatron compatibility patch reports a synthetic CUDA
        # capability.  SGLang's MUSA backend expects the real MP capability.
        _restore_musa_device_capability()
        _install_mtp_ipc_tensor_cache()


class _MusaModelRunnerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        # Bypass this finder while resolving the original module spec.
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _MusaModelRunnerLoader(spec.loader)
        return spec


if os.getenv("VERL_PLATFORM") == "musa" and not any(
    isinstance(finder, _MusaModelRunnerFinder) for finder in sys.meta_path
):
    sys.meta_path.insert(0, _MusaModelRunnerFinder())
