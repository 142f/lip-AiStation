import os


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def checkpoint_root() -> str:
    return os.path.abspath(
        os.environ.get("LIPFD_CHECKPOINT_ROOT", os.path.join(project_root(), "checkpoints"))
    )


def pretrained_root() -> str:
    return os.path.abspath(
        os.environ.get("LIPFD_PRETRAINED_ROOT", os.path.join(checkpoint_root(), "pretrained"))
    )


def cache_root() -> str:
    return os.path.abspath(
        os.environ.get("LIPFD_CACHE_ROOT", os.path.join(pretrained_root(), "cache"))
    )


def huggingface_root() -> str:
    return os.path.abspath(
        os.environ.get("LIPFD_HUGGINGFACE_ROOT", os.path.join(pretrained_root(), "huggingface"))
    )


def clip_root() -> str:
    return os.path.abspath(
        os.environ.get("LIPFD_CLIP_ROOT", os.path.join(pretrained_root(), "clip"))
    )


def torch_checkpoint_dir() -> str:
    return os.path.abspath(
        os.environ.get(
            "LIPFD_TORCH_CHECKPOINT_DIR",
            os.path.join(pretrained_root(), "torch", "hub", "checkpoints"),
        )
    )


def insightface_root() -> str:
    return os.path.abspath(
        os.environ.get("LIPFD_INSIGHTFACE_ROOT", os.path.join(pretrained_root(), "insightface"))
    )


def dfn_pretrained() -> str:
    return os.environ.get("LIPFD_DFN_PRETRAINED", "dfn2b")


def configure_runtime_cache_env() -> None:
    os.environ.setdefault("LIPFD_CHECKPOINT_ROOT", checkpoint_root())
    os.environ.setdefault("LIPFD_PRETRAINED_ROOT", pretrained_root())
    os.environ.setdefault("LIPFD_CLIP_ROOT", clip_root())
    os.environ.setdefault("LIPFD_TORCH_CHECKPOINT_DIR", torch_checkpoint_dir())
    os.environ.setdefault("LIPFD_INSIGHTFACE_ROOT", insightface_root())
    os.environ.setdefault("TORCH_HOME", os.path.join(pretrained_root(), "torch"))
    os.environ.setdefault("HF_HOME", huggingface_root())
    os.environ.setdefault("XDG_CACHE_HOME", cache_root())
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


configure_runtime_cache_env()
