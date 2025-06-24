"""
Utilities for working with .mpt (Model Package Tool) files
"""

import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from . import pt


@dataclass
class PolicyMetadata:
    """Typed metadata for model policies"""

    model_name: str
    version: str
    description: str
    created_at: str
    author: str
    tags: List[str]
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]


@dataclass
class TrainerState:
    """Typed trainer state information"""

    epoch: int
    global_step: int
    learning_rate: float
    loss: float
    optimizer_state: Optional[str] = None  # Path to optimizer state if saved
    scheduler_state: Optional[str] = None  # Path to scheduler state if saved


class MPTFile:
    """Handler for .mpt model package files"""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._temp_dir = None
        self._extracted = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Clean up temporary files"""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
            self._extracted = False

    def _ensure_extracted(self):
        """Ensure the archive is extracted to a temporary directory"""
        if not self._extracted:
            self._temp_dir = Path(tempfile.mkdtemp())
            with zipfile.ZipFile(self.path, "r") as zf:
                zf.extractall(self._temp_dir)
            self._extracted = True

    @property
    def model(self) -> torch.nn.Module:
        """Load the PyTorch model from the package"""
        self._ensure_extracted()
        model_path = self._temp_dir / "model.pt"
        return pt.load(model_path)

    @property
    def metadata(self) -> PolicyMetadata:
        """Get the metadata from the package"""
        self._ensure_extracted()
        metadata_path = self._temp_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            data = json.load(f)
        return PolicyMetadata(**data)

    @property
    def trainer_state(self) -> TrainerState:
        """Get the trainer state from the package"""
        self._ensure_extracted()
        state_path = self._temp_dir / "trainer_state.json"
        with open(state_path, "r") as f:
            data = json.load(f)
        return TrainerState(**data)

    @property
    def codebase_path(self) -> Path:
        """Get path to the extracted codebase directory"""
        self._ensure_extracted()
        return self._temp_dir / "codebase"

    @property
    def manifest(self) -> Dict[str, Any]:
        """Get the manifest information"""
        self._ensure_extracted()
        manifest_path = self._temp_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            return json.load(f)


def create_package(
    model: torch.nn.Module,
    metadata: PolicyMetadata,
    trainer_state: TrainerState,
    codebase_path: Union[str, Path],
    output_path: Union[str, Path],
    create_sidecar: bool = True,
) -> None:
    """Create an .mpt package file"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save model
        model_path = temp_path / "model.pt"
        pt.save_model(model, model_path)

        # Save metadata
        metadata_path = temp_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        # Save trainer state
        trainer_path = temp_path / "trainer_state.json"
        with open(trainer_path, "w") as f:
            json.dump(asdict(trainer_state), f, indent=2)

        # Copy codebase
        codebase_dest = temp_path / "codebase"
        shutil.copytree(codebase_path, codebase_dest)

        # Create manifest
        manifest = {
            "format_version": "1.0",
            "created_at": datetime.now().isoformat(),
            "files": {
                "model.pt": model_path.stat().st_size,
                "metadata.json": metadata_path.stat().st_size,
                "trainer_state.json": trainer_path.stat().st_size,
            },
        }

        manifest_path = temp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Create ZIP archive
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in temp_path.rglob("*"):
                if file_path.is_file():
                    arc_path = file_path.relative_to(temp_path)
                    zf.write(file_path, arc_path)

        # Create sidecar metadata file
        if create_sidecar:
            sidecar_path = output_path.with_suffix(".json")
            with open(sidecar_path, "w") as f:
                json.dump(asdict(metadata), f, indent=2)


def load_package(path: Union[str, Path]) -> MPTFile:
    """Load an .mpt package file"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Package not found: {path}")

    return MPTFile(path)


def get_package_info(path: Union[str, Path]) -> Dict[str, Any]:
    """Get basic information about an .mpt package without fully loading it"""
    path = Path(path)

    # Try to load from sidecar first
    sidecar_path = path.with_suffix(".json")
    if sidecar_path.exists():
        with open(sidecar_path, "r") as f:
            metadata = json.load(f)
    else:
        # Extract metadata from archive
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("metadata.json") as f:
                metadata = json.load(f)

    return {
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        "metadata": metadata,
        "has_sidecar": sidecar_path.exists(),
    }


def list_package_contents(path: Union[str, Path]) -> List[str]:
    """List the contents of an .mpt package"""
    with zipfile.ZipFile(path, "r") as zf:
        return zf.namelist()
