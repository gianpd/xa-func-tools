import json
import os
import re
import uuid
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # YAML is optional; if missing we fallback to JSON for config persistence


class ExperimentTracker:
    """
    Unified experiment tracker for agentic workflow runs.

    Responsibilities:
    - Generate unique run IDs and directory scaffolding
    - Persist run manifest with config, inputs, and environment metadata
    - Provide component-scoped logging and artifact writing
    - Ensure reproducibility by snapshotting config and inputs
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        code_version: Optional[str] = None,
    ):
        """
        Initialize the tracker. Call start_run() to create directories and manifest.

        Args:
            base_dir: Root logs path where runs are stored (e.g., logs/runs)
            run_id: Optional externally provided run id. If None, one is generated.
            config: Optional configuration dictionary to snapshot
            code_version: Optional version string (e.g., git SHA)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or self._generate_run_id()
        self.run_dir = self.base_dir / self.run_id

        self.config = config or {}
        self.code_version = code_version or self._detect_git_version()

        # Internal state
        self._started = False
        self._manifest_path = self.run_dir / "manifest.json"
        self._config_path = self.run_dir / "config.yaml"  # prefer yaml if available
        self._metadata_path = self.run_dir / "metadata.json"

        # Common component directories used by agents
        self._known_components = {"executor", "function_caller"}

    @staticmethod
    def load_yaml_config(config_path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            config_path: Path to YAML file

        Returns:
            Dict with configuration
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if yaml is None and path.suffix.lower() in {".yaml", ".yml"}:
            raise RuntimeError(
                "PyYAML is not installed but a YAML config was provided. "
                "Install pyyaml or provide a JSON config."
            )

        if yaml is not None and path.suffix.lower() in {".yaml", ".yml"}:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

        # Fallback to JSON if not YAML
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def start_run(self, input_text: Optional[str] = None) -> None:
        """
        Create run directory and write initial manifest/config.

        Args:
            input_text: The original problem/task input to snapshot
        """
        if self._started:
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create component directories
        for comp in self._known_components:
            (self.run_dir / comp).mkdir(parents=True, exist_ok=True)

        # Persist config
        self._persist_config()

        # Initial manifest
        manifest = {
            "run_id": self.run_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "code_version": self.code_version,
            "input_prompt": input_text or "",
            "run_directory": str(self.run_dir),
            "components": sorted(list(self._known_components)),
            "status": "running",
        }
        self._write_json(self._manifest_path, manifest)

        # Basic metadata (can be expanded later)
        metadata = {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "env": {k: v for k, v in os.environ.items() if k in ("HOSTNAME", "USER", "SHELL")},  # limited env snapshot
        }
        self._write_json(self._metadata_path, metadata)

        self._started = True

    def finalize_run(self, status: str = "completed", extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark run as finished and update manifest.

        Args:
            status: One of 'completed', 'failed', 'aborted'
            extra: Any extra metadata to include
        """
        if not self._started:
            return
        manifest = self._read_json(self._manifest_path) or {}
        manifest.update(
            {
                "end_time": datetime.now(timezone.utc).isoformat(),
                "status": status,
            }
        )
        if extra:
            manifest.setdefault("final_metadata", {}).update(extra)
        self._write_json(self._manifest_path, manifest)

    def component_dir(self, component: str) -> Path:
        """
        Get or create a directory for a specific component.

        Args:
            component: Component name (e.g., 'solver')

        Returns:
            Path to the component directory
        """
        safe_name = self._sanitize_component(component)
        comp_dir = self.run_dir / safe_name
        comp_dir.mkdir(parents=True, exist_ok=True)
        return comp_dir

    def log_event(self, component: str, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Append a structured event to component's events.jsonl

        Args:
            component: Component name
            event_type: Event type label (e.g., 'draft_generated')
            data: Arbitrary JSON-serializable dict
        """
        comp_dir = self.component_dir(component)
        events_file = comp_dir / "events.jsonl"
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "data": data or {},
        }
        with open(events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def write_artifact(
        self,
        component: str,
        filename: str,
        content: Union[str, Dict[str, Any], list, bytes],
        overwrite: bool = True,
    ) -> Path:
        """
        Write an artifact file under component directory.
        
        Args:
            component: Component name
            filename: File name (e.g., 'output.json', 'prompt.txt')
            content: Content to write (string/dict/list/bytes)
            overwrite: If True, replace existing file. If False, append for text files.
                      For structured data (JSON/YAML), always overwrites.
        
        Returns:
            Path to the written artifact
        
        Raises:
            FileExistsError: If binary file exists and overwrite=False
        """
        comp_dir = self.component_dir(component)
        target = comp_dir / filename
        
        # Handle structured data (JSON/YAML) - always overwrite for consistency
        if isinstance(content, (dict, list)):
            # Choose JSON or YAML based on extension
            if filename.lower().endswith((".yaml", ".yml")) and yaml is not None:
                with open(target, "w", encoding="utf-8") as f:
                    yaml.safe_dump(content, f, sort_keys=False)  # type: ignore
            else:
                with open(target, "w", encoding="utf-8") as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)
            return target
        
        # Handle binary content
        if isinstance(content, bytes):
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"Binary artifact already exists and overwrite=False: {target}"
                )
            with open(target, "wb") as f:
                f.write(content)
            return target
        
        # Handle text content with smart mode selection
        if overwrite:
            # Explicit overwrite requested
            mode = "w"
        else:
            # Use 'w' for new files, 'a' for existing files
            mode = "a" if target.exists() else "w"
        
        with open(target, mode, encoding="utf-8") as f:
            f.write(str(content))

    def llm_log_dir(self, component: str) -> Path:
        """
        Provide a directory intended for LLM raw logs for a given component.

        This can be passed to existing logging wrappers (e.g., enable_llm_logging).

        Args:
            component: Component name

        Returns:
            Path to directory under component 'llm' subfolder
        """
        comp_dir = self.component_dir(component)
        llm_dir = comp_dir / "llm"
        llm_dir.mkdir(parents=True, exist_ok=True)
        return llm_dir

    # -------------------------
    # Internal helpers
    # -------------------------

    def _persist_config(self) -> None:
        if not self.config:
            # Ensure an empty file still exists for traceability
            with open(self._config_path, "w", encoding="utf-8") as f:
                f.write("# No config provided\n")
            return

        # If YAML is available, persist original as YAML
        if yaml is not None:
            with open(self._config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.config, f, sort_keys=False)  # type: ignore
        else:
            # YAML not available, fallback to JSON but keep .yaml extension to match spec
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _generate_run_id(self) -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        rand = uuid.uuid4().hex[:6]
        return f"run-{ts}-{rand}"

    def _detect_git_version(self) -> str:
        try:
            sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            return sha.decode("utf-8").strip()
        except Exception:
            return "unknown"

    def _sanitize_component(self, name: str) -> str:
        safe = name.strip().lower()
        safe = re.sub(r"[^a-z0-9_\-/]", "_", safe)
        return safe

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)