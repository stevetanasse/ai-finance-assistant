import json
import uuid
from pathlib import Path


class GuidRegistry:
    def __init__(self, base_path: Path | str):
        self.registry_file = Path(base_path) / "url_guid_registry.json"

    def get_or_create_guid(self, url: str) -> str:
        data = self._load()
        if url not in data:
            data[url] = str(uuid.uuid4())
            self._save(data)
        return data[url]

    def _load(self) -> dict:
        if not self.registry_file.exists():
            return {}
        return json.loads(self.registry_file.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.registry_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
