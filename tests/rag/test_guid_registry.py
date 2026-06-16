import json
import re

import pytest

from src.rag.guid_registry import GuidRegistry

UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

URL_A = "https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks"
URL_B = "https://www.investor.gov/introduction-investing/investing-basics/investment-products/bonds"


@pytest.fixture
def registry(tmp_path):
    return GuidRegistry(base_path=tmp_path)


def test_get_or_create_guid_returns_string(registry):
    assert isinstance(registry.get_or_create_guid(URL_A), str)


def test_get_or_create_guid_returns_valid_uuid4_format(registry):
    guid = registry.get_or_create_guid(URL_A)
    assert UUID4_RE.match(guid), f"Expected UUID v4, got: {guid}"


def test_same_url_same_instance_returns_same_guid(registry):
    guid1 = registry.get_or_create_guid(URL_A)
    guid2 = registry.get_or_create_guid(URL_A)
    assert guid1 == guid2


def test_same_url_across_instances_returns_same_guid(tmp_path):
    r1 = GuidRegistry(base_path=tmp_path)
    guid1 = r1.get_or_create_guid(URL_A)
    r2 = GuidRegistry(base_path=tmp_path)
    guid2 = r2.get_or_create_guid(URL_A)
    assert guid1 == guid2


def test_different_urls_return_different_guids(registry):
    guid_a = registry.get_or_create_guid(URL_A)
    guid_b = registry.get_or_create_guid(URL_B)
    assert guid_a != guid_b


def test_registry_file_not_created_before_first_call(tmp_path):
    GuidRegistry(base_path=tmp_path)
    assert not (tmp_path / "url_guid_registry.json").exists()


def test_registry_file_created_on_first_call(registry, tmp_path):
    registry.get_or_create_guid(URL_A)
    assert (tmp_path / "url_guid_registry.json").exists()


def test_registry_file_contains_url_as_key(registry, tmp_path):
    registry.get_or_create_guid(URL_A)
    data = json.loads((tmp_path / "url_guid_registry.json").read_text(encoding="utf-8"))
    assert URL_A in data


def test_registry_file_value_is_valid_uuid4(registry, tmp_path):
    registry.get_or_create_guid(URL_A)
    data = json.loads((tmp_path / "url_guid_registry.json").read_text(encoding="utf-8"))
    assert UUID4_RE.match(data[URL_A])


def test_guid_persists_after_reload(tmp_path):
    r1 = GuidRegistry(base_path=tmp_path)
    guid_written = r1.get_or_create_guid(URL_A)
    r2 = GuidRegistry(base_path=tmp_path)
    guid_read = r2.get_or_create_guid(URL_A)
    assert guid_written == guid_read


def test_multiple_urls_all_stored(registry, tmp_path):
    registry.get_or_create_guid(URL_A)
    registry.get_or_create_guid(URL_B)
    data = json.loads((tmp_path / "url_guid_registry.json").read_text(encoding="utf-8"))
    assert URL_A in data
    assert URL_B in data


def test_two_registries_different_base_paths_are_isolated(tmp_path):
    path1 = tmp_path / "run1"
    path2 = tmp_path / "run2"
    path1.mkdir()
    path2.mkdir()
    r1 = GuidRegistry(base_path=path1)
    r2 = GuidRegistry(base_path=path2)
    guid1 = r1.get_or_create_guid(URL_A)
    guid2 = r2.get_or_create_guid(URL_A)
    # Both are valid UUIDs but may differ (independent registries)
    assert UUID4_RE.match(guid1)
    assert UUID4_RE.match(guid2)
    assert (path1 / "url_guid_registry.json").exists()
    assert (path2 / "url_guid_registry.json").exists()
