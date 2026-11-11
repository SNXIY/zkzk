"""DashVector delete must paginate past topk=1024."""

from __future__ import annotations

from typing import Any

from nanobot.knowledge.dashvector_store import DashVectorIndex


class _FakeCollection:
    def __init__(self, ids: list[str]) -> None:
        self.ids = list(ids)
        self.deleted: list[str] = []

    def query(self, *args: Any, **kwargs: Any) -> Any:
        topk = int(kwargs.get("topk") or 1024)
        filt = str(kwargs.get("filter") or "")
        matched = [i for i in self.ids if "doc_id = 'docA'" not in filt or i.startswith("docA_")]
        batch = matched[:topk]

        class _Doc:
            def __init__(self, doc_id: str) -> None:
                self.id = doc_id
                self.fields = {"doc_id": "docA"}

        class _Ret:
            code = 0
            output = [_Doc(i) for i in batch]

            def __bool__(self) -> bool:
                return True

        return _Ret()

    def delete(self, ids: list[str]) -> Any:
        for doc_id in ids:
            if doc_id in self.ids:
                self.ids.remove(doc_id)
            self.deleted.append(doc_id)

        class _Ret:
            code = 0

            def __bool__(self) -> bool:
                return True

        return _Ret()


def test_delete_by_doc_id_paginates_beyond_1024(monkeypatch) -> None:
    index = DashVectorIndex.__new__(DashVectorIndex)
    fake = _FakeCollection([f"docA_{i}" for i in range(2500)])
    index.collection = fake  # type: ignore[attr-defined]

    index.delete_by_doc_id("docA", page_size=1024)

    assert fake.ids == []
    assert len(fake.deleted) == 2500


def test_delete_by_doc_id_uses_known_chunk_ids() -> None:
    index = DashVectorIndex.__new__(DashVectorIndex)
    fake = _FakeCollection(["docA_0", "docA_1", "other"])
    index.collection = fake  # type: ignore[attr-defined]

    index.delete_by_doc_id("docA", known_chunk_ids=["docA_0", "docA_1"])

    assert fake.ids == ["other"]
    assert set(fake.deleted) >= {"docA_0", "docA_1"}
