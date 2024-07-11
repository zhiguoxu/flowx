import time
from abc import ABC, abstractmethod
from typing import Sequence, List, Dict

from pydantic import BaseModel, Field


class IndexManager(ABC):
    @abstractmethod
    def exists(self, keys: Sequence[str]) -> List[bool]:
        ...

    @abstractmethod
    def get_time(self) -> float:
        """
        Get the current server time as a high-resolution timestamp.
        Ensures a monotonic clock to prevent data loss during cleanup.
        Returns: Current server time as a float timestamp.
        """

    @abstractmethod
    def update(self,
               keys: Sequence[str],
               time_at_least: float,
               *,
               source_ids: Sequence[str | None] | None = None) -> None:
        """Upsert records into the database.

        Args:
            keys: List of record keys to upsert.
            time_at_least: timestamp to ensure the system's timestamp
                is at least this value, preventing time-drift issues.
            source_ids: Optional list of source IDs for the keys.
        Raises:
            ValueError: If lengths of keys and source_ids don't match.
        """

    @abstractmethod
    def list_keys(self,
                  *,
                  before: float | None = None,
                  after: float | None = None,
                  source_ids: Sequence[str] | None = None,
                  limit: int | None = None
                  ) -> List[str]:
        """List records in the database based on the provided filters.

        Args:
            before: Filter to list records updated before this time.
            after: Filter to list records updated after this time.
            source_ids: Filter to list records with specific group IDs.
            limit: optional limit on the number of records to return.

        Returns:
            A list of keys for the matching records.
        """

    @abstractmethod
    def delete(self, keys: Sequence[str]) -> None:
        ...

    @abstractmethod
    def delete_all(self) -> None:
        ...


class MemoryIndexManger(BaseModel, IndexManager):
    class Record(BaseModel):
        source_id: str | None = None
        updated_time: float

    records: Dict[str, Record] = Field(default_factory=dict)

    def exists(self, keys: Sequence[str]) -> List[bool]:
        return [key in self.records for key in keys]

    def get_time(self) -> float:
        return time.time()

    def update(self,
               keys: Sequence[str],
               time_at_least: float,
               *,
               source_ids: Sequence[str | None] | None = None) -> None:
        if source_ids and len(keys) != len(source_ids):
            raise ValueError("Length of keys must match length of group_ids")

        for index, key in enumerate(keys):
            source_id = source_ids[index] if source_ids else None
            if time_at_least > self.get_time():
                raise ValueError("time_at_least must be in the past")
            self.records[key] = self.Record(source_id=source_id, updated_time=self.get_time())

    def list_keys(self,
                  *,
                  before: float | None = None,
                  after: float | None = None,
                  source_ids: Sequence[str] | None = None,
                  limit: int | None = None
                  ) -> List[str]:
        result = []
        for key, data in self.records.items():
            if before and data.updated_time >= before:
                continue
            if after and data.updated_time <= after:
                continue
            if source_ids and data.source_id not in source_ids:
                continue
            result.append(key)
        if limit:
            return result[:limit]
        return result

    def delete(self, keys: Sequence[str]) -> None:
        for key in keys:
            if key in self.records:
                del self.records[key]

    def delete_all(self) -> None:
        self.records.clear()
