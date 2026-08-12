# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Content-addressed ownership tracking for AgentKV cache entries."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentKVGenerationKey:
    """Stable identity for one branch generation."""

    namespace: str
    session_id: str
    branch_id: str
    generation: int


class AgentKVOwnershipIndex:
    """Maintains the many-to-many relation between hashes and generations."""

    def __init__(self) -> None:
        self._owners_by_hash: dict[bytes, set[AgentKVGenerationKey]] = defaultdict(set)
        self._hashes_by_owner: dict[AgentKVGenerationKey, set[bytes]] = defaultdict(set)

    def add(
        self,
        owner: AgentKVGenerationKey,
        block_hashes: Iterable[bytes],
    ) -> None:
        owner_hashes = self._hashes_by_owner[owner]
        for block_hash in block_hashes:
            content_hash = bytes(block_hash)
            if content_hash in owner_hashes:
                continue
            owner_hashes.add(content_hash)
            self._owners_by_hash[content_hash].add(owner)

    def remove_owner(self, owner: AgentKVGenerationKey) -> None:
        for block_hash in self._hashes_by_owner.pop(owner, ()):
            owners = self._owners_by_hash[block_hash]
            owners.discard(owner)
            if not owners:
                del self._owners_by_hash[block_hash]

    def remove_session(self, namespace: str, session_id: str) -> None:
        stale_owners = [
            owner
            for owner in self._hashes_by_owner
            if owner.namespace == namespace and owner.session_id == session_id
        ]
        for owner in stale_owners:
            self.remove_owner(owner)

    def get_owners(self, block_hash: bytes) -> frozenset[AgentKVGenerationKey]:
        return frozenset(self._owners_by_hash.get(bytes(block_hash), ()))

    def get_hashes(self, owner: AgentKVGenerationKey) -> frozenset[bytes]:
        return frozenset(self._hashes_by_owner.get(owner, ()))

    def __bool__(self) -> bool:
        return bool(self._owners_by_hash)
