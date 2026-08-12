# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.agent_kv.ownership import (
    AgentKVGenerationKey,
    AgentKVOwnershipIndex,
)


def make_owner(session_id: str, generation: int = 1) -> AgentKVGenerationKey:
    return AgentKVGenerationKey("tenant", session_id, "main", generation)


def test_shared_hash_retains_other_owner_when_one_owner_is_removed() -> None:
    index = AgentKVOwnershipIndex()
    first = make_owner("first")
    second = make_owner("second")
    index.add(first, [b"shared", b"first-only"])
    index.add(second, [b"shared"])

    index.remove_owner(first)

    assert index.get_owners(b"shared") == {second}
    assert not index.get_owners(b"first-only")


def test_removing_session_cleans_every_branch_generation() -> None:
    index = AgentKVOwnershipIndex()
    first = make_owner("session", generation=1)
    second = AgentKVGenerationKey("tenant", "session", "branch", 2)
    survivor = make_owner("other")
    index.add(first, [b"shared"])
    index.add(second, [b"second"])
    index.add(survivor, [b"shared"])

    index.remove_session("tenant", "session")

    assert index.get_owners(b"shared") == {survivor}
    assert not index.get_owners(b"second")
