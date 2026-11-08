"""Aborting contracts for coordinate, scale, split, and promotion claims.

These functions are the API the property harness drives. A green synthetic
run does not promote a recipe and does not certify leaderboard behaviour.
"""

from umud.verification.contracts import (
    ContractAbort,
    assert_groups_disjoint,
    assert_promotable,
    assert_scale_not_relabeled,
    physical_cosine,
    pullback_length_sq,
)

__all__ = [
    "ContractAbort",
    "assert_groups_disjoint",
    "assert_promotable",
    "assert_scale_not_relabeled",
    "physical_cosine",
    "pullback_length_sq",
]
