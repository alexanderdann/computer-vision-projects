"""Implements algorithms related to union find."""

from pprint import pprint


class NaiveUnionFind:
    """Naive Union Find."""

    def __init__(self, num_elements: int, *, verbose: bool = False) -> None:
        """C'tor of NaiveUnionFind.

        Args:
            num_elements: count of components/nodes.
            verbose: whether to use the print statements

        """
        self._lookup = list(range(num_elements))
        self._verbose = verbose
        if self._verbose:
            pprint(f"Starting point is the following {self._lookup}")

    def connected(self, id1: int, id2: int) -> bool:
        """Check if two components are connected.

        Args:
            id1: if of the first node
            id2: id of the second node

        Returns:
            True if the two components are connected and false otherwise.

        """
        return self._lookup[id1] == self._lookup[id2]

    def union(self, id1: int, id2: int) -> None:
        """Merge two nodes at given indeces.

        Args:
            id1: if of the first node
            id2: id of the second node

        """
        entry1 = self._lookup[id1]
        entry2 = self._lookup[id2]

        if self._verbose:
            pprint(f"Current state: {self._lookup} with inputs id1={id1} & id2={id2}")
        for idx, elem in enumerate(self._lookup):
            if elem == entry1:
                if self._verbose:
                    pprint(f"Updating {idx} -> {entry2}")
                self._lookup[idx] = entry2
