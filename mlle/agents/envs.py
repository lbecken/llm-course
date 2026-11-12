"""
Deterministic task environments for agent experiments (Modules 18-19).

Each has an explicit goal, a finite set of legal actions, and a success test implemented
in code. Nothing here needs a network, a shell or a file system.

    NumberGame   reach a target number with the operations +3, *2, -1 in a limited number of moves
    KeyDoorWorld a tiny text adventure: find the key, unlock the door, reach the exit
"""

from __future__ import annotations

from collections import deque


class NumberGame:
    OPS = {"add3": lambda v: v + 3, "double": lambda v: v * 2, "sub1": lambda v: v - 1}

    def __init__(self, start: int, target: int, max_value: int = 500):
        self.start, self.target, self.max_value = start, target, max_value
        self.goal = f"turn {start} into {target} using add3, double, sub1"
        self.value = start

    def reset(self) -> str:
        self.value = self.start
        return f"value={self.value}"

    def actions(self) -> list[str]:
        return list(self.OPS)

    def step(self, action: str) -> tuple[str, bool, bool]:
        self.value = self.OPS[action](self.value)
        if self.value == self.target:
            return f"value={self.value}", True, True
        if abs(self.value) > self.max_value:
            return f"value={self.value} (out of range)", True, False
        return f"value={self.value}", False, False

    def state_key(self) -> str:
        return str(self.value)

    def shortest_solution(self, max_depth: int = 12) -> list[str] | None:
        """Breadth-first search: the optimal plan, used as a reference and by the planner policy."""
        queue, seen = deque([(self.start, [])]), {self.start}
        while queue:
            value, path = queue.popleft()
            if value == self.target:
                return path
            if len(path) >= max_depth:
                continue
            for name, fn in self.OPS.items():
                nxt = fn(value)
                if nxt not in seen and abs(nxt) <= self.max_value:
                    seen.add(nxt)
                    queue.append((nxt, path + [name]))
        return None


class KeyDoorWorld:
    """Rooms in a line: hall - library - cellar | locked door | vault (exit). The key is in the cellar."""

    ROOMS = ["hall", "library", "cellar", "vault"]

    def __init__(self):
        self.goal = "reach the vault (the door to it is locked; a key is somewhere)"
        self.reset()

    def reset(self) -> str:
        self.room, self.has_key, self.door_open = 0, False, False
        return self._describe()

    def _describe(self) -> str:
        here = self.ROOMS[self.room]
        extras = []
        if here == "cellar" and not self.has_key:
            extras.append("a key lies here")
        if here == "cellar":
            extras.append("a door leads east; it is " + ("open" if self.door_open else "locked"))
        return f"you are in the {here}" + ("; " + "; ".join(extras) if extras else "") + ("; you carry a key" if self.has_key else "")

    def actions(self) -> list[str]:
        return ["go east", "go west", "take key", "unlock door"]

    def step(self, action: str) -> tuple[str, bool, bool]:
        here = self.ROOMS[self.room]
        note = ""
        if action == "go east":
            if here == "cellar" and not self.door_open:
                note = "the door is locked. "
            elif self.room < 3:
                self.room += 1
        elif action == "go west":
            if self.room > 0:
                self.room -= 1
            else:
                note = "there is a wall. "
        elif action == "take key":
            if here == "cellar" and not self.has_key:
                self.has_key = True
            else:
                note = "there is no key here. "
        elif action == "unlock door":
            if here == "cellar" and self.has_key:
                self.door_open = True
            else:
                note = "you cannot do that here. "
        success = self.ROOMS[self.room] == "vault"
        return note + self._describe(), success, success

    def state_key(self) -> str:
        return f"{self.room}|{self.has_key}|{self.door_open}"
