from copy import deepcopy


class ConversationManager:
    MODES = ("rag", "baseline")

    def __init__(self, max_turns: int = 3):
        if max_turns < 0:
            raise ValueError("A max_turns nem lehet negatív.")

        self.max_turns = max_turns
        self.histories = {
            "rag": [],
            "baseline": []
        }

    def _check_mode(self, mode: str):
        if mode not in self.MODES:
            raise ValueError(f"Ismeretlen mód: {mode}")

    def get_history(self, mode: str) -> list[dict]:
        self._check_mode(mode)
        return deepcopy(self.histories[mode])

    def get_context(self, mode: str) -> list[dict]:
        self._check_mode(mode)

        if self.max_turns == 0:
            return []

        history = self.histories[mode]
        return deepcopy(history[-2 * self.max_turns:])

    def add_turn(self, mode: str, question: str, answer: str):
        self._check_mode(mode)

        self.histories[mode].append({
            "role": "user",
            "content": question
        })
        self.histories[mode].append({
            "role": "assistant",
            "content": answer
        })

    def clear(self, mode: str):
        self._check_mode(mode)
        self.histories[mode] = []
