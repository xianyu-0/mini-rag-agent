class Memory:
    def __init__(self, max_turns: int = 8):
        self.history = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self._compress()

    def _compress(self):
        # 超出窗口就丢最早的，防止上下文爆炸
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def get(self):
        return self.history

    def clear(self):
        self.history = []