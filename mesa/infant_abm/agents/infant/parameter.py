class Parameter:
    def __init__(self, exploitation):
        assert exploitation >= 0.0 and exploitation <= 1.0

        self.e1_base = 1.0 - exploitation
        self.e2_base = exploitation

        self.reset()

    def reset(self):
        self.e1 = self.e1_base
        self.e2 = self.e2_base

    def boost(self, value):
        self.e2 = min(1.0, self.e2 + value)
        self.e1 = 1 - self.e2
