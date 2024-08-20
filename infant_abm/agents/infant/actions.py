class Action:
    def __init__(self, metadata=None):
        self.metadata = metadata


class LookForToy(Action):
    number = 1
    pass


class Crawl(Action):
    number = 2


class InteractWithToy(Action):
    number = 3


class EvaluateToy(Action):
    def __init__(self, duration=0, metadata=None):
        super().__init__(metadata)
        self.duration = duration


class EvaluateThrow(Action):
    def __init__(self, duration=0, metadata=None):
        super().__init__(metadata)
        self.duration = duration
