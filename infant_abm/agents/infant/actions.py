class Action:
    def __init__(self, metadata=None):
        self.metadata = metadata


class Crawl(Action):
    pass


class LookForToy(Action):
    pass


class InteractWithToy(Action):
    pass


class EvaluateToy(Action):
    def __init__(self, duration=0, metadata=None):
        super().__init__(metadata)
        self.duration = duration


class EvaluateThrow(Action):
    def __init__(self, duration=0, metadata=None):
        super().__init__(metadata)
        self.duration = duration
