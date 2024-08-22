class InfantAction:
    def __init__(self, metadata=None):
        self.metadata = metadata


class Crawl(InfantAction):
    pass


class LookForToy(InfantAction):
    pass


class InteractWithToy(InfantAction):
    pass


class EvaluateToy(InfantAction):
    def __init__(self, duration=0, metadata=None):
        super().__init__(metadata)
        self.duration = duration


class EvaluateThrow(InfantAction):
    def __init__(self, duration=0, metadata=None):
        super().__init__(metadata)
        self.duration = duration
