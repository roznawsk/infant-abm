class Action:
    pass


class Crawl(Action):
    pass


class LookForToy(Action):
    pass


class InteractWithToy(Action):
    pass


class EvaluateToy(Action):
    def __init__(self, duration=0):
        self.duration = duration


class EvaluateThrow(Action):
    def __init__(self, duration=0):
        self.duration = duration
