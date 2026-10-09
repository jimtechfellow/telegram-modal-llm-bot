"""ReportModel adapter; reuse the runner's durable, cancellable inference path."""


class InferenceStopped(Exception):
    pass


class LifecycleReportModel:
    def __init__(self, infer):
        self.infer = infer
        self.result = None

    async def generate(self, messages):
        self.result = await self.infer(messages)
        return self.result["text"]
