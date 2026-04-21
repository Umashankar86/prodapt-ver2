from .agent_service import AgentService
class AgentRunner:
    __init__ = lambda self, settings, llm_client=None: setattr(self, "service", AgentService(settings, llm_client=llm_client))
    run = lambda self, question: self.service.run(question)
