from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent_state import IndicatorAgentState
from decision_agent import create_final_trade_decider
from graph_util import TechnicalTools
from indicator_agent import create_indicator_agent
from pattern_agent import create_pattern_agent
from trend_agent import create_trend_agent


class SetGraph:
    def __init__(
        self,
        agent_llm: ChatOpenAI,
        graph_llm: ChatOpenAI,
        toolkit: TechnicalTools,
    ):
        self.agent_llm = agent_llm
        self.graph_llm = graph_llm
        self.toolkit = toolkit

    def set_graph(self):
        graph = StateGraph(IndicatorAgentState)

        # --- Nodes ---
        graph.add_node("Indicator Agent", create_indicator_agent(self.graph_llm, self.toolkit))
        graph.add_node("Pattern Agent", create_pattern_agent(self.agent_llm, self.graph_llm, self.toolkit))
        graph.add_node("Trend Agent", create_trend_agent(self.agent_llm, self.graph_llm, self.toolkit))
        graph.add_node("Decision Maker", create_final_trade_decider(self.graph_llm))

        # --- Fan-out: all three analysis agents run in parallel ---
        graph.add_edge(START, "Indicator Agent")
        graph.add_edge(START, "Pattern Agent")
        graph.add_edge(START, "Trend Agent")

        # --- Fan-in: Decision Maker waits for all three to finish ---
        graph.add_edge("Indicator Agent", "Decision Maker")
        graph.add_edge("Pattern Agent", "Decision Maker")
        graph.add_edge("Trend Agent", "Decision Maker")

        graph.add_edge("Decision Maker", END)

        return graph.compile()
