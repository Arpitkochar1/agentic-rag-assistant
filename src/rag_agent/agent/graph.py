from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rag_agent.agent.nodes import AgentNodes
from rag_agent.agent.state import AgentState


def build_agent_graph(nodes: AgentNodes):
    """
    START -> guard_input -(blocked)-> END
                         -> route_query -(direct)-> generate_answer
                                        -(tool)--> use_tool -> generate_answer
    generate_answer -> guard_output -> add_citations -> END
    (Node names must differ from state keys, hence the verb_noun names.)
    """
    g = StateGraph(AgentState)
    g.add_node("guard_input", nodes.guard_input)
    g.add_node("route_query", nodes.route_query)
    g.add_node("use_tool", nodes.use_tool)
    g.add_node("generate_answer", nodes.generate_answer)
    g.add_node("guard_output", nodes.guard_output)
    g.add_node("add_citations", nodes.add_citations)

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", nodes.after_input, {"route_query": "route_query", "end": END})
    g.add_conditional_edges("route_query", nodes.after_route, {"use_tool": "use_tool", "generate_answer": "generate_answer"})
    g.add_edge("use_tool", "generate_answer")
    g.add_edge("generate_answer", "guard_output")
    g.add_edge("guard_output", "add_citations")
    g.add_edge("add_citations", END)
    return g.compile()
