import logging
from langgraph.graph import StateGraph, END, START
from .models import TravelPlannerState
from .agents import (
    info_extractor_agent_node,
    flight_agent_node,
    hotel_agent_node,
    destination_agent_node,
    itinerary_agent_node
)

logger = logging.getLogger(__name__)

workflow = StateGraph(TravelPlannerState)

workflow.add_node("info_extractor", info_extractor_agent_node)
workflow.add_node("flight_search", flight_agent_node)
workflow.add_node("hotel_search", hotel_agent_node)
workflow.add_node("destination_info", destination_agent_node)
workflow.add_node("itinerary_compiler", itinerary_agent_node)

workflow.set_entry_point("info_extractor")

workflow.add_edge("info_extractor", "flight_search")
workflow.add_edge("flight_search", "hotel_search")
workflow.add_edge("hotel_search", "destination_info")
workflow.add_edge("destination_info", "itinerary_compiler")
workflow.add_edge("itinerary_compiler", END)

app = workflow.compile()
logger.info("LangGraph workflow compiled successfully.")
