# main.py
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
import json 
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fastapi_app")

from .models import TravelPlannerState, TripRequest, FinalFullTravelItinerary
from .workflow import app as langgraph_app

api_app = FastAPI(
    title="GlobeTrekker AI Travel Planner API",
    description="An AI-powered service to generate personalized travel itineraries.",
    version="1.0.0"
)

origins = [
    "http://localhost",
    "http://localhost:5173",
    "https://itinerary-generator.onrender.com",
    "http://127.0.0.1", 
]

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@api_app.get("/", status_code=status.HTTP_200_OK)
async def read_root():
    """
    Root endpoint for basic health check.
    """
    return {"message": "GlobeTrekker AI Travel Planner API is running!"}

@api_app.post("/plan_trip", response_model=FinalFullTravelItinerary)
async def plan_trip_endpoint(request: TripRequest):
    """
    Generates a comprehensive travel itinerary based on user request.
    """
    logger.info(f"Received trip planning request: {request.model_dump_json(indent=2)}")

    initial_state = TravelPlannerState(
        user_info=request.model_dump(), 
        flight_results=None,
        hotel_results=None,
        destination_info_results=None,
        final_itinerary=None,
        notes=[]
    )

    try:
        final_state = await langgraph_app.ainvoke(initial_state) 

        if final_state and final_state.get("final_itinerary"):
            final_itinerary_data = final_state["final_itinerary"]
            logger.info("Successfully generated full itinerary.")
            return FinalFullTravelItinerary(**final_itinerary_data)
        else:
            logger.error("LangGraph workflow completed, but no final itinerary was found in the state.")
            raise HTTPException(status_code=500, detail="Failed to generate complete itinerary.")

    except Exception as e:
        logger.exception("An error occurred during trip planning:")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during trip planning: {e}")

