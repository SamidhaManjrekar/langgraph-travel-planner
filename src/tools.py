import json
import logging
from serpapi import GoogleSearch
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, Union, List, Dict, Any

from .config import SERPAPI_API_KEY 

logger = logging.getLogger(__name__)

# --- Pydantic Models for Tool Inputs ---
class FlightInput(BaseModel):
    source: str = Field(description="The 3-letter IATA airport code for the departure location (e.g., 'DXB').")
    destination: str = Field(description="The 3-letter IATA airport code for the arrival location (e.g., 'SYD').")
    departure_date: str = Field(description="The departure date in YYYY-MM-DD format (e.g., '2025-06-01').")
    return_date: str = Field(description="The return date in YYYY-MM-DD format (e.g., '2025-06-06').")

class HotelInput(BaseModel):
    place: str = Field(description="The city or location for the hotel search (e.g., 'Sydney').")
    check_in_date: str = Field(description="The check-in date in YYYY-MM-DD format (e.g., '2025-06-01').")
    check_out_date: str = Field(description="The check-out date in YYYY-MM-DD format (e.g., '2025-06-06').")
    number_of_adults: int = Field(description="The number of adult guests.")
    budget: Optional[str] = Field(None, description="Optional: 'economy', 'standard', or 'luxury' budget preference.")


@tool(args_schema=FlightInput)
def fetch_flights(source: str, destination: str, departure_date: str, return_date: str):
    """
    Fetch flight options between a source and destination using SerpApi.
    Returns a dictionary with 'source', 'destination', and a list of 'flights' (dicts with airline, times, airports, and price info).
    """
    logger.info(f"Attempting SerpApi Google Flights search for {source} to {destination} on {departure_date} to {return_date}")
    params = {
        "engine": "google_flights",
        "departure_id": source,
        "arrival_id": destination,
        "outbound_date": departure_date,
        "return_date": return_date,
        "stops": "2", 
        "currency": "USD",
        "hl": "en",
        "api_key": SERPAPI_API_KEY
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        logger.debug(f"SerpApi Flight Search Raw Results: {json.dumps(results, indent=2)}")
    except Exception as e:
        logger.error(f"Error during SerpApi Google Flights search: {e}", exc_info=True)
        return {
            "source": source,
            "destination": destination,
            "flights": [],
            "note": f"Error fetching flights: {e}"
        }

    best_flights = results.get("best_flights", [])
    if not best_flights:
        return {
            "source": source,
            "destination": destination,
            "flights": [],
            "note": "No flight options found."
        }

    simplified_flights = []
    
    if best_flights and isinstance(best_flights, list) and best_flights[0].get("flights"):
        for leg in best_flights[0].get("flights", []):
            simplified_flights.append({
                "airline": leg.get("airline", "Unknown"),
                "departure_time": leg.get("departure_airport", {}).get("time", "N/A"),
                "arrival_time": leg.get("arrival_airport", {}).get("time", "N/A"),
                "departure_airport": leg.get("departure_airport", {}).get("name", "Unknown"),
                "arrival_airport": leg.get("arrival_airport", {}).get("name", "Unknown"),
                "price": str(best_flights[0].get("price", "N/A"))
            })
    else:
        return {
            "source": source,
            "destination": destination,
            "flights": [],
            "note": "No detailed flight legs found within the best flight option."
        }

    return {
        "source": source,
        "destination": destination,
        "flights": simplified_flights
    }

@tool(args_schema=HotelInput)
def hotel_search(place: str,
                 check_in_date: str,
                 check_out_date: str,
                 number_of_adults: int,
                 budget: str = None):
    """
    Searches for hotels at the destination location using SerpApi for a specified number of adults.
    You can specify a budget: 'economy', 'standard', or 'luxury'.
    Returns a dictionary with 'place' and a list of 'hotels' (dicts with hotel_name, price_per_night, rating, and amenities).
    This tool does not support searching for children so if children are mentioned the search is only done for the no of adults.
    """
    logger.info(f"Attempting SerpApi Google Hotels search for {place} from {check_in_date} to {check_out_date} for {number_of_adults} adults. Budget: {budget}")
    params = {
        "engine": "google_hotels",
        "q": place,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "adults": number_of_adults,
        "currency": "USD",
        "hl": "en",
        "api_key": SERPAPI_API_KEY
    }

    price_ranges = {
        "economy": {"min_price": 50, "max_price": 175},
        "standard": {"min_price": 176, "max_price": 300},
        "luxury": {"min_price": 301, "max_price": 10000},
    }

    if budget:
        budget_lower = budget.lower()
        if budget_lower in price_ranges:
            params["min_price"] = price_ranges[budget_lower]["min_price"]
            params["max_price"] = price_ranges[budget_lower]["max_price"]
            logger.info(f"Applying budget filter: {budget_lower} (min: ${price_ranges[budget_lower]['min_price']}, max: ${price_ranges[budget_lower]['max_price']})")
        else:
            logger.warning(f"Invalid budget type '{budget}'. Supported types are 'economy', 'standard', 'luxury'. Ignoring budget filter.")

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        logger.debug(f"SerpApi Hotel Search Raw Results: {json.dumps(results, indent=2)}")
    except Exception as e:
        logger.error(f"Error during SerpApi Google Hotels search: {e}", exc_info=True)
        return {
            "place": place,
            "hotels": [],
            "note": f"Error fetching hotels: {e}"
        }

    simplified_hotels = []
    hotels_data_source = results.get("properties", [])
    if not hotels_data_source:
        hotels_data_source = results.get("ads", [])

    if not hotels_data_source:
        logger.info("No 'properties' or 'ads' found in SerpApi results.")

    for i, result in enumerate(hotels_data_source):
        if i >= 2: 
            break

        hotel_info = {
            "hotel_name": result.get("name"),
            "price_per_night": result.get("rate_per_night", {}).get("lowest") or result.get("price"),
            "rating": result.get("overall_rating") or result.get("rating"),
            "amenities": result.get("amenities", [])
        }
        simplified_hotels.append(hotel_info)

    if not simplified_hotels:
        return {
            "place": place,
            "hotels": [],
            "note": "No hotel options found."
        }

    return {
        "place": place,
        "hotels": simplified_hotels
    }
    
# --- List of all tools ---
all_tools = [fetch_flights, hotel_search]
