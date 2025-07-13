import json
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from pydantic import ValidationError

from .config import llm_general, llm_structured
from .tools import all_tools 
from .models import ( 
    TravelPlannerState,
    Flight,
    Hotel,
    ItineraryActivityDetail,
    TravelOption,
    ResearchDetail,
    ItineraryItem,
    FullTravelItinerary,
    FinalFullTravelItinerary,
    ExtractedTripInfo,
    DestinationInformation 
)

logger = logging.getLogger(__name__)

def create_extractor_chain(llm, system_prompt):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain

def create_agent(llm, tools, system_prompt):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    if tools:
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False) 
        return executor
    else:
        chain = prompt | llm | StrOutputParser()
        return chain


def info_extractor_agent_node(state: TravelPlannerState):
    logger.info("Info Extractor Agent: Parsing and standardizing user request using Pydantic.")
    user_info = state["user_info"]
    current_notes = state.get("notes", [])

    extractor_prompt = """You are a highly efficient information extraction agent.
    Your goal is to parse a user's travel request and extract all relevant details,
    standardizing them into a JSON object strictly conforming to the provided schema.

    Infer missing details where possible (e.g., IATA codes for cities/countries).
    If a country is provided for source/destination, use the IATA code of its most prominent or capital city's airport (e.g., 'USA' -> 'JFK' or 'LAX', 'UK' -> 'LHR').
    **IMPORTANT: Ensure IATA codes are exactly 3 uppercase letters, without any surrounding quotes.**
    If a country is provided for the hotel city, pick its most prominent city for tourism (e.g., 'Australia' -> 'Sydney', 'France' -> 'Paris'). **Ensure 'hotel_city' is always populated.**

    The number of days should be calculated based on departure and return dates.
    If 'no_of_children' is not specified, default to 0.

    Always respond with ONLY the JSON object, no conversational text before or after.
    """

    extractor_chain = ChatPromptTemplate.from_messages(
        [
            ("system", extractor_prompt),
            ("human", "{input}"),
        ]
    ) | llm_structured.with_structured_output(ExtractedTripInfo) 

    user_input_str = json.dumps(user_info) 

    extracted_data: Optional[ExtractedTripInfo] = None
    try:
        extracted_data = extractor_chain.invoke({"input": user_input_str})
        logger.debug(f"Extracted Data (Pydantic object): {extracted_data.model_dump()}")

        try:
            start_date = datetime.strptime(extracted_data.departure_date, '%Y-%m-%d')
            end_date = datetime.strptime(extracted_data.return_date, '%Y-%m-%d')
            calculated_num_days = (end_date - start_date).days + 1
            if extracted_data.num_days != calculated_num_days:
                logger.warning(f"Calculated num_days ({calculated_num_days}) differs from LLM's ({extracted_data.num_days}). Overwriting.")
            extracted_data.num_days = calculated_num_days 
            
        except (ValueError, KeyError):
            current_notes.append("Could not calculate number of days from provided dates. Using LLM's or default.")
        updated_user_info = user_info.copy()
        
        updated_user_info.update(extracted_data.model_dump())

        current_notes.append("User information successfully extracted and standardized.")

        return {
            "user_info": updated_user_info,
            "notes": current_notes
        }

    except ValidationError as e:
        error_msg = f"Failed to extract structured information due to Pydantic validation: {e}. Errors: {e.errors()}"
        logger.error(error_msg, exc_info=True)
        current_notes.append(error_msg)
        return {
            "user_info": user_info,
            "notes": current_notes
        }
    except Exception as e:
        error_msg = f"An unexpected error occurred during Info Extractor Agent processing: {e}"
        logger.critical(error_msg, exc_info=True)
        current_notes.append(error_msg)
        return {
            "user_info": user_info,
            "notes": current_notes
        }


def flight_agent_node(state: TravelPlannerState):
    logger.info("Flight Agent: Searching for flights.")
    user_info = state["user_info"]
    current_notes = state.get("notes", [])

    try:
        source_iata = user_info.get('source_iata')
        destination_iata = user_info.get('destination_iata')
        departure_date = user_info.get('departure_date')
        return_date = user_info.get('return_date')

        if not all([source_iata, destination_iata, departure_date, return_date]):
            missing_info = [f for f in ['source_iata', 'destination_iata', 'departure_date', 'return_date'] if not user_info.get(f)]
            raise ValueError(f"Missing required flight parameters from user_info: {', '.join(missing_info)}")

        tool_output = all_tools[0].invoke({ 
            'source': source_iata,
            'destination': destination_iata,
            'departure_date': departure_date,
            'return_date': return_date
        })
        logger.debug(f"Flight Tool Output: {tool_output}")

        flights = []
        if tool_output and tool_output.get("flights"):
            for flight_data in tool_output["flights"]:
                try:
                    flights.append(Flight(
                        airline=flight_data.get("airline", "Unknown Airline"),
                        departure_time=flight_data.get("departure_time", "N/A"),
                        arrival_time=flight_data.get("arrival_time", "N/A"),
                        departure_airport=flight_data.get("departure_airport", "Unknown Departure Airport"),
                        arrival_airport=flight_data.get("arrival_airport", "Unknown Arrival Airport"),
                        price=str(flight_data.get("price", "N/A"))
                    ))
                except ValidationError as e:
                    logger.warning(f"Error parsing flight data into Flight model: {e} for data: {flight_data}", exc_info=True)
        
        flight_info_for_state = {
            "flights": [f.model_dump() for f in flights],
            "note": tool_output.get("note", "Flight search completed.")
        }
        
        if tool_output.get("note"):
            current_notes.append(tool_output["note"])

        return {
            "flight_results": flight_info_for_state,
            "notes": current_notes
        }

    except Exception as e:
        error_msg = f"Error during Flight Agent processing: {e}"
        logger.error(error_msg, exc_info=True)
        current_notes.append(error_msg)
        return {
            "flight_results": {"flights": [], "note": error_msg},
            "notes": current_notes
        }


def hotel_agent_node(state: TravelPlannerState):
    logger.info("Hotel Agent: Searching for hotels.")
    user_info = state["user_info"]
    current_notes = state.get("notes", [])

    try:
        hotel_city = user_info.get('hotel_city')
        check_in_date = user_info.get('departure_date')
        check_out_date = user_info.get('return_date')
        number_of_adults = user_info.get('no_of_adults', 1)
        budget = user_info.get('budget')

        if not all([hotel_city, check_in_date, check_out_date, number_of_adults]):
            missing_info = [f for f in ['hotel_city', 'departure_date', 'return_date', 'no_of_adults'] if not user_info.get(f)]
            raise ValueError(f"Missing required hotel parameters from user_info: {', '.join(missing_info)}")

        tool_output = all_tools[1].invoke({ 
            'place': hotel_city,
            'check_in_date': check_in_date,
            'check_out_date': check_out_date,
            'number_of_adults': number_of_adults,
            'budget': budget
        })

        if tool_output.get("note"):
            current_notes.append(tool_output["note"])
        
        # logger.debug(f"Hotel Tool Output (raw): {tool_output}")

        return {"hotel_results": tool_output, "notes": current_notes}

    except Exception as e:
        error_msg = f"Hotel search failed: {e}"
        logger.error(error_msg, exc_info=True)
        current_notes.append(error_msg)
        return {
            "hotel_results": {"hotels": [], "note": error_msg},
            "notes": current_notes
        }


def destination_agent_node(state: TravelPlannerState):
    logger.info("Destination Agent: Planning activities, travel options, and gathering research with structured output.")
    user_info = state["user_info"]
    flight_info = state.get("flight_results", {})
    hotel_info = state.get("hotel_results", {})
    current_notes = state.get("notes", [])

    destination_prompt_content = """You are a comprehensive travel guide assistant.
             Given the user's travel details and collected flight/hotel information,
             generate a structured JSON object strictly conforming to the `DestinationInformation` schema.

             For 'activities':
             - Suggest suitable activities for the user's destination, travel dates,
               the arrival time of the plane (if available), and activity preferences.
             - Make sure to add **3-4** activities for each day after considering the arrival time of the flight and the number of days available.
             - MAKE SURE THAT EVERY DAY HAS AT LEAST ONE ACTIVITY INCLUDING THE LAST DAY. AND TRY TO FILL EACH DAY WITH 3-4 ACTIVITIES.
             - If the destination is a country, suggest activities in its most prominent cities while considering travel logistics (e.g., For Japan, suggest activities in Tokyo, Kyoto, and Osaka if there are enough days).
             - Provide: **Activity Name**, a small **Description**, typical **Ticket Price** (if applicable, e.g., '$45-$80' or 'Free'), and **Best Time to Visit** (e.g., 'Morning', 'Afternoon', 'Evening', 'All day').

             For 'local_travel_options':
             - Describe 3-4 common and convenient local transportation options for tourists in the specified destination.
             - Include: **Method** of transport (e.g., 'Subway', 'Bus', 'Taxi', 'High-Speed Rail') and a brief **Description** for each.

             For 'destination_research':
             - Provide general practical information for the specified destination and travel dates.
             - Include: **Weather Outlook**, **Local Customs**, **Safety Tips**, **Currency and Language**.

             Ensure all fields in the `DestinationInformation` schema are present and accurately populated.
             Do NOT include any conversational text outside of the JSON object.
             """

    info_agent_chain = ChatPromptTemplate.from_messages(
        [
            ("system", destination_prompt_content),
            ("human", "{input}"),
        ]
    ) | llm_structured.with_structured_output(DestinationInformation)

    query_input = (
        f"User destination: {user_info.get('destination') or user_info.get('hotel_city')}\n"
        f"Departure Date: {user_info.get('departure_date')}\n"
        f"Return Date: {user_info.get('return_date')}\n"
        f"Number of days: {user_info.get('num_days', 'unknown')}\n"
        f"Activity Preferences: {user_info.get('activity_preferences', 'no specific preferences')}\n"
        f"Budget Preference: {user_info.get('budget', 'any')}\n\n"
        f"Flight info summary: {flight_info.get('note', str(flight_info.get('flights', 'No flight info')))}.\n"
        f"Hotel info summary: {hotel_info.get('note', str(hotel_info.get('hotels', 'No hotel info')))}."
    )

    generated_info: Optional[DestinationInformation] = None
    try:
        generated_info = info_agent_chain.invoke({"input": query_input})
        logger.debug(f"Destination Agent Output (Pydantic object): {generated_info.model_dump()}")

        return {
            "destination_info_results": generated_info, 
        }
    except ValidationError as e:
        error_msg = f"Failed to generate structured destination information due to Pydantic validation: {e}. Errors: {e.errors()}"
        logger.error(error_msg, exc_info=True)
        current_notes.append(error_msg)
        return {
            "destination_info_results": None, 
            "notes": state.get("notes", []) + [error_msg]
        }
    except Exception as e:
        error_msg = f"An unexpected error occurred during Destination Agent processing: {e}"
        logger.critical(error_msg, exc_info=True)
        current_notes.append(error_msg)
        return {
            "destination_info_results": None,
            "notes": state.get("notes", []) + [error_msg]
        }


def itinerary_agent_node(state: TravelPlannerState):
    logger.info("Itinerary Agent: Compiling final itinerary.")
    user_info = state["user_info"]
    flight_results = state.get("flight_results", {})
    hotel_results = state.get("hotel_results", {})
    destination_info_object = state.get("destination_info_results", None)
    current_notes = state.get("notes", [])

    # --- Hotel Augmentation Logic ---
    hotel_updated = []
    if hotel_results and hotel_results.get("hotels"):
        for hotel_data in hotel_results["hotels"]:
            try:
                price_value = hotel_data.get("price_per_night", "N/A") 
                if isinstance(price_value, (int, float)):
                    price_value = f"${price_value}"
                    
                hotel_obj = Hotel(
                    hotel_name=hotel_data.get("hotel_name", "N/A"),
                    price_per_night=price_value,
                    rating=hotel_data.get("rating"),
                    amenities=hotel_data.get("amenities", [])[:5],
                    address="",
                    description="",
                    perks=""
                )
                
                hotel_updated.append(hotel_obj)
            except ValidationError as e:
                current_notes.append(f"Error preparing hotel data for augmentation: {e} for data: {hotel_data}")
                logger.warning(f"Error preparing hotel data for augmentation: {e} for data: {hotel_data}", exc_info=True)

    augmented_hotel_objects: List[Hotel] = []
    if hotel_updated:
        logger.info("Augmenting hotel details within Itinerary Agent...")
        hotels_updated_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a hotel expert that specialized in providing descriptive details for hotels across the world.
                 Given a hotel name, its city, and some existing details, provide:
                 - **Address**: The exact address of the hotel mentioned.
                 - **Description**: A concise, appealing description (in less than 14 words) highlighting its style, offerings, and target audience.
                 - **Perks**: A small line (in less than 12 words) summarizing the unique perks of staying at that hotel.

                 Base your response on general knowledge about hotels and cities.
                 Format your output strictly as plain text with clear labels for each piece of information.
                 Example for the hotel "The Park New Delhi":
                 Address: 15 Parliament Street, Connaught Place, New Delhi, Delhi 110001, India
                 Description: A stylish hotel with modern amenities and excellent service, ideal for a comfortable stay.
                 Perks: Located in the heart of Delhi, close to major attractions and transport links.

                 If you do not know the answer, just try to make an educated guess but never leave it blank.
                 If you do make a guess don't mention that in the response, just provide the guessed information as if it was factual.
                 Do NOT include any conversational text or explanations in your response like "This is a fictional address" etc.
                 """),
                ("human", """Hotel Name: {hotel_name}
                 City: {hotel_city}
                 Existing Details (Rating: {hotel_rating}, Amenities: {hotel_amenities})

                 Please provide the Address, Description, and Perks for this hotel.""")
            ]
        )
        augmentation_chain = hotels_updated_prompt | llm_general | StrOutputParser() 

        for hotel_obj in hotel_updated:
            hotel_name = hotel_obj.hotel_name
            hotel_city_updated = hotel_results.get("place", user_info.get('hotel_city'))
            hotel_rating = hotel_obj.rating if hotel_obj.rating is not None else "no rating"
            hotel_amenities = ", ".join(hotel_obj.amenities) or "basic amenities"

            try:
                augmentation_result_str = augmentation_chain.invoke({
                    "hotel_name": hotel_name,
                    "hotel_city": hotel_city_updated,
                    "hotel_rating": str(hotel_rating),
                    "hotel_amenities": hotel_amenities
                })

                augmented_details = {}
                for line in augmentation_result_str.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value = value.strip()
                        augmented_details[key] = value

                hotel_obj.address = augmented_details.get("address", hotel_obj.address)
                hotel_obj.description = augmented_details.get("description", hotel_obj.description)
                hotel_obj.perks = augmented_details.get("perks", hotel_obj.perks)
                
                augmented_hotel_objects.append(hotel_obj)
            except Exception as e:
                current_notes.append(f"Error during hotel augmentation for {hotel_name}: {e}")
                logger.error(f"Error during hotel augmentation for {hotel_name}: {e}", exc_info=True)
                augmented_hotel_objects.append(hotel_obj)

        logger.info("Hotel augmentation complete.")
    else:
        logger.info("No hotels found by the tool for augmentation or an error occurred during hotel search.")


    activities_list = destination_info_object.activities if destination_info_object else []
    travel_options = destination_info_object.local_travel_options if destination_info_object else []
    research_details = destination_info_object.destination_research if destination_info_object else []


    flights_for_prompt: List[Flight] = []
    if flight_results and flight_results.get("flights"):
        for flight_data in flight_results["flights"]:
            try:
                flights_for_prompt.append(Flight(**flight_data))
            except ValidationError as e:
                current_notes.append(f"Error parsing flight for prompt: {e} - data: {flight_data}")
                logger.warning(f"Error parsing flight for prompt: {e} - data: {flight_data}", exc_info=True)

    itinerary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a professional travel itinerary planner.
             Compile all the provided information into a structured JSON object conforming to the `FullTravelItinerary` schema.
             
             **Crucially, do NOT include 'user_request_summary' in your generated JSON output.** This field will be populated separately by the system.
             
             **Ensure ALL fields in the `FullTravelItinerary` schema are present and accurately populated.**

             For 'itinerary' (daily activities):
             - You MUST create an `ItineraryItem` object for EACH day of the trip, starting from the `arrival_time` of the last flight provided in the `flights_json`.
             - Each `ItineraryItem` must include `day` (1-indexed), `date` (YYYY-MM-DD), `city` (which is the hotel city or main destination city), and a `List[ItineraryActivityDetail]` for `activities`.
             - Distribute the activities provided in `activities_list_json` *evenly and logically* across ALL the days including the return_date. Do NOT put all activities on one day.
             - Make sure each day has **at least one activity** if possible.
             - Make sure that the return_date also has at least one activity listed, even if it's a short one.
             - If there are fewer activities than days, some days may have fewer activities listed. If there are many activities, spread them out appropriately, suggesting 2-3 activities per day.
             - Use the exact 'activity_name', 'description', 'ticket_price', and 'best_time_to_visit' from the `activities_list_json`. If 'ticket_price' is not explicitly mentioned, use "Varies" or "Check locally".

             For 'flight':
             - Ensure `airline`, `departure_time`, `arrival_time`, `departure_airport`, `arrival_airport`, and `price` are accurately included for each flight leg from `flights_json`.

             For 'hotels':
             - Ensure `hotel_name`, `address`, `price_per_night`, `rating`, `amenities`, `description`, and `perks` are accurately included for each augmented hotel from `augmented_hotel_details_objects_json`.

             For 'travel_options':
             - Populate with `method` and `description` for each local travel option from `travel_options_json`.

             For 'note' (research details):
             - Populate with `title` and `notes` for each research detail from `research_details_json`.

             Do NOT generate any text outside of the JSON object.
             
             Here is the collected travel information for your reference:
             User Request Summary (for context, NOT to be put in output JSON): {user_info}
             Flight Details: {flights_json}
             Augmented Hotel Details: {augmented_hotel_details_objects_json}
             Parsed Activities (from Destination Info Agent): {activities_list_json}
             Parsed Travel Options (from Destination Info Agent): {travel_options_json}
             Parsed Research Details (from Destination Info Agent): {research_details_json}
             Notes/Warnings: {notes}
             """
            ),
            ("human", """Generate the comprehensive travel itinerary as a JSON object:""")
        ]
    )
    flights_json = [f.model_dump() for f in flights_for_prompt]

    logger.debug(f"Flight json check: {flights_json}")
    augmented_hotel_details_objects_json = [h.model_dump() for h in augmented_hotel_objects] if augmented_hotel_objects else []
    activities_list_json = [a.model_dump() for a in activities_list]
    travel_options_json = [option.model_dump() for option in travel_options]
    research_details_json = [detail.model_dump() for detail in research_details]

    num_days_int = user_info.get('num_days', 0)
    try:
        num_days_int = int(num_days_int)
    except ValueError:
        num_days_int = 0 
        
    structured_itinerary_chain = itinerary_prompt.partial(num_days=num_days_int) | llm_structured.with_structured_output(FullTravelItinerary)

    try:
        compiled_itinerary_object_partial = structured_itinerary_chain.invoke({
            "user_info": user_info,
            "flights_json": flights_json,
            "augmented_hotel_details_objects_json": augmented_hotel_details_objects_json,
            "activities_list_json": activities_list_json,
            "travel_options_json": travel_options_json,
            "research_details_json": research_details_json,
            "notes": current_notes
        })
    except ValidationError as e:
        error_msg = f"ValidationError during structured itinerary compilation: {e}. Debugging info: {e.errors()}"
        logger.error(error_msg, exc_info=True)
        current_notes.append(error_msg)
        return {"final_itinerary": {"notes_and_warnings": current_notes, "disclaimer": "Partial itinerary due to internal error."}}
    except Exception as e:
        error_msg = f"An unexpected error occurred during structured itinerary compilation: {e}"
        logger.critical(error_msg, exc_info=True)
        current_notes.append(error_msg)
        return {"final_itinerary": {"notes_and_warnings": current_notes, "disclaimer": "Partial itinerary due to internal error."}}


    final_itinerary_data = compiled_itinerary_object_partial.model_dump()
    logger.debug(f"Compiled Itinerary Data (raw): {final_itinerary_data}")
    final_itinerary_data["user_request_summary"] = user_info

    try:
        final_compiled_itinerary = FinalFullTravelItinerary(**final_itinerary_data)
        logger.info("Itinerary Agent Output (structured object) successfully compiled.")
        return {"final_itinerary": final_compiled_itinerary.model_dump()}
    except ValidationError as e:
        error_msg = f"Final Pydantic Validation Error for Full Travel Itinerary: {e}. Debugging info: {e.errors()}"
        logger.critical(error_msg, exc_info=True)
        current_notes.append(error_msg)
        if "notes_and_warnings" in final_itinerary_data:
            final_itinerary_data["notes_and_warnings"].append(f"Validation failed for final itinerary: {e}")
        else:
            final_itinerary_data["notes_and_warnings"] = [f"Validation failed for final itinerary: {e}"]
        final_itinerary_data["disclaimer"] = "Partial itinerary generated due to final validation error."
        return {"final_itinerary": final_itinerary_data}
