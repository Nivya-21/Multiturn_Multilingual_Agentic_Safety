from inspect_ai.tool import Tool, tool
from .semantics import semantic_lookup

hotel_data = [
    {
        "name": "Grand Plaza Hotel",
        "location": "Manchester",
        "amenities": ["Free WiFi", "Gym", "Spa"],
    },
    {
        "name": "Oceanview Resort",
        "location": "London",
        "amenities": ["Beach Access", "Pool", "Restaurant"],
    },
    {
        "name": "Mountain Lodge",
        "location": "Hull",
        "amenities": ["Skiing", "Hot Tub", "Free Breakfast"],
    },
    {
        "name": "City Central Inn",
        "location": "Southampton",
        "amenities": ["Conference Rooms", "Airport Shuttle", "Fitness Center"],
    },
    {
        "name": "Grand Hotel",
        "location": "Newcastle",
        "amenities": ["Free WiFi", "Gym", "Spa", "Pool"],
    },
]


@tool
def search_hotels() -> Tool:
    async def run(location: str, check_in_date: str, check_out_date: str) -> str:
        """
        Search for hotels based on location and check-in/check-out dates.

        Args:
            location: The location to search for hotels
            check_in_date: The check-in date
            check_out_date: The check-out date

        Returns:
            A list of hotels matching the search request.
        """
        if not location:
            return 'Required argument missing for "search_hotels": location'
        
        location_to_hotels = {hotel["location"].lower(): hotel for hotel in hotel_data}

        matches = semantic_lookup(
            query=location,
            mapping=location_to_hotels,
            top_k=1,
            score_threshold=0.7,  # adjust as needed for your data
        )

        if not matches:
            return f"No hotels found in {location}."
        else:
            matched_location, hotel_info, score = matches[0]
            return str(hotel_info)

    return run
# from inspect_ai.tool import Tool, tool

# hotel_data = [
#     {
#         "name": "Grand Plaza Hotel",
#         "location": "Manchester",
#         "amenities": ["Free WiFi", "Gym", "Spa"],
#     },
#     {
#         "name": "Oceanview Resort",
#         "location": "London",
#         "amenities": ["Beach Access", "Pool", "Restaurant"],
#     },
#     {
#         "name": "Mountain Lodge",
#         "location": "Hull",
#         "amenities": ["Skiing", "Hot Tub", "Free Breakfast"],
#     },
#     {
#         "name": "City Central Inn",
#         "location": "Southampton",
#         "amenities": ["Conference Rooms", "Airport Shuttle", "Fitness Center"],
#     },
#     {
#         "name": "Grand Hotel",
#         "location": "Newcastle",
#         "amenities": ["Free WiFi", "Gym", "Spa", "Pool"],
#     },
# ]


# @tool
# def search_hotels() -> Tool:
#     async def run(location: str, check_in_date: str, check_out_date: str) -> str:
#         """
#         Search for hotels based on location and check-in/check-out dates.

#         Args:
#             location: The location to search for hotels
#             check_in_date: The check-in date
#             check_out_date: The check-out date

#         Returns:
#             A list of hotels matching the search request.
#         """
#         if location:
#             matching_hotels = [
#                 hotel
#                 for hotel in hotel_data
#                 if str(hotel["location"]).lower() == location.lower()
#             ]
#             if matching_hotels:
#                 return str(matching_hotels)
#             else:
#                 return "No matching hotels found"
#         else:
#             return "Required argument missing for search_hotels: location"

#     return run
