from flask import Flask, request, jsonify, send_file
import logging
from flask_cors import CORS
import traceback
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from werkzeug.utils import secure_filename
import requests
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, Dict, List, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from gtts import gTTS
from io import BytesIO
import subprocess
from pydub import AudioSegment
import json
import os
import time
import openai
import tempfile
import uuid
import datetime
from langchain_openai import ChatOpenAI
from functools import lru_cache
from cachetools import cached, TTLCache
import requests
import datetime
# Initialize persistent session globally

# Get current time
current_datetime = datetime.datetime.now()

# Add 5 hours
future_datetime = current_datetime + datetime.timedelta(hours=5)

session = requests.Session()
weather_cache = TTLCache(maxsize=100, ttl=1800)  # Cache weather data for 30 mins


logging.basicConfig(
    level=logging.INFO,  # Temporarily set to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("weather_api.log")]
)

# Configure logging

logger = logging.getLogger(__name__)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
print("Api key -------"+openai_api_key)
# Configure OpenAI client

# Load environment variables


client = openai.OpenAI(api_key=openai_api_key)


llm = ChatOpenAI(
    model="gpt-4o-mini",  
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Define the state structure with memory
class StateWithMemory(TypedDict):
    messages: Annotated[list, add_messages]
    conversation_context: Dict[str, Any]  # Store location and other context information
    bottype: str

# Define your system prompt (keeping your existing one)
system_prompt_text = f"""

You are a friendly weather assistant named (WeatherWalaBot) with access to real-time and forecast weather data. Your goal is to provide accurate, concise, and helpful weather information. Respond briefly (in one line) unless the user explicitly asks for details.

AVAILABLE TOOLS:
- get_weather_by_location(location_name): Fetches weather data for a given location name.
- get_weather(latitude, longitude): Fetches weather data for given coordinates.
- If user asks for **today's weather**, call `fetch_today_weather_by_location`.
- If user asks for **hourly**, **now**, or **specific hour**, call `fetch_hourly_weather_by_location` and match timestamp with user intent.
- If user asks about **another date or tomorrow**, call `fetch_forecast_weather_by_location` and extract relevant date/day/hour.
- Match timestamps like `2025-08-06T14:00:00+0500` to user’s local time automatically.

- **Language Handling:**
    - For Roman Urdu reply in Roman Strictly   (not in urdu)
    - English queries → Reply in English.
    - so on.
    
LOCATION HANDLING:
- Automatically correct spelling mistakes in location names.
- Expand Pakistani city abbreviations as follows:
  isl, isb → islamabad | lh, lhr → lahore | karc, khi, krc → karachi
  pindi, rwp → rawalpindi | mul → multan | faisl, fsd → faisalabad
  gujr, grw → gujranwala | kashmir → kashmir
- Default location: Nust (use automatically if no location provided).
- In follow-up queries, use the most recently mentioned location.

RESPONSE FORMAT:
- Conversational, concise, and engaging.
- Include date, day, and time clearly with all weather data.
- Contextualize temperature clearly (e.g., "23°C, which feels pleasant this spring").
- Simplify weather information (current conditions, brief forecast, significant events only).

  
Additional Information:
- You handle queries specifically for Pakistani users, so avoid mixing Hindi in your responses.
- You're integrated into the WeatherWalay website (https://www.weatherwalay.com/) and its mobile app.
- You are a WeatherExpert familiar with detailed weather information; for anything unfamiliar, refer users to the official website at https://www.weatherwalay.com/.
- **WeatherWalay**, established in 2021 by founder and CEO Junaid Yamin, is Pakistan's first private digital weather service provider dedicated to delivering hyper-local and precise weather information nationwide. Through an extensive network of over 300 Automated Weather Stations (AWS) strategically located across the country, WeatherWalay ensures accurate, real-time updates, forecasts, and timely alerts to more than 20 million Pakistanis. Weather information is made accessible through digital platforms such as mobile apps, websites, and social media, as well as conventional communication channels including SMS, IVR, contact centers, and voice messaging services (VMS). Recognizing the vital role weather data plays in agriculture and industry, WeatherWalay aims to achieve 100% accuracy in nowcasting and 95% accuracy in forecasting. Headquartered in I9, Islamabad, the company remains committed to empowering communities and businesses with reliable, localized weather intelligence.

For assistance:
- **Phone**: +92 51 272 2008  
- **Email**: support@weatherwalay.com  
- **Website**: https://www.weatherwalay.com/
- **Address**: Plot 250, Street 6, I-9/2 Islamabad - Pakistan

GUIDELINES:
- Politely steer non-weather queries back to weather briefly.
- For unclear or gibberish inputs, reply: "I'm not sure I understand. Could you please ask a weather-related question?"
- Keep responses brief, relevant, and clear.
- Always invoke the appropriate weather tool for accurate data.
- Use easily understandable units for temperature and measurements.
- if user ask about current temperature or todays weather or any thing tell us according to the current date and time.which is **{future_datetime}**
"""

system_prompt_audio = f"""

You are a friendly weather assistant named (WeatherWalaBot) with access to real-time and forecast weather data. Your goal is to provide accurate, concise, and helpful weather information. Respond briefly (in one line) unless the user explicitly asks for details.


AVAILABLE TOOLS:
- get_weather_by_location(location_name): Fetches weather data for a given location name.
- get_weather(latitude, longitude): Fetches weather data for given coordinates.

### **LANGUAGE RULES**
- **For English queries → Reply in English.**  (please strictly follow language rules)
- **For Urdu queries → Reply in Urdu**  

LOCATION HANDLING:
- Automatically correct spelling mistakes in location names.
- Expand Pakistani city abbreviations as follows:
  isl, isb → islamabad | lh, lhr → lahore | karc, khi, krc → karachi
  pindi, rwp → rawalpindi | mul → multan | faisl, fsd → faisalabad
  gujr, grw → gujranwala | kashmir → kashmir
- Default location: Islamabad (use automatically if no location provided).
- In follow-up queries, use the most recently mentioned location.

RESPONSE FORMAT:
- Conversational, concise, and engaging.
- Include date, day, and time clearly with all weather data.
- Contextualize temperature clearly (e.g., "23°C, which feels pleasant this spring").
- Simplify weather information (current conditions, brief forecast, significant events only).


 Information:
- You handle queries specifically for Pakistani users, so avoid mixing Hindi in your responses.
- You're integrated into the WeatherWalay website (https://www.weatherwalay.com/) and its mobile app.
- You are a WeatherExpert familiar with detailed weather information; for anything unfamiliar, refer users to the official website at https://www.weatherwalay.com/.
- **WeatherWalay**, established in 2021 by founder and CEO Junaid Yamin, is Pakistan's first private digital weather service provider dedicated to delivering hyper-local and precise weather information nationwide. Through an extensive network of over 300 Automated Weather Stations (AWS) strategicallyAdditional located across the country, WeatherWalay ensures accurate, real-time updates, forecasts, and timely alerts to more than 20 million Pakistanis. Weather information is made accessible through digital platforms such as mobile apps, websites, and social media, as well as conventional communication channels including SMS, IVR, contact centers, and voice messaging services (VMS). Recognizing the vital role weather data plays in agriculture and industry, WeatherWalay aims to achieve 100% accuracy in nowcasting and 95% accuracy in forecasting. Headquartered in I9, Islamabad, the company remains committed to empowering communities and businesses with reliable, localized weather intelligence.

For assistance:
- **Phone**: +92 51 272 2008  
- **Email**: support@weatherwalay.com  
- **Website**: https://www.weatherwalay.com/
- **Address**: Plot 250, Street 6, I-9/2 Islamabad - Pakistan

GUIDELINES:
- Politely steer non-weather queries back to weather briefly.
- For unclear or gibberish inputs, reply: "I'm not sure I understand. Could you please ask a weather-related question?"
- Keep responses brief, relevant, and clear.
- Always invoke the appropriate weather tool for accurate data.
- Use easily understandable units for temperature and measurements.
- if user ask about current temperature or todays weather tell us according to the current date and time.which is **{future_datetime} and match this time that present in JSON response**
"""

# Your existing geocoding and weather functions can remain the same
@lru_cache(maxsize=100)
def get_coordinates(location_name: str) -> Dict[str, Any]:
    """
    Fetches latitude, longitude and a display name for a given location
    from WeatherWalay's location/search API instead of Google.
    """
    logger.info(f"Cache info: {get_coordinates.cache_info()}")
    try:
        logger.info(f"Looking up coordinates for: {location_name}")
        url = "https://services.weatherwalay.com/location/search"
        auth = ("Kalambot", "Ww_12365")
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"searchWord": location_name}

        resp = session.post(url, auth=auth, headers=headers, data=data)
        resp.raise_for_status()

        payload = resp.json()
        if not payload.get("success"):
            msg = payload.get("msg", "Unknown error from location API")
            logger.warning(f"Location API returned failure: {msg}")
            return {"error": msg}

        records = payload.get("record", {}).get("response", [])
        if not records:
            msg = f"No location found for '{location_name}'"
            logger.warning(msg)
            return {"error": msg}

        first = records[0]
        lat = float(first["lat"])
        lng = float(first["lng"])
        display = first.get("address", first.get("city", location_name))

        logger.info(f"Found {location_name} → lat={lat}, lon={lng}, display='{display}'")
        return {
            "latitude": lat,
            "longitude": lng,
            "display_name": display
        }

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error during location lookup: {e}")
        return {"error": str(e)}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed during location lookup: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in get_coordinates: {e}")
        return {"error": str(e)}

@cached(weather_cache)
def get_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    try:
        # Fetch from WeatherWalay
        response = session.post(
            "https://services.weatherwalay.com/v2/weather/byLatLong",
            auth=("Kalambot", "Ww_12365"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"lat": latitude, "long": longitude}
        )
        response.raise_for_status()
        api_response = response.json()

        if not api_response.get("success"):
            return {"error": "Weather unavailable."}

        daily_data = api_response["record"].get("daily", [])
        now = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()

        simplified = {"current_timestamp": now, "forecast": []}

        for i, day in enumerate(daily_data):
            day_entry = {
                "date": day["date"],
                "day_of_week": day.get("dayOfWeek"),
                "condition": day.get("weatherConditionEnglish"),
                "temperature": day.get("temp"),
                "min_temperature": day.get("minTemp"),
                "max_temperature": day.get("maxTemp"),
                "humidity": day.get("hum"),
                "pressure": day.get("pressure"),
                "wind_speed": day.get("windSpeed"),
                "cloud_cover": day.get("cloudCover"),
                "uv_index": day.get("uv"),
                "precipitation_type": day.get("precType"),
                "sunrise": day.get("sunrise"),
                "sunset": day.get("sunset"),
                "day_parts": [],
                "hourly_flattened": []
            }

            for part in day.get("intraday", []):
                dp = {
                    "name": part["daypartName"],
                    "min_temp": part["minTemp"],
                    "max_temp": part["maxTemp"],
                    "condition": part["english"],
                    "day_or_night": "Day" if part["dayOrNight"] == "D" else "Night",
                    "range": part.get("range", [])
                }
                day_entry["day_parts"].append(dp)

                for hour in part.get("range", []):
                    day_entry["hourly_flattened"].append({
                        "timestamp": hour,
                        "daypart": part["daypartName"],
                        "condition": part["english"],
                        "min_temp": part["minTemp"],
                        "max_temp": part["maxTemp"]
                    })

            if i == 0:
                simplified["current"] = day_entry

            simplified["forecast"].append(day_entry)

        return simplified

    except Exception as e:
        return {"error": str(e)}


# def get_weather(latitude: float, longitude: float) -> Dict[str, Any]:
#     """Fetches weather data for given coordinates and returns a simplified response."""
#     try:
#         logger.info(f"Fetching weather data for coordinates: lat={latitude}, lon={longitude}")
#         Define the API endpoint URL
#         url = "https://services.weatherwalay.com/v2/weather/byLatLong"
        
#         Basic Authentication credentials
#         auth = ("Kalambot", "Ww_12365")
        
#         Headers for the POST request
#         headers = {
#             "Content-Type": "application/x-www-form-urlencoded"
#         }
        
#         Data to be sent in the POST request
#         data = {
#             "lat": latitude,
#             "long": longitude
#         }
        
#         Make the POST request to the weather API

#         response = session.post(url, auth=auth, headers=headers, data=data)
#         response.raise_for_status()  # Raise an error if the request fails
        
#         Parse the JSON response
#         api_response = response.json()
        
#         Check if the API call was successful
#         if response.status_code != 200 or not api_response.get('success'):
#             logger.warning("Currently, weather information is unavailable. Please try again shortly.")
#             return {"error": "Currently, weather information is unavailable. Please try again shortly."}
            
#         Extract and simplify the weather data
#         if not api_response.get('record') or not api_response.get('record').get('daily'):
#             logger.warning("Invalid response format from weather API")
#             return {"error": "Currently, weather information is unavailable. Please try again shortly."}
        
#         record = api_response['record']
#         daily_data = record.get('daily', [])
        
#         Create a simplified response
#         simplified_weather = {
#             "current": {},
#             "forecast": []
#         }
        
#         Process current day and forecast
#         for i, day in enumerate(daily_data):
#             day_data = {
#                 "date": day.get('date'),
#                 "day_of_week": day.get('dayOfWeek'),
#                 "condition": day.get('weatherConditionEnglish'),
#                 "temperature": day.get('temp'),
#                 "min_temperature": day.get('minTemp'),
#                 "max_temperature": day.get('maxTemp'),
#                 "humidity": day.get('hum'),
#                 "pressure": day.get('pressure'),
#                 "wind_speed": day.get('windSpeed'),
#                 "cloud_cover": day.get('cloudCover'),
#                 "uv_index": day.get('uv'),
#                 "precipitation_type": day.get('precType'),
#                 "sunrise": day.get('sunrise'),
#                 "sunset": day.get('sunset')
#             }
            
#             Add daily parts breakdown if available
#             if day.get('intraday'):
#                 day_data["day_parts"] = []
#                 for part in day.get('intraday', []):
#                     day_data["day_parts"].append({
#                         "name": part.get('daypartName'),
#                         "min_temp": part.get('minTemp'),
#                         "max_temp": part.get('maxTemp'),
#                         "condition": part.get('english'),
#                         "day_or_night": "Day" if part.get('dayOrNight') == 'D' else "Night"
#                     })
            
#             First day is current day
#             if i == 0:
#                 simplified_weather["current"] = day_data
            
#             All days go to forecast
#             simplified_weather["forecast"].append(day_data)
        
#         logger.info("Weather data successfully fetched and processed")
#         return simplified_weather
        
#     except requests.exceptions.HTTPError as e:
#         logger.error(f"HTTP error occurred while fetching weather: {str(e)}")
#         return {"error": f"HTTP error occurred: {str(e)}"}
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Request failed while fetching weather: {str(e)}")
#         return {"error": f"Request failed: {str(e)}"}
#     except Exception as e:
#         logger.error(f"An unexpected error occurred while fetching weather: {str(e)}")
#         return {"error": f"An unexpected error occurred: {str(e)}"}

def get_weather_by_location(location_name: str) -> Dict[str, Any]:
    """Get weather information for a location by name."""
    # First, geocode the location to get coordinates
    location_data = get_coordinates(location_name)
    
    # Check if geocoding returned an error
    if "error" in location_data:
        return location_data
    
    # Now get the weather using the coordinates
    weather_data = get_weather(location_data["latitude"], location_data["longitude"])
    
    # Add location information to the weather data
    if "error" not in weather_data:
        weather_data["location"] = {
            "name": location_name,
            "formatted_address": location_data.get("display_name"),
            "latitude": location_data["latitude"],
            "longitude": location_data["longitude"]
        }
    
    return weather_data


# ---- STEP 4 CLEARLY GOES HERE ---- #

from langchain_core.tools import tool

# Decorate clearly using @tool
@tool
def fetch_weather(latitude: float, longitude: float) -> dict:
    """Fetches weather data using latitude and longitude."""
    return get_weather(latitude, longitude)

@tool
def fetch_weather_by_location(location_name: str) -> dict:
    """Fetches weather data by location name."""
    return get_weather_by_location(location_name)

@tool
def fetch_today_weather_by_location(location_name: str) -> dict:
    """Fetch today's weather based on location name."""
    loc = get_coordinates(location_name)
    if "error" in loc:
        return loc
    data = get_weather(loc["latitude"], loc["longitude"])
    data["location"] = loc
    return {"today": data["current"], "location": loc, "timestamp": data["current_timestamp"]}

@tool
def fetch_hourly_weather_by_location(location_name: str) -> dict:
    """Fetch hourly weather for today based on location name."""
    loc = get_coordinates(location_name)
    if "error" in loc:
        return loc
    data = get_weather(loc["latitude"], loc["longitude"])
    data["location"] = loc
    return {
        "today": data["current"].get("hourly_flattened", []),
        "location": loc,
        "timestamp": data["current_timestamp"]
    }

@tool
def fetch_forecast_weather_by_location(location_name: str) -> dict:
    """Fetch multi-day forecast for a location."""
    loc = get_coordinates(location_name)
    if "error" in loc:
        return loc
    data = get_weather(loc["latitude"], loc["longitude"])
    data["location"] = loc
    return {
        "forecast": data.get("forecast", []),
        "location": loc,
        "timestamp": data["current_timestamp"]
    }

# Tools list
tools = [
    fetch_weather, 
    fetch_weather_by_location,
    fetch_today_weather_by_location,
    fetch_hourly_weather_by_location,
    fetch_forecast_weather_by_location
    ]

# Now bind tools explicitly (after llm is initialized)
llm_with_tools = llm.bind_tools(tools)

# Function to update conversation context
def update_conversation_context(state: StateWithMemory) -> StateWithMemory:
    updated_state = state.copy()
    updated_state["conversation_context"] = updated_state.get("conversation_context", {})

    messages = updated_state["messages"]

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            if "location" in content:
                updated_state["conversation_context"]["last_location"] = content["location"]["name"]
                logger.info(f"Context updated with location: {content['location']['name']}")
                break

    return updated_state


# Enhanced chatbot  function that uses the conversation context
def enhanced_chatbot(state: StateWithMemory) -> Dict:
    try:
        bottype = state.get("bottype", "text")
        system_prompt = system_prompt_text if bottype == "text" else system_prompt_audio

        logger.info(f"System is Processing:{bottype}")

        # Ensure messages exist in state
        messages = state.get("messages", []).copy()

        # Remove any existing system message to avoid duplication
        messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]

        # Insert the correct system prompt for this request
        messages.insert(0, SystemMessage(content=system_prompt))
        # First call to get tool calls
        response = llm_with_tools.invoke(messages)

        # Check for tool calls in the response's additional_kwargs
        tool_calls = response.additional_kwargs.get('tool_calls', [])

        if tool_calls:
            logger.info("Model requested tool execution.")

            # First, add the AI response WITH the tool calls to the messages
            messages.append(response)

            for tool_call in tool_calls:
                # Extract values from the dictionary
                tool_call_id = tool_call.get('id')
                function_info = tool_call.get('function', {})
                tool_name = function_info.get('name')
                tool_args = json.loads(function_info.get('arguments', '{}'))

                logger.info(f"Calling tool {tool_name} with args {tool_args}, id {tool_call_id}")

                # Execute the appropriate tool
                if tool_name == "fetch_weather_by_location":
                    tool_result = get_weather_by_location(**tool_args)
                elif tool_name == "fetch_weather":
                    tool_result = get_weather(**tool_args)
                elif tool_name == "fetch_today_weather_by_location":
                    tool_result = fetch_today_weather_by_location.invoke(tool_args)
                elif tool_name == "fetch_hourly_weather_by_location":
                    tool_result = fetch_hourly_weather_by_location.invoke(tool_args)
                elif tool_name == "fetch_forecast_weather_by_location":
                    tool_result = fetch_forecast_weather_by_location.invoke(tool_args)
                else:
                    logger.error(f"Unknown tool requested: {tool_name}")
                    continue

                # Add the tool response as a ToolMessage
                tool_message = ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_name,
                    tool_call_id=tool_call_id
                )
                messages.append(tool_message)

            # Second call with tool results included
            final_response = llm_with_tools.invoke(messages)
            return {"messages": [final_response]}

        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in enhanced_chatbot: {str(e)}")
        logger.error(traceback.format_exc())
        return {"messages": [AIMessage(content="I'm having trouble processing your request.")]}



# Rebuild the graph with a simpler structure
graph_builder = StateGraph(StateWithMemory)

# Add nodes
graph_builder.add_node("assistant", enhanced_chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("context_updater", update_conversation_context)

# Simpler linear path
graph_builder.add_edge(START, "assistant")
graph_builder.add_edge("assistant", "tools")
graph_builder.add_edge("tools", "context_updater")

# Explicitly compile with return_only=True
react_graph = graph_builder.compile()

def convert_audio_to_wav(input_audio):
    """
    Convert audio file to WAV format (16kHz, mono) for processing.
    Includes advanced error handling and multiple fallback methods.
    
    Args:
        input_audio (str): Path to the input audio file
        
    Returns:
        str: Path to the converted WAV file or None if conversion fails
    """
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    logger.info(f"Attempting to convert audio file: {input_audio} to WAV format at {temp_wav}")
    
    try:
        # Method 1: Try using pydub's from_file with format hint
        try:
            file_ext = os.path.splitext(input_audio)[1].lower().lstrip('.')
            logger.info(f"Detected file extension: {file_ext}")
            
            # Explicitly specify format to help pydub
            audio = AudioSegment.from_file(input_audio, format=file_ext)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(temp_wav, format="wav")
            logger.info("Successfully converted audio using pydub with format hint")
            return temp_wav
        except Exception as e:
            logger.warning(f"Method 1 failed: {str(e)}")
            
        # Method 2: Try using pydub without format hint
        try:
            audio = AudioSegment.from_file(input_audio)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(temp_wav, format="wav")
            logger.info("Successfully converted audio using pydub without format hint")
            return temp_wav
        except Exception as e:
            logger.warning(f"Method 2 failed: {str(e)}")
            
        # Method 3: Try using ffmpeg directly (if available)
        try:
            cmd = ['ffmpeg', '-i', input_audio, '-ar', '16000', '-ac', '1', '-y', temp_wav]
            logger.info(f"Trying ffmpeg with command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Successfully converted audio using ffmpeg: {result.stdout}")
            return temp_wav
        except Exception as e:
            logger.warning(f"Method 3 failed (ffmpeg): {str(e)}")
            
        # If all methods failed, raise exception
        raise Exception("All conversion methods failed")
            
    except Exception as e:
        error_msg = f"Failed to convert audio to WAV: {str(e)}"
        logger.error(error_msg)
        
        # Clean up the temp file if it exists
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
                logger.info(f"Removed temporary WAV file: {temp_wav}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to remove temporary file: {str(cleanup_error)}")
        
        return None

def transcribe_audio(audio_file_path):
    """
    Transcribe audio file using OpenAI's Whisper API.
    First converts the audio to WAV format.
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        str: Transcribed text
    """
    wav_file_path = None
    
    try:
        logger.info(f"Preparing to transcribe audio file: {audio_file_path}")
        file_size = os.path.getsize(audio_file_path)
        logger.info(f"Audio file size: {file_size} bytes")
        
        # Get file extension
        file_ext = os.path.splitext(audio_file_path)[1].lower()
        logger.info(f"Audio file extension: {file_ext}")
        
        # Convert audio to 16kHz WAV format for Whisper
        wav_file_path = convert_audio_to_wav(audio_file_path)
        
        if not wav_file_path or not os.path.exists(wav_file_path):
            # Try a direct approach with Whisper API if conversion fails
            logger.warning("WAV conversion failed, attempting to use the original file directly")
            
            with open(audio_file_path, "rb") as audio_file:
                try:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    transcribed_text = transcription.text
                    logger.info(f"Direct transcription successful: {transcribed_text[:100]}...")
                    
                    additional_prompt = f"\nAdditional Instructions: Reply in Same Langauage as Message (Roman->Roman, English->Englihs, Urdu->Urdu){future_datetime}"
                    message = transcribed_text + additional_prompt
                    return message
                except Exception as direct_error:
                    logger.error(f"Direct transcription also failed: {str(direct_error)}")
                    raise Exception(f"Both conversion and direct transcription failed: {str(direct_error)}")
        
        logger.info(f"Converted audio to WAV format: {wav_file_path}")
        if os.path.exists(wav_file_path):
            wav_size = os.path.getsize(wav_file_path)
            logger.info(f"WAV file size: {wav_size} bytes")
        
        # Use the converted WAV file for transcription
        with open(wav_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        transcribed_text = transcription.text
        logger.info(f"Transcription successful: {transcribed_text[:100]}...")
        
        additional_prompt = f"\nAdditional Instructions: Reply in Same Langauage as Message (Roman->Roman, English->Englihs, Urdu->Urdu){future_datetime}"
        message = transcribed_text + additional_prompt
        return message
    
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Handle the case where no text was transcribed
        # Return a fallback message instead of raising an exception
        fallback_message = "Sorry, I couldn't understand the audio. Could you please try again or type your message?"
        logger.warning(f"Using fallback message: {fallback_message}")
        return fallback_message
    
    finally:
        # Clean up temp files if they exist
        if wav_file_path and os.path.exists(wav_file_path):
            try:
                os.remove(wav_file_path)
                logger.info(f"Removed temporary WAV file: {wav_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup WAV file: {str(cleanup_error)}")


def text_to_speech(text, voice="ash", speed=1.0):
    """
    Convert text to speech using OpenAI's TTS API.
    
    Args:
        text (str): Text to convert to speech.
        voice (str): Voice to use (options: alloy, echo, fable, onyx, nova, shimmer).
        speed (float): Speed factor for audio playback.
        
    Returns:
        tuple: (BytesIO object, file_path) - Audio file in memory and path to saved file.
    """
    try:
        logger.info(f"Converting text to speech using OpenAI TTS: {text[:500]}...")
        
        # Create a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a directory for saved responses if it doesn't exist
        save_dir = os.path.join(os.getcwd(), "saved_responses")
        os.makedirs(save_dir, exist_ok=True)
        
        # Define the MP3 file path
        mp3_file_path = os.path.join(save_dir, f"response_{timestamp}.mp3")
        
        # Use OpenAI's TTS API to generate speech
        response = client.audio.speech.create(
            model="tts-1-hd",  # or "tts-1-hd" for higher quality
            voice=voice,
            input=text,
            speed=speed
        )
        
        # Save the audio to a file
        response.stream_to_file(mp3_file_path)
        logger.info(f"Text-to-speech conversion successful, saved to {mp3_file_path}")
        
        # Read the file into memory for the response
        with open(mp3_file_path, 'rb') as f:
            audio_data = f.read()
        
        audio_io = BytesIO(audio_data)
        audio_io.seek(0)
        
        return audio_io, mp3_file_path
    
    except Exception as e:
        logger.error(f"Error converting text to speech: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Text-to-speech conversion failed: {str(e)}")

        
# Update the process_audio_to_chat function
def process_audio_to_chat(audio_file_path, state=None):
    """
    Process audio input through transcription, chatbot, and text-to-speech.
    Robust error handling ensures the function doesn't crash.
    
    Args:
        audio_file_path (str): Path to the audio file
        state (dict, optional): Existing conversation state
        
    Returns:
        tuple: (updated_state, response_text, audio_io, saved_path, weather_data)
    """
    try:
        # Check if file exists
        if not os.path.isfile(audio_file_path):
            logger.error(f"Audio file does not exist: {audio_file_path}")
            error_message = "The audio file could not be found."
            return generate_error_response(state, error_message)
        
        # Check file size
        file_size = os.path.getsize(audio_file_path)
        logger.info(f"Processing audio file: {audio_file_path}, Size: {file_size} bytes")
        
        if file_size == 0:
            logger.error("Audio file is empty (0 bytes)")
            error_message = "The audio file is empty. Please try again with a valid recording."
            return generate_error_response(state, error_message)
        
        # Step 1: Transcribe the audio
        transcribed_text = transcribe_audio(audio_file_path)
        logger.info(f"Transcribed text: {transcribed_text}")
        
        # If transcription returned a fallback message (error case)
        if transcribed_text.startswith("Sorry, I couldn't understand"):
            error_message = transcribed_text
            return generate_error_response(state, error_message)
        
        # Step 2: Process with chat bot
        updated_state, response_text, weather_data = chat_with_bot(transcribed_text, "audio", state)
        
        # Step 3: Convert response to speech
        try:
            # Choose voice based on language
            has_urdu_chars = any(ord(c) > 1536 and ord(c) < 1791 for c in response_text)
            voice = "alloy"  # Default voice
            speed = 1.0      # Default speed
            
            audio_io, saved_path = text_to_speech(response_text, voice=voice, speed=speed)
            return updated_state, response_text, audio_io, saved_path, weather_data
            
        except Exception as tts_error:
            logger.error(f"Text-to-speech failed: {str(tts_error)}")
            # If TTS fails, still return the text response
            error_mp3 = create_error_audio()
            return updated_state, response_text, None, error_mp3, weather_data
    
    except Exception as e:
        logger.error(f"Error in process_audio_to_chat: {str(e)}")
        logger.error(traceback.format_exc())
        error_message = "I'm sorry, I encountered an error processing your audio. Please try again."
        return generate_error_response(state, error_message)

def generate_error_response(state, error_message):
    """
    Generate a standard error response for audio processing failures.
    
    Args:
        state: The current conversation state
        error_message: The error message to include in the response
        
    Returns:
        tuple: (state, error_message, None, error_audio_path, None)
    """
    try:
        # Create an error audio file
        error_audio_path = create_error_audio(error_message)
        return state, error_message, None, error_audio_path, None
    except Exception as e:
        logger.error(f"Failed to generate error response: {str(e)}")
        # Last resort fallback
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(os.getcwd(), "saved_responses")
        os.makedirs(save_dir, exist_ok=True)
        fallback_path = os.path.join(save_dir, f"error_{timestamp}.mp3")
        
        # Create an empty MP3 file as absolute last resort
        with open(fallback_path, 'wb') as f:
            f.write(b'')
            
        return state, error_message, None, fallback_path, None

def create_error_audio(message="I'm sorry, I couldn't process your audio. Please try again."):
    """
    Create an audio file with an error message.
    
    Args:
        message: The error message to convert to speech
        
    Returns:
        str: Path to the error audio file
    """
    try:
        audio_io, saved_path = text_to_speech(message, voice="alloy", speed=1.0)
        return saved_path
    except Exception as e:
        logger.error(f"Failed to create error audio: {str(e)}")
        
        # Create a simple audio file using gTTS as fallback
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(os.getcwd(), "saved_responses")
            os.makedirs(save_dir, exist_ok=True)
            fallback_path = os.path.join(save_dir, f"error_{timestamp}.mp3")
            
            tts = gTTS(text=message, lang='en', slow=False)
            tts.save(fallback_path)
            return fallback_path
        except Exception as gtts_error:
            logger.error(f"gTTS fallback also failed: {str(gtts_error)}")
            
            # Create an empty MP3 file as absolute last resort
            empty_path = os.path.join(save_dir, f"empty_error_{timestamp}.mp3")
            with open(empty_path, 'wb') as f:
                f.write(b'')
            return empty_path


def chat_with_bot(message, bottype, state=None):
    if bottype == "text":
        system_prompt = system_prompt_text
    else:
        system_prompt = system_prompt_audio

    logger.info(f"User message: {message}")

    if state is None:
        state = {
            "messages": [],
            "conversation_context": {"last_location": None, "last_query_type": None}
        }

    # **🛠 Keep only relevant message history (User and AI responses)**
    conversation_history = [
        msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))
    ]

    # **🚫 Remove duplicates (Check last user message before appending)**
    if conversation_history and conversation_history[-1].content == message:
        logger.info("Duplicate message detected. Skipping reinsertion.")
    else:
        conversation_history.append(HumanMessage(content=message))  # Append new user message

    # **📝 Construct the new state with a clean conversation history**
    new_state = {
        "messages": conversation_history[-10:],  # Keep last 10 messages only
        "conversation_context": state.get("conversation_context", {}).copy(),
        "bottype": bottype
    }

    try:
        logger.info(f"Sending cleaned state to react_graph: {new_state}")
        result = react_graph.invoke(new_state)

        ai_response = None
        weather_data = None

        # **📌 Extract only AI responses from the result**
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and ai_response is None:
                ai_response = msg.content
            if isinstance(msg, ToolMessage) and weather_data is None:
                content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                if "current" in content or "forecast" in content:
                    weather_data = content

        # **⚠️ Handle empty AI responses (fallback to weather summary)**
        if not ai_response or not ai_response.strip():
            logger.warning("Empty AI response detected. Using fallback.")
            if weather_data and "location" in weather_data:
                current = weather_data["current"]
                location_name = weather_data["location"]["name"]
                ai_response = f"Weather in {location_name}: {current['condition']} with {current['temperature']}°C."

        # **✔ Append AI response while ensuring uniqueness**
        if not conversation_history or conversation_history[-1].content != ai_response:
            conversation_history.append(AIMessage(content=ai_response))

        # **🔁 Ensure the conversation history only keeps the last 10 exchanges**
        new_state["messages"] = conversation_history[-10:]

        return new_state, ai_response, weather_data

    except Exception as e:
        logger.error(f"chat_with_bot Exception: {str(e)}")
        return state, "Error generating response.", None

# Flask Application
app = Flask(__name__)

CORS(app, origins="*")
# Global state storage (in a real app, you'd use a database)
user_states = {}

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        # Process request data
        if request.is_json:
            data = request.json
        elif request.form:
            data = request.form.to_dict()
        elif request.data:
            try:
                data = json.loads(request.data)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON format in request body")
                return jsonify({"status": "error", "message": "Invalid JSON format", "code": 400}), 400
        else:
            logger.warning("Missing request body")
            return jsonify({"status": "error", "message": "Missing request body", "code": 400}), 400
            
        if not data:
            logger.warning("Missing request body")
            return jsonify({"status": "error", "message": "Missing request body", "code": 400}), 400
        
        # Use a single default user since we don't need user_id
        
        user_id = 'default'
        message = data.get('message')
        additional_prompt = f"\nAdditional Instructions: Reply in Same Langauage as Message (Roman->Roman, English->Englihs, Urdu->Urdu){future_datetime}"
        message = message+additional_prompt
        
        if not message:
            logger.warning("Missing 'message' field in request")
            return jsonify({"status": "error", "message": "Missing 'message' field", "code": 400}), 400
        
        # Get existing state or None for new conversation
        state = user_states.get(user_id)
        
        # Create a variable to store weather API response
        #weather_api_data = None
        
        # Process the message with retry mechanism
        max_retries = 3
        retry_count = 0
        response = None
        new_state, response, weather_api_data = chat_with_bot(message, "text", state)


        # If we still don't have a valid response after all retries
        if not response or not response.strip() or response.strip() == "I apologize, but I couldn't generate a response. Please try asking again.":
            response = "I'm currently having trouble generating a detailed response. The weather service is working, but I couldn't format the information properly. Please try again in a moment."
            logger.error("Failed to generate valid response after max retries")
        
        # Store the updated state
        user_states[user_id] = new_state
        
        # Extract conversation context for the response
        context = {}
        if new_state and "conversation_context" in new_state:
            context = new_state["conversation_context"]
        
        logger.info(f"Returning response: {response[:50]}...")
        # Include weather_api_response in the JSON response
        return jsonify({
            "status": "success",
            "code": 200,
            "response": response,
          
        })
    
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": "An internal server error occurred",
            "code": 500
        }), 500

@app.route('/api/audio-chat', methods=['POST'])
def api_audio_chat():
    temp_dir = None
    temp_audio_path = None
    
    try:
        # Check if an audio file is included in the request
        if 'audio' not in request.files:
            logger.warning("No audio file in request")
            return jsonify({
                "status": "error",
                "message": "No audio file provided",
                "code": 400
            }), 400
        
        audio_file = request.files['audio']
        
        # Check if a filename is provided
        if audio_file.filename == '':
            logger.warning("Empty audio filename")
            return jsonify({
                "status": "error",
                "message": "No audio file selected",
                "code": 400
            }), 400
            
        # Log file information to help with debugging
        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        logger.info(f"Received audio file: {audio_file.filename} with extension {file_ext}")
        
        # Get user ID from the request or use default
        user_id = request.form.get('user_id', 'default')
        logger.info(f"Processing audio chat for user_id: {user_id}")
            
        # Create a temporary file to store the uploaded audio
        temp_dir = tempfile.mkdtemp()
        audio_filename = secure_filename(audio_file.filename)
        temp_audio_path = os.path.join(temp_dir, audio_filename)
        
        # Save the uploaded file
        audio_file.save(temp_audio_path)
        logger.info(f"Saved uploaded audio to {temp_audio_path}")
        
        # Get existing state for the user
        state = user_states.get(user_id)
        
        # Process the audio
        try:
            updated_state, response_text, audio_response, saved_path, weather_data = process_audio_to_chat(temp_audio_path, state)
            
            # Update the user's state in the global dictionary
            user_states[user_id] = updated_state
            
            # Prepare the JSON data with metadata
            response_data = {
                "status": "success",
                "code": 200,
                "transcribed_text": response_text,
                "saved_file_path": saved_path,
                "user_id": user_id
            }
            
            # Create a response object with the file
            response = send_file(
                saved_path,
                mimetype="audio/mpeg",
                as_attachment=True,
                download_name="response.mp3"
            )
            
            # Add metadata as a custom header
            response.headers['X-Response-Data'] = json.dumps(response_data)
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return jsonify({
                "status": "error",
                "message": error_msg,
                "code": 500
            }), 500
        
    except Exception as e:
        logger.error(f"API error in audio chat: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"An internal server error occurred: {str(e)}",
            "code": 500
        }), 500
    finally:
        # Clean up temporary files
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception as e:
                logger.warning(f"Error removing temporary audio file: {str(e)}")
        
        if temp_dir and os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Error removing temporary directory: {str(e)}")


def process_audio_to_chat(audio_file_path, state=None):
    """
    Process audio input through transcription, chatbot, and text-to-speech.
    
    Args:
        audio_file_path (str): Path to the audio file
        state (dict, optional): Existing conversation state
        
    Returns:
        tuple: (updated_state, response_text, audio_io, saved_path, weather_data)
    """
    try:
        # Step 1: Transcribe the audio
        transcribed_text = transcribe_audio(audio_file_path)
        logger.info(f"Transcribed text: {transcribed_text}")
        
        # Step 2: Log existing context if available
        if state and "conversation_context" in state:
            context = state.get("conversation_context", {})
            logger.info(f"Processing with existing context: last_location={context.get('last_location', 'None')}")
        else:
            logger.info("Processing without existing context")
        
        # Step 3: Send the transcribed text to the chatbot
        updated_state, response_text, weather_data = chat_with_bot(transcribed_text, "audio", state)
        
        # Step 4: Log updated context
        if updated_state and "conversation_context" in updated_state:
            updated_context = updated_state.get("conversation_context", {})
            logger.info(f"Updated context: last_location={updated_context.get('last_location', 'None')}")
        
        # Step 5: Convert the response text to speech
        audio_io, saved_path = text_to_speech(response_text)
        
        return updated_state, response_text, audio_io, saved_path, weather_data
    
    except Exception as e:
        logger.error(f"Error processing audio to chat: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Audio processing failed: {str(e)}")
    
@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    try:
        # Always use default user since we don't need user_id
        user_id = 'default'
        
        # Reset the user's state
        if user_id in user_states:
            del user_states[user_id]
            logger.info(f"Reset conversation for user {user_id}")
        
        return jsonify({
            "status": "success",
            "code": 200,
            "message": "Conversation reset successfully"
        })
        
    except Exception as e:
        logger.error(f"Reset API error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "An internal server error occurred",
            "code": 500
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "success",
        "code": 200,
        "message": "Weather API is operational"
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "Not found",
        "code": 404
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "status": "error",
        "message": "Method not allowed",
        "code": 405
    }), 405

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "status": "error",
        "message": "Internal server error",
        "code": 500
    }), 500

if __name__ == "__main__":
    logger.info("Starting Weather Assistant API server")
    app.run(debug=False, host='0.0.0.0', port=8029)