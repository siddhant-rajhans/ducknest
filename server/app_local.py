from flask import Flask, request, jsonify
import os
import json
import requests
import base64
from dotenv import load_dotenv
import logging
import time
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# API endpoints
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

def load_housing_data():
    """
    Load housing data from Supabase
    """
    try:
        response = supabase.table('houses').select('*').execute()
        return response.data
    except Exception as e:
        logger.error(f"Error loading houses from Supabase: {e}")
        # Fall back to local data if database is unavailable
        return DUMMY_HOUSING_DATA

# Initialize Supabase tables and seed data if needed
def initialize_database():
    """
    Initialize database with dummy data if tables are empty
    """
    try:
        # Check if houses table has data
        houses_response = supabase.table('houses').select('count').execute()
        house_count = houses_response.count if hasattr(houses_response, 'count') else len(houses_response.data)
        
        if house_count == 0:
            logger.info("Seeding houses table with dummy data")
            for house in DUMMY_HOUSING_DATA:
                supabase.table('houses').insert(house).execute()
            
        # Ensure responses table exists (will not fail if already exists)
        logger.info("Database initialization completed")
    
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

def query_groq(sentences):
    """
    Query the Groq API with user sentences to get housing recommendations
    """
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set in environment variables")
        return {"error": "API key configuration error"}, 500
    
    try:
        # Load housing data from database
        housing_database = load_housing_data()
        
        # Prepare the prompt for the Groq model
        user_query = " ".join(sentences)
        system_prompt = """
        You are a housing recommendation assistant for college students looking for housing near Stevens Institute of Technology in Hoboken, NJ. 
        
        Based on the user's query, analyze their preferences and return suitable housing options from our database.
        
        First, provide your housing recommendations in valid JSON format ONLY with this structure:
        {
            "matches": [list of matching properties with all fields],
            "recommendation": {
                "title": "street_address",
                "images": ["image_urls"],
                "Cost": "rent as string with $ sign",
                "Room": "house_type or no_of_bedrooms info",
                "Lease": "lease_duration",
                "ownerPhone": "contact_details"
            }
        }
        
        Then, AFTER the JSON (not within it), provide a short explanation paragraph about why you made this recommendation, mentioning the proximity to Stevens and any other relevant factors.
        """
        
        # Create the payload for the Groq API
        payload = {
            "model": "llama3-70b-8192",  # Using Llama 3 70B model
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User query: {user_query}\n\nHousing database: {json.dumps(housing_database)}"}
            ],
            "temperature": 0.2,  # Lower temperature - consistent results
            "max_tokens": 2000
        }
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Sending request to Groq API for query: {user_query}")
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        # Extract the model's response
        result = response.json()
        model_response = result["choices"][0]["message"]["content"]
        
        # Parse the JSON part from the response
        try:
            # Find the JSON part in the response
            json_start = model_response.find('{')
            json_end = model_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = model_response[json_start:json_end]
                parsed_json = json.loads(json_str)
                
                # Get explanation text after the JSON
                explanation_text = model_response[json_end:].strip()
                
                return {
                    "recommendation_data": parsed_json,
                    "explanation_text": explanation_text
                }
            else:
                # If JSON parsing fails, try to find JSON within code blocks
                if "```json" in model_response:
                    json_str = model_response.split("```json")[1].split("```")[0].strip()
                    parsed_json = json.loads(json_str)
                    
                    # Get explanation text after the code block
                    explanation_parts = model_response.split("```")
                    if len(explanation_parts) > 2:
                        explanation_text = explanation_parts[2].strip()
                    else:
                        explanation_text = ""
                        
                    return {
                        "recommendation_data": parsed_json,
                        "explanation_text": explanation_text
                    }
                elif "```" in model_response:
                    json_str = model_response.split("```")[1].split("```")[0].strip()
                    parsed_json = json.loads(json_str)
                    
                    # Get explanation text after the code block
                    explanation_parts = model_response.split("```")
                    if len(explanation_parts) > 2:
                        explanation_text = explanation_parts[2].strip()
                    else:
                        explanation_text = ""
                        
                    return {
                        "recommendation_data": parsed_json,
                        "explanation_text": explanation_text
                    }
                else:
                    logger.error(f"Could not find valid JSON in the response")
                    return {"error": "Failed to parse model response", "raw_response": model_response}, 500
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing model response as JSON: {e}")
            logger.error(f"Raw response: {model_response}")
            return {"error": "Failed to parse model response", "raw_response": model_response}, 500
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request to Groq API: {e}")
        return {"error": f"Error communicating with language model API: {str(e)}"}, 500
    except Exception as e:
        logger.error(f"Unexpected error in query_groq: {e}")
        return {"error": f"Unexpected error: {str(e)}"}, 500

def generate_audio(text):
    """
    Generate audio from text using ElevenLabs API
    """
    if not ELEVENLABS_API_KEY:
        logger.error("ELEVENLABS_API_KEY is not set in environment variables")
        return None
    
    try:
        # Basic text cleaning for audio synthesis
        clean_text = text.replace("\n", " ").strip()
        
        # Limit text length for audio generation
        if len(clean_text) > 5000:
            clean_text = clean_text[:5000] + "..."
        
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        
        # Using a default voice, you can customize this
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice ID, can be changed
        
        payload = {
            "text": clean_text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        logger.info("Sending request to ElevenLabs API")
        response = requests.post(
            f"{ELEVENLABS_API_URL}/{voice_id}",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        # Save audio file locally
        timestamp = int(time.time())
        audio_filename = f"audio_{timestamp}.mp3"
        audio_path = os.path.join("audio_files", audio_filename)
        
        # Ensure directory exists
        os.makedirs("audio_files", exist_ok=True)
        
        with open(audio_path, "wb") as f:
            f.write(response.content)
        
        # Return audio content as base64 and the file path
        audio_base64 = base64.b64encode(response.content).decode('utf-8')
        return {
            "audio_base64": audio_base64,
            "audio_path": audio_path
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request to ElevenLabs API: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in generate_audio: {e}")
        return None

def store_response_in_db(user_query, recommendation_data, explanation_text, audio_path=None):
    """
    Store the response in the Supabase database
    """
    try:
        response_data = {
            "user_query": user_query,
            "recommendation_data": recommendation_data,
            "explanation_text": explanation_text,
            "audio_path": audio_path,
            "created_at": str(time.time())
        }
        
        result = supabase.table('responses').insert(response_data).execute()
        logger.info(f"Stored response in database with ID: {result.data[0]['id'] if result.data else 'unknown'}")
        return result.data[0]['id'] if result.data else None
    except Exception as e:
        logger.error(f"Error storing response in database: {e}")
        return None

@app.route('/recommend', methods=['POST'])
def recommend_housing():
    """
    API endpoint to recommend housing based on input sentences
    """
    try:
        data = request.json
        sentences = data.get('sentences', [])
        
        if not sentences:
            return jsonify({"error": "No input sentences provided"}), 400
        
        user_query = " ".join(sentences)
        
        # Query the Groq API for recommendations
        recommendation_result = query_groq(sentences)
        
        # Check if there was an error
        if isinstance(recommendation_result, tuple) and len(recommendation_result) == 2 and isinstance(recommendation_result[0], dict) and "error" in recommendation_result[0]:
            return jsonify(recommendation_result[0]), recommendation_result[1]
        
        recommendation_data = recommendation_result.get("recommendation_data", {})
        explanation_text = recommendation_result.get("explanation_text", "")
        
        # Generate audio ONLY for the explanation text
        audio_result = generate_audio(explanation_text)
        
        # Store the response in the database
        audio_path = audio_result.get("audio_path") if audio_result else None
        response_id = store_response_in_db(user_query, recommendation_data, explanation_text, audio_path)
        
        # Prepare the final response
        response = {
            "text_response": recommendation_data,
            "explanation_text": explanation_text,
            "audio_response": audio_result.get("audio_base64") if audio_result else None,
            "response_id": response_id
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in recommend_housing endpoint: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# # Dummy housing data for Stevens Institute of Technology area
# DUMMY_HOUSING_DATA = [
#     {
#         "id": "h001",
#         "city": "Hoboken",
#         "street_address": "215 River St",
#         "rent": 1850,
#         "lease_duration": "12 months",
#         "availability_date": "2025-04-15",
#         "furnished": True,
#         "utilities_included": True,
#         "wifi_available": True,
#         "no_of_bedrooms": 1,
#         "no_of_bathrooms": 1,
#         "house_type": "Apartment",
#         "distance_to_college": "0.3 miles",
#         "transportation_options": ["Bus", "Path Train"],
#         "recommended_by": "John Smith",
#         "contact_details": "15551112233",
#         "image_urls": ["assets/house1.jpg", "assets/house1_2.jpg", "assets/house1_3.jpg"]
#     },
#     {
#         "id": "h002",
#         "city": "Hoboken",
#         "street_address": "78 Washington St",
#         "rent": 2100,
#         "lease_duration": "9 months",
#         "availability_date": "2025-04-10",
#         "furnished": True,
#         "utilities_included": True,
#         "wifi_available": True,
#         "no_of_bedrooms": 1,
#         "no_of_bathrooms": 1,
#         "house_type": "Studio",
#         "distance_to_college": "0.2 miles",
#         "transportation_options": ["Bus", "Path Train"],
#         "recommended_by": "Sarah Johnson",
#         "contact_details": "15551115566",
#         "image_urls": ["assets/house2.jpg", "assets/house2_2.jpg", "assets/house2_3.jpg"]
#     },
#     {
#         "id": "h003",
#         "city": "Hoboken",
#         "street_address": "420 Hudson St",
#         "rent": 2650,
#         "lease_duration": "12 months",
#         "availability_date": "2025-05-01",
#         "furnished": True,
#         "utilities_included": False,
#         "wifi_available": True,
#         "no_of_bedrooms": 2,
#         "no_of_bathrooms": 1,
#         "house_type": "Apartment",
#         "distance_to_college": "0.4 miles",
#         "transportation_options": ["Bus", "Path Train"],
#         "recommended_by": "Mike Peters",
#         "contact_details": "15551117788",
#         "image_urls": ["assets/house3.jpg", "assets/house3_2.jpg"]
#     },
#     {
#         "id": "h004",
#         "city": "Jersey City",
#         "street_address": "25 River Dr",
#         "rent": 1950,
#         "lease_duration": "6 months",
#         "availability_date": "2025-04-05",
#         "furnished": False,
#         "utilities_included": False,
#         "wifi_available": False,
#         "no_of_bedrooms": 1,
#         "no_of_bathrooms": 1,
#         "house_type": "Condo",
#         "distance_to_college": "1.2 miles",
#         "transportation_options": ["Bus", "Path Train", "Light Rail"],
#         "recommended_by": "Jane Doe",
#         "contact_details": "15551118899",
#         "image_urls": ["assets/house4.jpg"]
#     },
#     {
#         "id": "h005",
#         "city": "Hoboken",
#         "street_address": "89 Willow Ave",
#         "rent": 1700,
#         "lease_duration": "12 months",
#         "availability_date": "2025-04-20",
#         "furnished": False,
#         "utilities_included": True,
#         "wifi_available": True,
#         "no_of_bedrooms": 1,
#         "no_of_bathrooms": 1,
#         "house_type": "Apartment",
#         "distance_to_college": "0.5 miles",
#         "transportation_options": ["Bus"],
#         "recommended_by": "Robert Chen",
#         "contact_details": "15551119900",
#         "image_urls": ["assets/house5.jpg", "assets/house5_2.jpg"]
#     },
#     {
#         "id": "h006",
#         "city": "Hoboken",
#         "street_address": "505 Madison St",
#         "rent": 3200,
#         "lease_duration": "12 months",
#         "availability_date": "2025-05-15",
#         "furnished": True,
#         "utilities_included": True,
#         "wifi_available": True,
#         "no_of_bedrooms": 3,
#         "no_of_bathrooms": 2,
#         "house_type": "Townhouse",
#         "distance_to_college": "0.7 miles",
#         "transportation_options": ["Bus", "Path Train"],
#         "recommended_by": "Lisa Martinez",
#         "contact_details": "15551110011",
#         "image_urls": ["assets/house6.jpg", "assets/house6_2.jpg", "assets/house6_3.jpg"]
#     },
#     {
#         "id": "h007",
#         "city": "Hoboken",
#         "street_address": "123 Clinton St",
#         "rent": 2000,
#         "lease_duration": "9 months",
#         "availability_date": "2025-04-01",
#         "furnished": True,
#         "utilities_included": False,
#         "wifi_available": True,
#         "no_of_bedrooms": 1,
#         "no_of_bathrooms": 1,
#         "house_type": "Apartment",
#         "distance_to_college": "0.4 miles",
#         "transportation_options": ["Bus"],
#         "recommended_by": "David Wilson",
#         "contact_details": "15551112222",
#         "image_urls": ["assets/house7.jpg"]
#     },
#     {
#         "id": "h008",
#         "city": "Weehawken",
#         "street_address": "50 Harbor Blvd",
#         "rent": 2300,
#         "lease_duration": "12 months",
#         "availability_date": "2025-05-10",
#         "furnished": False,
#         "utilities_included": False,
#         "wifi_available": True,
#         "no_of_bedrooms": 1,
#         "no_of_bathrooms": 1,
#         "house_type": "Apartment",
#         "distance_to_college": "1.8 miles",
#         "transportation_options": ["Bus", "Ferry"],
#         "recommended_by": "Amanda Thompson",
#         "contact_details": "15551113333",
#         "image_urls": ["assets/house8.jpg", "assets/house8_2.jpg"]
#     }
# ]

if __name__ == "__main__":
    # Check if API keys are set
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY is not set. The API will not work properly.")
    if not ELEVENLABS_API_KEY:
        logger.warning("ELEVENLABS_API_KEY is not set. Audio generation will not work.")
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase credentials not set. Using local storage only.")
    else:
        # Initialize database with sample data
        initialize_database()
        
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)