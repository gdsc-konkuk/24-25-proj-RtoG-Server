from google import genai
from dotenv import load_dotenv
import os
import sys

def use_gemini( ):
    load_dotenv()

    # Get API key and remove any trailing semicolons
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip(";")
    if not gemini_key:
        print("Error: GEMINI_API_KEY not found in environment variables")
        sys.exit(1)
    
    try:
        client = genai.Client(api_key=gemini_key)
        
        # Check if file exists
        video_path = 'wildfire-test-1.mp4'
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found")
            sys.exit(1)
            
        print("Uploading video file...")
        video = client.files.upload(file=video_path)
        print("Video uploaded successfully")
        
        contents = [
            video,
            "This video is suspected to have a wildfire "
            "using the YOLO model for primary verification. "
            "This video could contains a smoke or flame inside. "
            "Assess the wildfire in this video and if there is one "
            "please describe the detail of the cue of the wildfire "
            "Only in a human understandable manner."
            "There is no need to notify the uncertainty of the AI."
        ]

        print("Generating content...")
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=contents
        )
        
        print("\nResponse:")
        print(response.text)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)
    
if __name__ == "__main__":
    use_gemini()