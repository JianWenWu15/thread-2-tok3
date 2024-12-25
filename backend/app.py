from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import praw  # Python Reddit Wrapper
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
from dotenv import load_dotenv
import os
import random

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
CORS(app)  # Enable Cross-Origin Resource Sharing for React

# Reddit API setup
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent="thread-2-tok/0.1 by u/Complex_Balance4016"
)

# Helper function to fetch a Reddit story
def fetch_story(subreddit="AmItheAsshole"):
    """Fetch a random story from a subreddit."""
    subreddit = reddit.subreddit(subreddit)
    posts = [post for post in subreddit.hot(limit=10) if post.selftext]
    if posts:
        selected_post = random.choice(posts)
        return {
            "title": selected_post.title,
            "body": selected_post.selftext
        }
    return None

# Helper function to generate narration audio
def generate_narration(text, output_file="narration.mp3"):
    """Generate audio from text using gTTS and save as an MP3."""
    try:
        output_file = os.path.join(os.getcwd(), output_file)
        tts = gTTS(text)
        tts.save(output_file)
        return output_file
    except Exception as e:
        print(f"Error generating narration: {e}")
        return None

# Helper function to create a TikTok-compatible video
def create_video(input_video_file, input_audio_file, output_file):
    """Creates a TikTok-style video with a 9:16 aspect ratio and overlays the audio."""
    try:
        output_path = os.path.join(os.getcwd(), output_file)

        # Load video and audio
        video = VideoFileClip(input_video_file)
        audio = AudioFileClip(input_audio_file)

        # Calculate audio duration and select a video slice
        audio_duration = audio.duration
        max_start_time = max(0, video.duration - audio_duration)
        start_time = random.uniform(0, max_start_time)
        end_time = start_time + audio_duration
        video_slice = video.subclip(start_time, end_time)

        # Crop video to fit TikTok's 9:16 aspect ratio
        target_aspect_ratio = 9 / 16
        video_width, video_height = video_slice.size
        current_aspect_ratio = video_width / video_height

        if current_aspect_ratio > target_aspect_ratio:
            # Crop width (landscape video)
            new_width = int(video_height * target_aspect_ratio)
            crop_x1 = (video_width - new_width) // 2
            crop_x2 = crop_x1 + new_width
            video_cropped = video_slice.crop(x1=crop_x1, x2=crop_x2)
        else:
            # Crop height (portrait video)
            new_height = int(video_width / target_aspect_ratio)
            crop_y1 = (video_height - new_height) // 2
            crop_y2 = crop_y1 + new_height
            video_cropped = video_slice.crop(y1=crop_y1, y2=crop_y2)

        # Add audio to the cropped video
        video_with_audio = video_cropped.set_audio(audio)

        # Write the final video
        video_with_audio.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            fps=24
        )

        # Ensure the file exists before returning
        return output_path if os.path.exists(output_path) else None
    except Exception as e:
        print(f"Error creating video: {e}")
        return None

if __name__ == "__main__":
    print("Fetching a story from Reddit and generating video...")

    # Fetch a story from the specified subreddit
    subreddit_name = "AmItheAsshole"
    story = fetch_story(subreddit_name)

    if story:
        print("Story fetched: ", story)
        # Combine the title and body for narration
        narration_text = f"{story['title']} {story['body']}"

        # File paths
        input_video = os.path.join(os.getcwd(), "backend/static/minecraft_background.mp4")  # Path to test video
        input_audio = os.path.join(os.getcwd(), "narration.mp3")  # Path to generated narration audio
        output_video = os.path.join(os.getcwd(), "generated_video.mp4")  # Output video file name

        # Generate narration audio from the fetched story
        narration_path = generate_narration(narration_text, input_audio)

        # Create the video with the narration audio
        if narration_path and os.path.exists(input_video):
            print("Creating video...")
            video_path = create_video(input_video, input_audio, output_video)
            if video_path:
                print(f"Video successfully created: {video_path}")
            else:
                print("Error: Video generation failed.")
        else:
            print("Error: Input video or narration file not found.")
    else:
        print(f"No stories found in the subreddit '{subreddit_name}'.")