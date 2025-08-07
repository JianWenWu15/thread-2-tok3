import praw  # Python Reddit Wrapper
import edge_tts  # Better TTS
import asyncio
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip, CompositeAudioClip
from dotenv import load_dotenv
import os
import random
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load environment variables
load_dotenv()

# Video quality configuration
VIDEO_CONFIG = {
    "codec": "libx264",
    "audio_codec": "aac", 
    "fps": 30,
    "bitrate": "8000k",
    "ffmpeg_params": [
        "-crf", "18",  # High quality encoding
        "-preset", "slow",  # Better compression
        "-profile:v", "high",
        "-level", "4.0", 
        "-pix_fmt", "yuv420p"
    ]
}

# Text timing configuration
TEXT_TIMING_CONFIG = {
    "speech_rate": 2.0,  # words per second (~120 WPM)
    "char_duration": 0.12,  # seconds per character
    "sentence_pause": 0.3,  # pause after sentences
    "comma_pause": 0.15,  # pause after commas
    "min_word_duration": 0.5,  # minimum word display time
    "text_early_display": 0.2,  # show text before speech starts
    "min_segment_duration": 0.5  # minimum segment duration
}

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

# Helper function to clean and process text
def process_text(text):
    """Clean and format text for better readability and TTS."""
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic
    text = re.sub(r'~~(.*?)~~', r'\1', text)      # Remove strikethrough
    
    # Clean up common Reddit formatting
    text = re.sub(r'\n+', ' ', text)              # Replace multiple newlines with space
    text = re.sub(r'\s+', ' ', text)              # Replace multiple spaces with single space
    text = text.strip()
    
    # Limit text length for reasonable video duration (adjust as needed)
    if len(text) > 1500:
        text = text[:1500] + "..."
    
    return text

# Helper function to generate better quality narration
async def generate_narration_async(text, output_file="narration.wav", voice="en-US-AriaNeural"):
    """Generate high-quality audio from text using Edge TTS."""
    try:
        output_path = os.path.join(os.getcwd(), output_file)
        
        # Available voices: en-US-AriaNeural (female), en-US-GuyNeural (male), 
        # en-US-JennyNeural (female), en-US-DavisNeural (male)
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        
        return output_path
    except Exception as e:
        print(f"Error generating narration: {e}")
        return None

def generate_narration(text, output_file="narration.wav", voice="en-US-AriaNeural"):
    """Wrapper for async narration generation."""
    return asyncio.run(generate_narration_async(text, output_file, voice))

# Helper function to create text image using PIL with better fonts
# Optimized helper function to create text images with pre-loaded font
def create_optimized_text_image(text, font_obj, width=1100, height=220):
    """Creates an optimized text image using a pre-loaded font object for efficiency."""
    try:
        print(f"🚀 Creating optimized text image for: {text[:20]}...")
        
        # Use 2x resolution for anti-aliasing (quality optimization)
        scale_factor = 2
        high_width = width * scale_factor
        high_height = height * scale_factor
        
        # Create high-resolution image with transparency
        img = Image.new('RGBA', (high_width, high_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Split text into lines and calculate layout
        lines = text.split('\n')
        print(f"📏 Text split into {len(lines)} lines")
        
        # Use pre-loaded font (major performance improvement)
        font = font_obj
        
        # Calculate text dimensions and positioning
        line_heights = []
        max_line_width = 0
        
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height = bbox[3] - bbox[1]
            line_width = bbox[2] - bbox[0]
            line_heights.append(line_height)
            max_line_width = max(max_line_width, line_width)
        
        total_text_height = sum(line_heights) + (len(lines) - 1) * (scale_factor * 10)
        start_y = (high_height - total_text_height) // 2
        line_spacing = scale_factor * 10
        current_y = start_y
        
        # Render text with optimized styling
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (high_width - text_width) // 2
            
            # Efficient shadow and outline rendering
            shadow_offset = 4 * scale_factor
            outline_width = 6 * scale_factor
            
            # Single shadow layer for efficiency
            draw.text((x + shadow_offset, current_y + shadow_offset), line, font=font, fill=(0, 0, 0, 120))
            
            # Simplified outline for performance
            for adj in range(-outline_width, outline_width + 1, 2):  # Step by 2 for efficiency
                for adj2 in range(-outline_width, outline_width + 1, 2):
                    if adj != 0 or adj2 != 0:
                        draw.text((x + adj, current_y + adj2), line, font=font, fill=(0, 0, 0, 200))
            
            # Main text
            draw.text((x, current_y), line, font=font, fill=(255, 255, 245, 255))
            current_y += line_heights[i] + line_spacing
        
        # Resize with high-quality resampling
        img = img.resize((width, height), Image.LANCZOS)
        
        # Convert to numpy array
        result = np.array(img)
        return result
        
    except Exception as e:
        print(f"Error creating optimized text image: {e}")
        return None

# Helper function to estimate speech timing
def estimate_speech_timing(text, total_duration):
    """Estimate when each word will be spoken based on average speech rate."""
    words = text.split()
    # More realistic speech rate: ~120 words per minute = 2.0 words per second
    # This accounts for natural pauses and slower TTS speech
    estimated_speech_rate = TEXT_TIMING_CONFIG["speech_rate"]
    
    # Calculate timing for each word
    word_timings = []
    current_time = 0
    
    for word in words:
        # More conservative timing with better punctuation handling
        base_duration = len(word) * TEXT_TIMING_CONFIG["char_duration"]
        
        # Add extra time for punctuation and longer words
        if word.endswith(('.', '!', '?')):
            base_duration += TEXT_TIMING_CONFIG["sentence_pause"]
        elif word.endswith(','):
            base_duration += TEXT_TIMING_CONFIG["comma_pause"]
        
        word_duration = max(TEXT_TIMING_CONFIG["min_word_duration"], base_duration)
        word_timings.append({
            'word': word,
            'start': current_time,
            'end': current_time + word_duration
        })
        current_time += word_duration
    
    # Scale to fit actual audio duration
    if word_timings:
        total_estimated_time = word_timings[-1]['end']
        time_scale = total_duration / total_estimated_time
        
        for timing in word_timings:
            timing['start'] *= time_scale
            timing['end'] *= time_scale
    
    return word_timings

# Helper function to create synchronized text segments
def create_synchronized_text_segments(text, audio_duration, words_per_segment=3):
    """Create text segments that are synchronized with speech timing."""
    word_timings = estimate_speech_timing(text, audio_duration)
    segments = []
    
    # Group words into segments with proper non-overlapping timing
    for i in range(0, len(word_timings), words_per_segment):
        segment_words = word_timings[i:i + words_per_segment]
        if segment_words:
            segment_text = ' '.join([word['word'] for word in segment_words])
            original_start = segment_words[0]['start']
            original_end = segment_words[-1]['end']
            
            # Add a small buffer to ensure text appears before speech starts
            segment_start = max(0, original_start - TEXT_TIMING_CONFIG["text_early_display"])
            
            # Calculate duration based on the original timing to prevent overlap
            segment_duration = original_end - original_start + TEXT_TIMING_CONFIG["text_early_display"]
            
            # Ensure segments don't overlap by checking against previous segment
            if segments:
                previous_segment = segments[-1]
                previous_end = previous_segment['start'] + previous_segment['duration']
                if segment_start < previous_end:
                    # Adjust start time to prevent overlap
                    segment_start = previous_end
                    # Recalculate duration to maintain original end time
                    segment_duration = max(TEXT_TIMING_CONFIG["min_segment_duration"], original_end - segment_start)
            
            segments.append({
                'text': segment_text,
                'start': segment_start,
                'duration': segment_duration
            })
    
    return segments

# Helper function to add background music
def add_background_music(video_path, music_folder="static/music", volume=0.15):
    """Add background music to video if music files are available."""
    try:
        if not os.path.exists(music_folder):
            return video_path
        
        music_files = [f for f in os.listdir(music_folder) if f.endswith(('.mp3', '.wav', '.m4a'))]
        if not music_files:
            return video_path
        
        # Select random background music
        selected_music = random.choice(music_files)
        music_path = os.path.join(music_folder, selected_music)
        
        video = VideoFileClip(video_path)
        background_music = AudioFileClip(music_path)
        
        # Loop music if it's shorter than video
        if background_music.duration < video.duration:
            background_music = background_music.loop(duration=video.duration)
        else:
            background_music = background_music.subclip(0, video.duration)
        
        # Lower the volume of background music
        background_music = background_music.volumex(volume)
        
        # Mix original audio with background music
        final_audio = CompositeAudioClip([video.audio, background_music])
        final_video = video.set_audio(final_audio)
        
        # Save with background music using shared high quality settings
        output_with_music = video_path.replace('.mp4', '_with_music.mp4')
        final_video.write_videofile(
            output_with_music,
            **VIDEO_CONFIG
        )
        
        return output_with_music
    
    except Exception as e:
        print(f"Error adding background music: {e}")
        return video_path

# Helper function to create enhanced video with text overlays
def create_video_with_text(input_video_file, input_audio_file, text, output_file):
    """Creates a TikTok-style video with text overlays and audio."""
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

        # Crop video to fit TikTok's 9:16 aspect ratio with quality preservation
        target_aspect_ratio = 9 / 16
        video_width, video_height = video_slice.size
        current_aspect_ratio = video_width / video_height

        if current_aspect_ratio > target_aspect_ratio:
            # Crop width (landscape video) - ensure we maintain resolution
            new_width = int(video_height * target_aspect_ratio)
            # Ensure width is even for FFmpeg compatibility
            if new_width % 2 != 0:
                new_width -= 1
            crop_x1 = (video_width - new_width) // 2
            crop_x2 = crop_x1 + new_width
            video_cropped = video_slice.crop(x1=crop_x1, x2=crop_x2)
        else:
            # Crop height (portrait video) - ensure we maintain resolution  
            new_height = int(video_width / target_aspect_ratio)
            # Ensure height is even for FFmpeg compatibility
            if new_height % 2 != 0:
                new_height -= 1
            crop_y1 = (video_height - new_height) // 2
            crop_y2 = crop_y1 + new_height
            video_cropped = video_slice.crop(y1=crop_y1, y2=crop_y2)
        
        # Note: Removed upscaling to avoid PIL compatibility issues
        # The video quality will be preserved at its original resolution
        print(f"📏 Video dimensions: {video_cropped.size[0]}x{video_cropped.size[1]}")

        # Efficient text overlay creation with font pre-loading
        try:
            print("🎨 Creating synchronized text overlays with optimized rendering...")
            
            # Create synchronized text segments
            text_segments = create_synchronized_text_segments(text, audio_duration, words_per_segment=3)
            
            # Pre-load font ONCE for efficiency
            fonts_dir = os.path.join(os.getcwd(), "fonts")
            font_paths = [
                os.path.join(fonts_dir, "Poppins-Bold.ttf"),
                os.path.join(fonts_dir, "Oswald-Bold.ttf"),
                os.path.join(fonts_dir, "Montserrat-Bold.ttf"),
                "C:/Windows/Fonts/impact.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "arial.ttf"  # Final fallback
            ]
            
            selected_font_path = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    selected_font_path = font_path
                    print(f"✅ Using font: {font_path}")
                    break
            
            if not selected_font_path:
                print("⚠️ No custom fonts found, using system default")
                selected_font_path = "arial.ttf"
            
            # Pre-load the font object ONCE (major optimization)
            try:
                font_obj = ImageFont.truetype(selected_font_path, size=128)  # 2x size for quality
                print(f"✅ Font pre-loaded: {selected_font_path}")
            except Exception as font_error:
                print(f"⚠️ Font loading failed: {font_error}")
                font_obj = ImageFont.load_default()
            
            # Create text clips with optimized approach
            text_clips = []
            print(f"📊 Processing {len(text_segments)} text segments...")
            
            for i, segment in enumerate(text_segments):
                if i % 20 == 0:  # Progress updates every 20 segments
                    print(f"📝 Processing segment {i+1}/{len(text_segments)}")
                
                # Smart text wrapping
                words = segment['text'].split()
                if len(words) > 4:
                    if len(words) <= 6:
                        mid_point = len(words) // 2
                        wrapped_text = ' '.join(words[:mid_point]) + '\n' + ' '.join(words[mid_point:])
                    else:
                        third = len(words) // 3
                        wrapped_text = ' '.join(words[:third]) + '\n' + ' '.join(words[third:third*2]) + '\n' + ' '.join(words[third*2:])
                elif len(words) in [3, 4]:
                    mid_point = len(words) // 2
                    wrapped_text = ' '.join(words[:mid_point]) + '\n' + ' '.join(words[mid_point:])
                else:
                    wrapped_text = segment['text']
                
                # Create optimized text image with pre-loaded font
                text_img = create_optimized_text_image(wrapped_text, font_obj, width=1100, height=220)
                
                if text_img is not None:
                    text_clip = (ImageClip(text_img, transparent=True)
                               .set_duration(segment['duration'])
                               .set_start(segment['start'])
                               .set_position('center'))
                    text_clips.append(text_clip)

            # Composite video with synchronized text overlays
            if text_clips:
                final_video = CompositeVideoClip([video_cropped] + text_clips)
                print(f"✅ {len(text_clips)} synchronized text overlays created successfully!")
            else:
                final_video = video_cropped
                print("⚠️  No text overlays created, using video without text")
            
        except Exception as text_error:
            print(f"⚠️  Text overlay failed: {text_error}")
            print("📝 Continuing without text overlays...")
            final_video = video_cropped
        
        # Add audio to the video
        video_with_audio = final_video.set_audio(audio)

        # Write the final video with shared high quality settings
        video_with_audio.write_videofile(
            output_path,
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            **VIDEO_CONFIG
        )

        # Clean up clips
        video.close()
        audio.close()
        video_with_audio.close()
        if 'final_video' in locals():
            final_video.close()

        return output_path if os.path.exists(output_path) else None
    except Exception as e:
        print(f"Error creating video: {e}")
        return None

if __name__ == "__main__":
    print("🎬 Enhanced Reddit Story to Video Generator")
    print("Fetching a story from Reddit and generating video with text overlays and background music...")

    # Fetch a story from the specified subreddit
    subreddit_name = "AmItheAsshole"
    story = fetch_story(subreddit_name)

    if story:
        print(f"✅ Story fetched from r/{subreddit_name}")
        print(f"Title: {story['title'][:100]}...")
        
        # Process and clean the text
        raw_text = f"{story['title']} {story['body']}"
        processed_text = process_text(raw_text)
        print(f"📝 Text processed (length: {len(processed_text)} characters)")

        # File paths
        input_video = os.path.join(os.getcwd(), "static/minecraft_background.mp4")
        input_audio = os.path.join(os.getcwd(), "narration.wav")  # Changed to WAV for better quality
        output_video = os.path.join(os.getcwd(), "generated_video.mp4")

        # Generate high-quality narration audio
        print("🎤 Generating high-quality narration...")
        narration_path = generate_narration(processed_text, input_audio, voice="en-US-AriaNeural")

        if narration_path and os.path.exists(input_video):
            print("🎬 Creating video with text overlays...")
            video_path = create_video_with_text(input_video, input_audio, processed_text, output_video)
            
            if video_path:
                print(f"✅ Base video created: {video_path}")
                
                # Add background music if available
                print("🎵 Adding background music...")
                final_video_path = add_background_music(video_path)
                print(f"🎉 Final video created: {final_video_path}")
                
                # Print summary
                print("\n" + "="*50)
                print("VIDEO GENERATION COMPLETE!")
                print("="*50)
                print(f"📁 Output file: {final_video_path}")
                print(f"📊 Original text length: {len(raw_text)} characters")
                print(f"📊 Processed text length: {len(processed_text)} characters")
                print(f"🎵 Background music: {'Added' if final_video_path != video_path else 'Not added (no music files found)'}")
                print("="*50)
            else:
                print("❌ Error: Video generation failed.")
        else:
            print("❌ Error: Input video or narration file not found.")
            if not os.path.exists(input_video):
                print(f"   Missing: {input_video}")
            if not narration_path:
                print("   Failed to generate narration audio")
    else:
        print(f"❌ No stories found in the subreddit 'r/{subreddit_name}'.")