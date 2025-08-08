import praw
import edge_tts
import asyncio
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip, CompositeAudioClip
from moviepy.video.fx.resize import resize as video_resize
from dotenv import load_dotenv
import os
import random
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json

# Fix for Pillow 10.x compatibility with MoviePy
try:
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.LANCZOS
except AttributeError:
    pass

# Load environment variables
load_dotenv()

# Video quality configuration - Optimized 1080p Settings (Best Performance/Quality Balance)
VIDEO_CONFIG = {
    "codec": "libx264",
    "audio_codec": "aac", 
    "fps": 60,              # 60 FPS for smooth motion
    "bitrate": "10000k",    # 10 Mbps - optimal for 1080p
    "ffmpeg_params": [
        "-crf", "18",       # High quality for 1080p
        "-preset", "medium",    # Balanced speed vs quality
        "-profile:v", "high",
        "-level", "4.2",    # Standard level for 1080p
        "-pix_fmt", "yuv420p",
        "-x264-params", "ref=4:bframes=3:b-pyramid=normal:mixed-refs=1:8x8dct=1:trellis=1:fast-pskip=1:mbtree=1:merange=16:me=hex:subme=7",  # Optimized x264 settings for speed
        "-movflags", "+faststart",  # Optimize for web streaming
        "-maxrate", "12000k",      # Peak bitrate for 1080p
        "-bufsize", "20000k"       # Buffer size for 1080p
    ]
}

# Audio quality configuration
AUDIO_CONFIG = {
    "audio_bitrate": "320k",    # High quality AAC audio
    "audio_channels": 2,        # Stereo
    "audio_samplerate": 48000   # Professional sample rate
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
    
    # No hard cutoff - return the full story
    return text

# Helper function to detect story perspective for voice selection
def detect_story_perspective(text):
    """Analyze text to determine if story is from male or female perspective."""
    text_lower = text.lower()
    
    # Female indicators
    female_indicators = [
        # Self-referential female terms
        "i'm a woman", "i'm a girl", "i'm female", "as a woman", "as a girl",
        "i am a woman", "i am a girl", "i am female",
        
        # Female relationship terms when used as self-reference
        "my husband", "my boyfriend", "my fiance", "my ex-husband", "my ex-boyfriend",
        
        # Female-specific experiences
        "i'm pregnant", "i am pregnant", "when i was pregnant", "my pregnancy",
        "my period", "menstrual", "maternity leave",
        
        # Female titles/roles
        "i'm a mother", "i am a mother", "as a mom", "my children", "my kids",
        "i'm a wife", "i am a wife", "as a wife",
        
        # Female pronouns in self-reference context
        "i consider myself", "people call me she", "they call me her"
    ]
    
    # Male indicators  
    male_indicators = [
        # Self-referential male terms
        "i'm a man", "i'm a guy", "i'm male", "as a man", "as a guy", 
        "i am a man", "i am a guy", "i am male",
        
        # Male relationship terms when used as self-reference
        "my wife", "my girlfriend", "my fiancee", "my ex-wife", "my ex-girlfriend",
        
        # Male titles/roles
        "i'm a father", "i am a father", "as a dad", "i'm a husband", "i am a husband",
        "as a husband",
        
        # Male pronouns in self-reference context
        "people call me he", "they call me him"
    ]
    
    # Count indicators
    female_count = sum(1 for indicator in female_indicators if indicator in text_lower)
    male_count = sum(1 for indicator in male_indicators if indicator in text_lower)
    
    # Determine perspective
    if female_count > male_count:
        return "female"
    elif male_count > female_count:
        return "male"
    else:
        return "neutral"  # Default to neutral/female voice if unclear

# Helper function to select appropriate voice based on perspective
def select_voice_for_perspective(text):
    """Select TTS voice based on story perspective."""
    perspective = detect_story_perspective(text)
    
    voices = {
        "female": "en-US-AriaNeural",    # Clear, professional female voice
        "male": "en-US-GuyNeural",       # Clear, professional male voice  
        "neutral": "en-US-AriaNeural"    # Default to female voice
    }
    
    selected_voice = voices[perspective]
    print(f"🎤 Detected perspective: {perspective} - Using voice: {selected_voice}")
    
    return selected_voice

# Helper function to calculate optimal speech rate
def calculate_optimal_speech_rate(text, target_max_duration=180):
    """Calculate speech rate - using constant 1.3x for optimal timing."""
    # Use constant 1.3x speed for optimal text synchronization
    optimal_speed = 1.3
    
    # Estimate duration for reference
    words = len(text.split())
    base_duration_minutes = words / 150  # Normal speed
    estimated_duration = (base_duration_minutes * 60) / optimal_speed
    
    print(f"📊 Story analysis: {words} words, estimated {estimated_duration:.1f}s at 1.3x speed")
    print(f"🎯 Using constant 1.3x speed for optimal synchronization")
    
    return optimal_speed

# Helper function to generate high-quality narration
async def generate_narration_async(text, output_file="narration.wav", voice="en-US-AriaNeural", speech_rate="1.3"):
    """Generate high-quality audio from text using Edge TTS."""
    try:
        output_path = os.path.join(os.getcwd(), output_file)
        communicate = edge_tts.Communicate(text, voice, rate=f"+{int((float(speech_rate) - 1.0) * 100)}%")
        await communicate.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error generating narration: {e}")
        return None

def generate_narration(text, output_file="narration.wav", voice=None, speech_rate="1.3"):
    """Wrapper for async narration generation with automatic voice selection."""
    if voice is None:
        voice = select_voice_for_perspective(text)
    
    print(f"🎤 Generating narration at {speech_rate}x speed...")
    return asyncio.run(generate_narration_async(text, output_file, voice, speech_rate))

# Optimized text image creation with pre-loaded font
def create_optimized_text_image(text, font_obj, width=1100, height=220):
    """Creates high quality text image with optimized anti-aliasing."""
    try:
        # Use 2x resolution for anti-aliasing
        scale_factor = 2
        high_width = width * scale_factor
        high_height = height * scale_factor
        
        # Create high resolution image with transparency
        img = Image.new('RGBA', (high_width, high_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Split text into lines and calculate layout
        lines = text.split('\n')
        font = font_obj
        
        # Calculate text dimensions and positioning
        line_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_heights.append(bbox[3] - bbox[1])
        
        total_text_height = sum(line_heights) + (len(lines) - 1) * (scale_factor * 10)
        start_y = (high_height - total_text_height) // 2
        line_spacing = scale_factor * 10
        current_y = start_y
        
        # Render text with enhanced styling
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (high_width - text_width) // 2
            
            # Enhanced shadow and outline
            shadow_offset = 5 * scale_factor
            outline_width = 7 * scale_factor
            
            # Dual-layer shadow for depth
            draw.text((x + shadow_offset + 2, current_y + shadow_offset + 2), line, font=font, fill=(0, 0, 0, 60))
            draw.text((x + shadow_offset, current_y + shadow_offset), line, font=font, fill=(0, 0, 0, 140))
            
            # High-quality outline
            for adj in range(-outline_width, outline_width + 1, 2):
                for adj2 in range(-outline_width, outline_width + 1, 2):
                    if adj != 0 or adj2 != 0:
                        distance = (adj**2 + adj2**2)**0.5
                        if distance <= outline_width:
                            alpha = max(180, 255 - int(distance * 8))
                            draw.text((x + adj, current_y + adj2), line, font=font, fill=(0, 0, 0, alpha))
            
            # Main text
            draw.text((x, current_y), line, font=font, fill=(255, 255, 250, 255))
            current_y += line_heights[i] + line_spacing
        
        # Resize with high quality resampling
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        return np.array(img)
        
    except Exception as e:
        print(f"Error creating text image: {e}")
        return None

# Helper function to analyze audio for precise word timing
def analyze_audio_timing(audio_file_path, text, voice=None):
    """Use TTS-optimized timing estimation for precise word timing."""
    try:
        print("🎯 Analyzing audio for precise word timing...")
        
        # Check for cached timing data - include voice in cache key
        cache_file = audio_file_path.replace('.wav', '_timing_cache.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    cache_key = f"{hash(text)}_{voice}"
                    if cached_data.get('cache_key') == cache_key:
                        print("✅ Using cached timing data")
                        return cached_data['word_timings']
            except Exception:
                pass  # Cache corrupted, continue with fresh analysis
        
        # Use TTS-optimized timing estimation with voice-specific adjustments
        print("🎯 Using voice-optimized timing analysis")
        word_timings = estimate_audio_timing_fallback(audio_file_path, text, 1.5, voice)
        
        if word_timings:
            # Cache the results for future use - include voice in cache
            try:
                cache_data = {
                    'cache_key': f"{hash(text)}_{voice}",
                    'word_timings': word_timings,
                    'timestamp': os.path.getmtime(audio_file_path),
                    'voice': voice
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
                print("💾 Cached voice-specific timing data for future use")
            except Exception:
                pass  # Cache save failed, not critical
        
        return word_timings
        
    except Exception as e:
        print(f"❌ Audio timing analysis failed: {e}")
        return None

def estimate_audio_timing_fallback(audio_file_path, text, speech_rate=1.3, voice=None):
    """TTS-optimized timing with voice-specific adjustments."""
    try:
        # Get actual audio duration
        audio = AudioFileClip(audio_file_path)
        total_duration = audio.duration
        audio.close()
        
        words = text.split()
        if not words:
            return []
        
        # Voice-specific timing adjustments
        voice_timing_factors = {
            "en-US-GuyNeural": 0.961,      # Male voice is 3.9% faster
            "en-US-AriaNeural": 1.0,       # Female voice baseline
            "en-US-JennyNeural": 1.0,      
            "en-US-DavisNeural": 0.961     
        }
        
        # Apply voice-specific timing factor
        timing_factor = voice_timing_factors.get(voice, 1.0)
        if voice and timing_factor != 1.0:
            print(f"🎤 Applying voice timing factor: {timing_factor:.3f} for {voice}")
        
        # Use observed rate from actual audio for accuracy
        observed_wps = len(words) / total_duration
        print(f"🎯 Speech analysis: {len(words)} words in {total_duration:.1f}s = {observed_wps:.2f} words/sec")
        
        word_timings = []
        effective_duration = total_duration * timing_factor
        
        for i, word in enumerate(words):
            # Calculate precise timing based on position
            start_ratio = i / len(words)
            end_ratio = (i + 1) / len(words)
            
            word_start = start_ratio * effective_duration
            word_end = end_ratio * effective_duration
            
            # Ensure timing stays within bounds
            word_start = min(word_start, total_duration - 0.1)
            word_end = min(word_end, total_duration)
            
            # Small word length adjustment
            word_length_factor = max(0.95, min(1.05, len(word) / 6))
            duration_adjustment = (word_end - word_start) * (word_length_factor - 1) * 0.3
            
            adjusted_start = word_start + (duration_adjustment * 0.5)
            adjusted_end = word_end + (duration_adjustment * 0.5)
            
            # Ensure valid bounds
            adjusted_start = max(0, min(adjusted_start, total_duration - 0.1))
            adjusted_end = max(adjusted_start + 0.1, min(adjusted_end, total_duration))
            
            word_timings.append({
                'word': word,
                'start': adjusted_start,
                'end': adjusted_end
            })
        
        # Final verification
        if word_timings:
            word_timings[-1]['end'] = total_duration
        
        print(f"✅ Generated {len(word_timings)} voice-optimized word timings")
        
        return word_timings
        
    except Exception as e:
        print(f"❌ Timing estimation failed: {e}")
        return None

# Helper function to create synchronized text segments
def create_synchronized_text_segments(text, audio_duration, speech_rate=1.3, audio_file_path=None, voice=None):
    """Create text segments synchronized with speech timing using professional subtitle-based approach."""
    
    # Try professional subtitle-based synchronization first
    try:
        from subtitle_sync import create_professional_text_segments_from_subtitles
        
        print("🎯 Attempting professional subtitle-based synchronization...")
        
        if audio_file_path and os.path.exists(audio_file_path):
            professional_segments = create_professional_text_segments_from_subtitles(
                text, audio_file_path, audio_duration, words_per_segment=4
            )
            
            if professional_segments:
                print("✅ Using professional subtitle-based synchronization")
                return professional_segments
            else:
                print("⚠️ Professional sync failed, falling back to enhanced method")
        
    except Exception as e:
        print(f"⚠️ Professional sync unavailable: {e}, using enhanced fallback")
    
    # Enhanced fallback method with improved timing
    return create_enhanced_fallback_segments(text, audio_duration, speech_rate, audio_file_path, voice)

def create_enhanced_fallback_segments(text, audio_duration, speech_rate=1.3, audio_file_path=None, voice=None):
    """Enhanced fallback synchronization with improved timing."""
    
    # Get precise word timings from audio analysis
    if audio_file_path and os.path.exists(audio_file_path):
        word_timings = analyze_audio_timing(audio_file_path, text, voice)
    else:
        print("⚠️ Audio file not provided, using timing estimation")
        word_timings = estimate_audio_timing_fallback(None, text, speech_rate, voice)
    
    if not word_timings:
        print("❌ Failed to get word timings")
        return []
    
    segments = []
    words_per_segment = 4
    
    print(f"🔤 Using {words_per_segment} words per segment with enhanced fallback timing")
    
    # Use actual word timings when available for better sync
    total_words = len(text.split())
    
    # Create segments using actual word timing data
    for i in range(0, total_words, words_per_segment):
        segment_words = text.split()[i:i + words_per_segment]
        segment_index = i // words_per_segment
        
        if segment_words:
            segment_text = ' '.join(segment_words)
            
            # Use word timings for accurate speech timing
            if word_timings and i < len(word_timings):
                # Get timing for the first word in segment
                segment_speech_start = word_timings[i]['start']
                
                # Get timing for the last word in segment 
                last_word_idx = min(i + len(segment_words) - 1, len(word_timings) - 1)
                segment_speech_end = word_timings[last_word_idx]['end']
            else:
                # Fallback to proportional timing if word data missing
                total_segments = (total_words + words_per_segment - 1) // words_per_segment
                time_per_segment = audio_duration / total_segments
                segment_speech_start = segment_index * time_per_segment
                segment_speech_end = (segment_index + 1) * time_per_segment
            
            # Text appears before speech starts (optimized early display)
            early_display = 0.15  # 150ms early - balanced for good sync without excessive delay
            intended_start = max(0, segment_speech_start - early_display)
            text_start = intended_start
            
            # Smart overlap prevention - adjust both current and previous segments
            overlap_adjustment = 0
            if segments:
                prev_segment = segments[-1]
                prev_end = prev_segment['start'] + prev_segment['duration']
                
                if text_start < prev_end:
                    overlap_amount = prev_end - text_start
                    
                    # Strategy: Split the overlap adjustment between shortening previous 
                    # segment and slightly delaying current one
                    prev_reduction = min(overlap_amount * 0.7, prev_segment['duration'] - 0.5)
                    current_delay = overlap_amount - prev_reduction + 0.05
                    
                    # Update previous segment duration
                    segments[-1]['duration'] = max(0.5, prev_segment['duration'] - prev_reduction)
                    
                    # Move current segment start
                    text_start = intended_start + current_delay
                    overlap_adjustment = current_delay
                    
                    print(f"    🔧 Smart overlap fix: prev shortened -{prev_reduction:.2f}s, current delayed +{current_delay:.2f}s")
            
            # Adaptive reading buffer based on position
            if len(segments) == 0:
                reading_buffer = 0.4  # First segment gets longer buffer
            else:
                reading_buffer = 0.25  # Later segments get shorter buffer
            text_duration = (segment_speech_end - text_start) + reading_buffer
            
            # Don't exceed audio bounds
            if text_start + text_duration > audio_duration:
                text_duration = audio_duration - text_start
            
            if text_start < audio_duration and text_duration > 0.3:
                lead_time = text_start - segment_speech_start
                segments.append({
                    'text': segment_text,
                    'start': text_start,
                    'duration': text_duration,
                    'word_count': len(segment_words),
                    'speech_start': segment_speech_start,
                    'confidence': 'word_timing' if word_timings and i < len(word_timings) else 'fallback'
                })
                
                print(f"  Seg {segment_index + 1}: '{segment_text[:20]}...' Text@{text_start:.2f}s Speech@{segment_speech_start:.2f}s Lead={lead_time:+.2f}s{' (delayed +{:.2f}s)'.format(overlap_adjustment) if overlap_adjustment > 0 else ''}")
    
    print(f"📋 Created {len(segments)} text segments with enhanced fallback timing")
    
    # Debug output
    if segments:
        total_coverage = sum(seg['duration'] for seg in segments)
        print(f"🎯 Text coverage: {total_coverage:.1f}s of {audio_duration:.1f}s audio ({(total_coverage/audio_duration)*100:.1f}%)")
    
    return segments

# Helper function to add background music
def add_background_music(video_path, music_folder="static/music", volume=0.025):
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
        
        # Save with background music using premium quality settings
        output_with_music = video_path.replace('.mp4', '_with_music.mp4')
        final_video.write_videofile(
            output_with_music,
            audio_bitrate=AUDIO_CONFIG["audio_bitrate"],
            **VIDEO_CONFIG
        )
        
        return output_with_music
    
    except Exception as e:
        print(f"Error adding background music: {e}")
        return video_path

# Enhanced video creation with text overlays
def create_video_with_text(input_video_file, input_audio_file, text, output_file, speech_rate=1.3, voice=None):
    """Creates a TikTok-style video with text overlays and audio."""
    try:
        output_path = os.path.join(os.getcwd(), output_file)

        # Load video and audio
        video = VideoFileClip(input_video_file)
        audio = AudioFileClip(input_audio_file)
        audio_duration = audio.duration

        # Select random video slice
        max_start_time = max(0, video.duration - audio_duration)
        start_time = random.uniform(0, max_start_time)
        video_slice = video.subclip(start_time, start_time + audio_duration)

        # Crop to 9:16 aspect ratio
        target_aspect_ratio = 9 / 16
        video_width, video_height = video_slice.size
        current_aspect_ratio = video_width / video_height

        if current_aspect_ratio > target_aspect_ratio:
            # Crop width
            new_width = int(video_height * target_aspect_ratio)
            if new_width % 2 != 0:
                new_width -= 1
            crop_x1 = (video_width - new_width) // 2
            video_cropped = video_slice.crop(x1=crop_x1, x2=crop_x1 + new_width)
        else:
            # Crop height  
            new_height = int(video_width / target_aspect_ratio)
            if new_height % 2 != 0:
                new_height -= 1
            crop_y1 = (video_height - new_height) // 2
            video_cropped = video_slice.crop(y1=crop_y1, y2=crop_y1 + new_height)
        
        # Scale to 1080p (1080x1920)
        video_cropped = video_resize(video_cropped, newsize=(1080, 1920))
        
        # Slow down video slightly for more relaxed pacing
        video_speed_factor = 0.75
        if video_speed_factor != 1.0:
            original_duration = video_cropped.duration
            new_duration = original_duration / video_speed_factor
            video_cropped = video_cropped.set_duration(new_duration)
        
        # Ensure video matches audio duration
        video_cropped = video_cropped.set_duration(audio_duration)

        # Create text overlays
        try:
            print("Creating text overlays...")
            
            # Create synchronized text segments
            text_segments = create_synchronized_text_segments(text, audio_duration, speech_rate, input_audio_file, voice)
            
            # Load font
            fonts_dir = os.path.join(os.getcwd(), "fonts")
            font_paths = [
                os.path.join(fonts_dir, "Poppins-Bold.ttf"),
                "C:/Windows/Fonts/impact.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "arial.ttf"
            ]
            
            selected_font_path = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    selected_font_path = font_path
                    break
            
            if not selected_font_path:
                selected_font_path = "arial.ttf"
            
            # Load font object
            try:
                font_obj = ImageFont.truetype(selected_font_path, size=156)
                print(f"Font loaded: {selected_font_path}")
            except Exception:
                font_obj = ImageFont.load_default()
            
            # Create text clips
            text_clips = []
            
            for segment in text_segments:
                words = segment['text'].split()
                
                # Wrap long text
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
                
                # Create text image
                text_img = create_optimized_text_image(wrapped_text, font_obj)
                
                if text_img is not None:
                    text_clip = (ImageClip(text_img, transparent=True)
                               .set_duration(segment['duration'])
                               .set_start(segment['start'])
                               .set_position('center'))
                    text_clips.append(text_clip)

            # Composite video with text overlays
            if text_clips:
                final_video = CompositeVideoClip([video_cropped] + text_clips)
                print(f"✅ {len(text_clips)} text overlays created successfully!")
            else:
                final_video = video_cropped
                print("⚠️  No text overlays created")
            
        except Exception as text_error:
            print(f"⚠️  Text overlay failed: {text_error}")
            final_video = video_cropped
        
        # Add audio and save
        video_with_audio = final_video.set_audio(audio)
        video_with_audio.write_videofile(
            output_path,
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            audio_bitrate=AUDIO_CONFIG["audio_bitrate"],
            **VIDEO_CONFIG
        )

        # Clean up
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
    print("🎬 Reddit Story to Video Generator")
    print("Generating video with text overlays and background music...")

    # Fetch story
    subreddit_name = "AmItheAsshole"
    story = fetch_story(subreddit_name)

    if story:
        print(f"✅ Story fetched from r/{subreddit_name}")
        print(f"Title: {story['title'][:100]}...")
        
        # Process text
        raw_text = f"{story['title']} {story['body']}"
        processed_text = process_text(raw_text)
        print(f"📝 Text processed (length: {len(processed_text)} characters)")
        
        # Calculate speech rate
        optimal_speed = calculate_optimal_speech_rate(processed_text)

        # File paths
        input_video = os.path.join(os.getcwd(), "static/minecraft_background.mp4")
        input_audio = os.path.join(os.getcwd(), "narration.wav")
        output_video = os.path.join(os.getcwd(), "generated_video.mp4")

        # Generate narration
        print("🎤 Generating narration...")
        selected_voice = select_voice_for_perspective(processed_text)
        narration_path = generate_narration(processed_text, input_audio, speech_rate=str(optimal_speed))

        if narration_path and os.path.exists(input_video):
            print("🎬 Creating video with text overlays...")
            video_path = create_video_with_text(input_video, input_audio, processed_text, output_video, optimal_speed, selected_voice)
            
            if video_path:
                print(f"✅ Base video created: {video_path}")
                
                # Add background music
                print("🎵 Adding background music...")
                final_video_path = add_background_music(video_path)
                print(f"🎉 Final video created: {final_video_path}")
                
                # Print summary
                print("\n" + "="*50)
                print("VIDEO GENERATION COMPLETE!")
                print("="*50)
                print(f"Output file: {final_video_path}")
                print(f"Text length: {len(processed_text)} characters")
                print(f"Speech rate: {optimal_speed:.1f}x speed")
                print(f"Voice: {selected_voice}")
                print(f"Quality: 1080p, 60 FPS, 10 Mbps")
                print(f"Audio: 320k AAC stereo")
                print(f"Background music: {'Added' if final_video_path != video_path else 'Not added'}")
                print("="*50)
            else:
                print("Error: Video generation failed.")
        else:
            print("Error: Input files not found.")
            if not os.path.exists(input_video):
                print(f"   Missing: {input_video}")
            if not narration_path:
                print("   Failed to generate narration")
    else:
        print(f"❌ No stories found in r/{subreddit_name}.")