"""
Professional subtitle-based text synchronization.
Uses subtitle files (.srt/.vtt) for precise text timing control.
"""

import os
import re
import json
from typing import List, Dict, Optional, Tuple

class SubtitleTimingEngine:
    """Professional subtitle-based timing for text overlays."""
    
    def __init__(self):
        self.subtitle_cache = {}
    
    def create_word_level_subtitles(self, text: str, audio_path: str, output_dir: str = ".") -> str:
        """
        Create word-level subtitle file from audio and text.
        
        Args:
            text: Original text
            audio_path: Path to audio file
            output_dir: Directory to save subtitle file
            
        Returns:
            Path to created subtitle file
        """
        try:
            # Try to use forced alignment if available
            subtitle_path = os.path.join(output_dir, "word_timing.srt")
            
            # First try advanced forced alignment
            if self._try_whisper_alignment(text, audio_path, subtitle_path):
                return subtitle_path
                
            # Fallback to smart estimation
            return self._create_smart_subtitle_file(text, audio_path, subtitle_path)
            
        except Exception as e:
            print(f"❌ Subtitle creation failed: {e}")
            return ""
    
    def _try_whisper_alignment(self, text: str, audio_path: str, output_path: str) -> bool:
        """Try to use Whisper for forced alignment."""
        try:
            # Import here to handle missing dependencies gracefully
            from forced_alignment import ForcedAlignmentEngine, create_subtitle_file
            
            print("🎯 Using professional forced alignment...")
            engine = ForcedAlignmentEngine()
            word_timings = engine.get_precise_word_timestamps(audio_path, text)
            
            if word_timings:
                create_subtitle_file(word_timings, output_path, "srt")
                print(f"✅ Professional subtitle file created: {output_path}")
                return True
                
        except ImportError:
            print("⚠️ Advanced forced alignment not available, using fallback method")
        except Exception as e:
            print(f"⚠️ Forced alignment failed: {e}, using fallback method")
            
        return False
    
    def _create_smart_subtitle_file(self, text: str, audio_path: str, output_path: str) -> str:
        """Create smart subtitle file using audio analysis."""
        try:
            from moviepy.editor import AudioFileClip
            
            # Get audio duration
            audio = AudioFileClip(audio_path)
            duration = audio.duration
            audio.close()
            
            words = text.split()
            if not words:
                return ""
            
            # Smart word timing distribution
            word_timings = self._calculate_smart_word_timing(words, duration)
            
            # Create SRT file
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, (word, start, end) in enumerate(word_timings, 1):
                    start_time = self._format_srt_time(start)
                    end_time = self._format_srt_time(end)
                    
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{word}\n\n")
            
            print(f"✅ Smart subtitle file created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Smart subtitle creation failed: {e}")
            return ""
    
    def _calculate_smart_word_timing(self, words: List[str], duration: float) -> List[Tuple[str, float, float]]:
        """Calculate smart word timing based on word characteristics."""
        word_timings = []
        
        # Calculate relative word weights (longer words take more time)
        word_weights = []
        for word in words:
            # Base weight from word length
            length_weight = len(word) / 6.0  # Average word length
            
            # Punctuation adds pause time
            punct_weight = 0.2 if word.rstrip('.,!?;:') != word else 0
            
            # Common words are spoken faster
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            speed_multiplier = 0.8 if word.lower() in common_words else 1.0
            
            total_weight = max(0.3, (length_weight + punct_weight) * speed_multiplier)
            word_weights.append(total_weight)
        
        # Distribute time based on weights
        total_weight = sum(word_weights)
        time_per_weight = duration / total_weight
        
        current_time = 0
        for i, (word, weight) in enumerate(zip(words, word_weights)):
            word_duration = weight * time_per_weight
            start_time = current_time
            end_time = current_time + word_duration
            
            word_timings.append((word, start_time, end_time))
            current_time = end_time
        
        return word_timings
    
    def load_subtitle_timing(self, subtitle_path: str) -> List[Dict]:
        """Load timing data from subtitle file."""
        if subtitle_path in self.subtitle_cache:
            return self.subtitle_cache[subtitle_path]
        
        if not os.path.exists(subtitle_path):
            print(f"❌ Subtitle file not found: {subtitle_path}")
            return []
        
        try:
            if subtitle_path.endswith('.srt'):
                timings = self._parse_srt_file(subtitle_path)
            elif subtitle_path.endswith('.vtt'):
                timings = self._parse_vtt_file(subtitle_path)
            elif subtitle_path.endswith('.json'):
                timings = self._parse_json_file(subtitle_path)
            else:
                print(f"❌ Unsupported subtitle format: {subtitle_path}")
                return []
            
            self.subtitle_cache[subtitle_path] = timings
            print(f"✅ Loaded {len(timings)} subtitle entries from {subtitle_path}")
            return timings
            
        except Exception as e:
            print(f"❌ Failed to parse subtitle file: {e}")
            return []
    
    def _parse_srt_file(self, filepath: str) -> List[Dict]:
        """Parse SRT subtitle file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into subtitle blocks
        blocks = re.split(r'\n\s*\n', content.strip())
        timings = []
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Parse timing line
                timing_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                if timing_match:
                    start_time = self._parse_srt_time(timing_match.group(1))
                    end_time = self._parse_srt_time(timing_match.group(2))
                    text = ' '.join(lines[2:])
                    
                    timings.append({
                        'text': text,
                        'start': start_time,
                        'end': end_time
                    })
        
        return timings
    
    def _parse_vtt_file(self, filepath: str) -> List[Dict]:
        """Parse VTT subtitle file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip header and split into blocks
        lines = content.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == 'WEBVTT':
                start_idx = i + 1
                break
        
        blocks = '\n'.join(lines[start_idx:]).strip().split('\n\n')
        timings = []
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 2:
                # Parse timing line
                timing_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', lines[0])
                if timing_match:
                    start_time = self._parse_vtt_time(timing_match.group(1))
                    end_time = self._parse_vtt_time(timing_match.group(2))
                    text = ' '.join(lines[1:])
                    
                    timings.append({
                        'text': text,
                        'start': start_time,
                        'end': end_time
                    })
        
        return timings
    
    def _parse_json_file(self, filepath: str) -> List[Dict]:
        """Parse JSON subtitle file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        else:
            return data.get('timings', [])
    
    def create_professional_text_segments(self, subtitle_path: str, words_per_segment: int = 4, speech_rate: float = 1.3) -> List[Dict]:
        """
        Create professional text segments from subtitle timing data.
        
        Args:
            subtitle_path: Path to subtitle file
            words_per_segment: Number of words per text segment
            
        Returns:
            List of text segments with precise timing
        """
        word_timings = self.load_subtitle_timing(subtitle_path)
        if not word_timings:
            return []
        
        # Apply speech rate correction to timings
        # When speech is sped up by 1.3x, events happen 1/1.3 = 0.77x as fast
        time_factor = 1.0 / speech_rate if speech_rate > 0 else 1.0
        
        if speech_rate != 1.0:
            print(f"🎯 Applying speech rate correction: {speech_rate}x speed (time factor: {time_factor:.3f})")
            for word_info in word_timings:
                word_info['start'] *= time_factor
                word_info['end'] *= time_factor
        
        segments = []
        current_segment_words = []
        current_segment_start = None
        current_segment_end = None
        
        print(f"🎯 Creating {words_per_segment}-word segments from {len(word_timings)} word timings...")
        
        for word_info in word_timings:
            if current_segment_start is None:
                current_segment_start = word_info['start']
            
            current_segment_words.append(word_info['text'])
            current_segment_end = word_info['end']
            
            # Create segment when we have enough words
            if len(current_segment_words) >= words_per_segment:
                segment_text = ' '.join(current_segment_words)
                
                # Text appears 150ms before speech starts for better sync
                early_display = 0.15
                text_start = max(0, current_segment_start - early_display)
                
                # Text stays until 200ms after speech ends for readability
                reading_buffer = 0.2
                text_duration = (current_segment_end - text_start) + reading_buffer
                
                segments.append({
                    'text': segment_text,
                    'start': text_start,
                    'duration': text_duration,
                    'word_count': len(current_segment_words),
                    'speech_start': current_segment_start,
                    'speech_end': current_segment_end,
                    'confidence': 'subtitle_based'
                })
                
                print(f"  Seg {len(segments)}: '{segment_text[:25]}...' Text@{text_start:.2f}s Speech@{current_segment_start:.2f}s-{current_segment_end:.2f}s")
                
                # Reset for next segment
                current_segment_words = []
                current_segment_start = None
                current_segment_end = None
        
        # Handle remaining words
        if current_segment_words:
            segment_text = ' '.join(current_segment_words)
            early_display = 0.15
            text_start = max(0, current_segment_start - early_display)
            reading_buffer = 0.2
            text_duration = (current_segment_end - text_start) + reading_buffer
            
            segments.append({
                'text': segment_text,
                'start': text_start,
                'duration': text_duration,
                'word_count': len(current_segment_words),
                'speech_start': current_segment_start,
                'speech_end': current_segment_end,
                'confidence': 'subtitle_based'
            })
            
            print(f"  Seg {len(segments)}: '{segment_text[:25]}...' Text@{text_start:.2f}s Speech@{current_segment_start:.2f}s-{current_segment_end:.2f}s")
        
        # Prevent overlaps while maintaining good timing
        segments = self._optimize_segment_timing(segments)
        
        print(f"✅ Created {len(segments)} professional text segments")
        return segments
    
    def _optimize_segment_timing(self, segments: List[Dict]) -> List[Dict]:
        """Optimize segment timing to prevent overlaps while maintaining sync."""
        if len(segments) <= 1:
            return segments
        
        optimized = [segments[0]]  # First segment unchanged
        
        for i in range(1, len(segments)):
            current = segments[i].copy()
            previous = optimized[-1]
            
            prev_end = previous['start'] + previous['duration']
            
            # Check for overlap
            if current['start'] < prev_end:
                overlap = prev_end - current['start']
                
                # Strategy: Reduce previous segment duration slightly, minimal delay for current
                if overlap <= 0.3:  # Small overlap
                    # Shorten previous segment
                    optimized[-1]['duration'] = max(0.5, previous['duration'] - overlap - 0.05)
                else:  # Larger overlap
                    # Minimal delay for current segment
                    current['start'] = prev_end + 0.05
                    current['duration'] = max(0.5, current['duration'] - overlap)
            
            optimized.append(current)
        
        return optimized
    
    def _parse_srt_time(self, time_str: str) -> float:
        """Parse SRT time format to seconds."""
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    def _parse_vtt_time(self, time_str: str) -> float:
        """Parse VTT time format to seconds."""
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def create_professional_text_segments_from_subtitles(text: str, audio_path: str, audio_duration: float, words_per_segment: int = 4, speech_rate: float = 1.3) -> List[Dict]:
    """
    Create professional text segments using subtitle-based timing.
    
    This is the main function to replace the old synchronization system.
    """
    print("🎯 Using professional subtitle-based synchronization...")
    
    try:
        # Initialize subtitle timing engine
        engine = SubtitleTimingEngine()
        
        # Create word-level subtitle file
        subtitle_path = engine.create_word_level_subtitles(text, audio_path)
        
        if not subtitle_path or not os.path.exists(subtitle_path):
            print("❌ Failed to create subtitle file, falling back to basic timing")
            return []
        
        # Create professional text segments from subtitles with speech rate adjustment
        segments = engine.create_professional_text_segments(subtitle_path, words_per_segment, speech_rate)
        
        if segments:
            print(f"✅ Professional synchronization complete: {len(segments)} segments")
            return segments
        else:
            print("❌ No segments created from subtitles")
            return []
            
    except Exception as e:
        print(f"❌ Professional synchronization failed: {e}")
        return []
