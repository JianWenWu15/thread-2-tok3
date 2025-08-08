"""
Professional forced alignment for precise word-level timestamps.
This module provides accurate word timing for perfect text-speech synchronization.
"""

import os
import json
import subprocess
import tempfile
from typing import List, Dict, Optional
import wave
import librosa
import numpy as np

try:
    import whisper_timestamped as whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ whisper-timestamped not available. Install with: pip install whisper-timestamped")

class ForcedAlignmentEngine:
    """Professional forced alignment engine for word-level timestamps."""
    
    def __init__(self):
        self.model = None
        self.temp_dir = tempfile.mkdtemp()
        
    def initialize_whisper(self):
        """Initialize Whisper model for forced alignment."""
        if not WHISPER_AVAILABLE:
            return False
            
        try:
            print("🎯 Loading Whisper model for forced alignment...")
            self.model = whisper.load_model("base")
            print("✅ Whisper model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            return False
    
    def get_precise_word_timestamps(self, audio_path: str, text: str) -> Optional[List[Dict]]:
        """
        Get precise word-level timestamps using forced alignment.
        
        Args:
            audio_path: Path to audio file
            text: Original text that was spoken
            
        Returns:
            List of word timing dictionaries with start/end times
        """
        print("🎯 Performing forced alignment for precise timestamps...")
        
        # Try Whisper timestamped first (most accurate)
        if WHISPER_AVAILABLE and self.model is None:
            self.initialize_whisper()
            
        if WHISPER_AVAILABLE and self.model is not None:
            return self._whisper_alignment(audio_path, text)
        
        # Fallback to other methods if Whisper unavailable
        return self._fallback_alignment(audio_path, text)
    
    def _whisper_alignment(self, audio_path: str, text: str) -> Optional[List[Dict]]:
        """Use Whisper timestamped for forced alignment."""
        try:
            print("🎯 Using Whisper for forced alignment...")
            
            # Transcribe with word-level timestamps
            result = whisper.transcribe(self.model, audio_path, language="en")
            
            # Extract word-level timestamps
            word_timings = []
            if 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment:
                        for word_info in segment['words']:
                            word_timings.append({
                                'word': word_info['text'].strip(),
                                'start': word_info['start'],
                                'end': word_info['end'],
                                'confidence': word_info.get('confidence', 1.0)
                            })
            
            print(f"✅ Extracted {len(word_timings)} precise word timestamps")
            return self._align_with_original_text(word_timings, text)
            
        except Exception as e:
            print(f"❌ Whisper alignment failed: {e}")
            return None
    
    def _align_with_original_text(self, detected_words: List[Dict], original_text: str) -> List[Dict]:
        """Align detected words with original text."""
        original_words = original_text.lower().split()
        aligned_timings = []
        
        detected_idx = 0
        for i, orig_word in enumerate(original_words):
            # Find best match in detected words
            best_match = None
            search_range = min(3, len(detected_words) - detected_idx)
            
            for j in range(search_range):
                if detected_idx + j < len(detected_words):
                    detected_word = detected_words[detected_idx + j]['word'].lower().strip('.,!?;:')
                    
                    # Check for exact match or substring match
                    if (orig_word == detected_word or 
                        orig_word in detected_word or 
                        detected_word in orig_word):
                        best_match = detected_words[detected_idx + j]
                        detected_idx += j + 1
                        break
            
            if best_match:
                aligned_timings.append({
                    'word': original_words[i],  # Use original word
                    'start': best_match['start'],
                    'end': best_match['end'],
                    'confidence': best_match.get('confidence', 1.0)
                })
            else:
                # Estimate timing for unmatched words
                if aligned_timings:
                    last_end = aligned_timings[-1]['end']
                    estimated_start = last_end + 0.05
                    estimated_end = estimated_start + 0.3
                else:
                    estimated_start = i * 0.35
                    estimated_end = estimated_start + 0.3
                
                aligned_timings.append({
                    'word': original_words[i],
                    'start': estimated_start,
                    'end': estimated_end,
                    'confidence': 0.5
                })
        
        return aligned_timings
    
    def _fallback_alignment(self, audio_path: str, text: str) -> List[Dict]:
        """Fallback alignment using audio analysis."""
        print("🎯 Using fallback alignment method...")
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path)
            duration = len(y) / sr
            
            # Simple energy-based segmentation
            words = text.split()
            hop_length = 512
            frame_time = hop_length / sr
            
            # Calculate energy
            energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            
            # Smooth energy
            window_size = int(0.1 / frame_time)  # 100ms window
            smoothed_energy = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
            
            # Find speech segments
            threshold = np.mean(smoothed_energy) * 0.3
            speech_frames = smoothed_energy > threshold
            
            # Convert to time segments
            speech_times = []
            in_speech = False
            start_time = 0
            
            for i, is_speech in enumerate(speech_frames):
                time = i * frame_time
                if is_speech and not in_speech:
                    start_time = time
                    in_speech = True
                elif not is_speech and in_speech:
                    speech_times.append((start_time, time))
                    in_speech = False
            
            if in_speech:
                speech_times.append((start_time, duration))
            
            # Distribute words across speech segments
            word_timings = []
            words_per_segment = len(words) / max(len(speech_times), 1)
            
            word_idx = 0
            for seg_start, seg_end in speech_times:
                seg_duration = seg_end - seg_start
                words_in_segment = int(words_per_segment) + (1 if word_idx % 2 == 0 else 0)
                words_in_segment = min(words_in_segment, len(words) - word_idx)
                
                for i in range(words_in_segment):
                    if word_idx >= len(words):
                        break
                        
                    word_start = seg_start + (i / words_in_segment) * seg_duration
                    word_end = seg_start + ((i + 1) / words_in_segment) * seg_duration
                    
                    word_timings.append({
                        'word': words[word_idx],
                        'start': word_start,
                        'end': word_end,
                        'confidence': 0.7
                    })
                    word_idx += 1
            
            # Handle remaining words
            if word_idx < len(words):
                last_end = word_timings[-1]['end'] if word_timings else 0
                remaining_duration = duration - last_end
                remaining_words = len(words) - word_idx
                
                for i in range(remaining_words):
                    word_start = last_end + (i / remaining_words) * remaining_duration
                    word_end = last_end + ((i + 1) / remaining_words) * remaining_duration
                    
                    word_timings.append({
                        'word': words[word_idx + i],
                        'start': word_start,
                        'end': word_end,
                        'confidence': 0.6
                    })
            
            print(f"✅ Generated {len(word_timings)} fallback word timestamps")
            return word_timings
            
        except Exception as e:
            print(f"❌ Fallback alignment failed: {e}")
            return []

def create_subtitle_file(word_timings: List[Dict], output_path: str, format_type: str = "srt") -> str:
    """
    Create subtitle file from word timings.
    
    Args:
        word_timings: List of word timing dictionaries
        output_path: Output file path
        format_type: Format type ("srt", "vtt", "json")
    """
    try:
        if format_type.lower() == "srt":
            return _create_srt_file(word_timings, output_path)
        elif format_type.lower() == "vtt":
            return _create_vtt_file(word_timings, output_path)
        else:
            return _create_json_file(word_timings, output_path)
    except Exception as e:
        print(f"❌ Failed to create subtitle file: {e}")
        return ""

def _create_srt_file(word_timings: List[Dict], output_path: str) -> str:
    """Create SRT subtitle file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, word_info in enumerate(word_timings, 1):
            start_time = _format_srt_time(word_info['start'])
            end_time = _format_srt_time(word_info['end'])
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{word_info['word']}\n\n")
    
    print(f"✅ Created SRT file: {output_path}")
    return output_path

def _create_vtt_file(word_timings: List[Dict], output_path: str) -> str:
    """Create VTT subtitle file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        
        for word_info in word_timings:
            start_time = _format_vtt_time(word_info['start'])
            end_time = _format_vtt_time(word_info['end'])
            
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{word_info['word']}\n\n")
    
    print(f"✅ Created VTT file: {output_path}")
    return output_path

def _create_json_file(word_timings: List[Dict], output_path: str) -> str:
    """Create JSON subtitle file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(word_timings, f, indent=2)
    
    print(f"✅ Created JSON file: {output_path}")
    return output_path

def _format_srt_time(seconds: float) -> str:
    """Format time for SRT format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def _format_vtt_time(seconds: float) -> str:
    """Format time for VTT format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
