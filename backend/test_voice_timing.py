#!/usr/bin/env python3
"""
Test script to compare male vs female voice timing differences in Edge TTS
"""

import edge_tts
import asyncio
import os
from moviepy.editor import AudioFileClip

async def test_voice_timing():
    """Test timing differences between male and female voices."""
    
    # Test text - same content for both voices
    test_text = """
    This is a test to see if there are timing differences between male and female voices 
    in Edge TTS when using the same speech rate. We want to check if the male voice 
    actually speaks slower or faster than the female voice at the same rate setting.
    """
    
    # Voice configurations
    voices_to_test = {
        "female": "en-US-AriaNeural",
        "male": "en-US-GuyNeural"
    }
    
    speech_rates = ["1.0", "1.5", "2.0"]  # Test different rates
    
    results = {}
    
    for rate in speech_rates:
        print(f"\n🎯 Testing speech rate: {rate}x")
        print("=" * 40)
        
        rate_results = {}
        
        for gender, voice in voices_to_test.items():
            output_file = f"test_{gender}_rate_{rate}.wav"
            
            try:
                # Generate TTS with specific rate
                rate_percentage = f"+{int((float(rate) - 1.0) * 100)}%"
                communicate = edge_tts.Communicate(test_text, voice, rate=rate_percentage)
                await communicate.save(output_file)
                
                # Measure actual duration
                audio = AudioFileClip(output_file)
                duration = audio.duration
                audio.close()
                
                # Calculate words per second
                word_count = len(test_text.split())
                wps = word_count / duration
                
                rate_results[gender] = {
                    'duration': duration,
                    'words_per_second': wps,
                    'voice': voice,
                    'file': output_file
                }
                
                print(f"🎤 {gender.capitalize()} ({voice}):")
                print(f"   Duration: {duration:.2f}s")
                print(f"   Words/sec: {wps:.2f}")
                
                # Clean up test file
                if os.path.exists(output_file):
                    os.remove(output_file)
                    
            except Exception as e:
                print(f"❌ Error testing {gender} voice: {e}")
        
        # Calculate difference
        if 'male' in rate_results and 'female' in rate_results:
            male_duration = rate_results['male']['duration']
            female_duration = rate_results['female']['duration']
            
            duration_diff = male_duration - female_duration
            percent_diff = (duration_diff / female_duration) * 100
            
            print(f"\n📊 Analysis for {rate}x speed:")
            print(f"   Male duration: {male_duration:.2f}s")
            print(f"   Female duration: {female_duration:.2f}s")
            print(f"   Difference: {duration_diff:+.2f}s ({percent_diff:+.1f}%)")
            
            if abs(percent_diff) > 5:  # More than 5% difference
                slower_voice = "male" if duration_diff > 0 else "female"
                print(f"   ⚠️  {slower_voice.capitalize()} voice is significantly slower!")
            else:
                print(f"   ✅ Voices have similar timing")
        
        results[rate] = rate_results
    
    # Summary
    print(f"\n" + "=" * 50)
    print("VOICE TIMING ANALYSIS SUMMARY")
    print("=" * 50)
    
    for rate in speech_rates:
        if rate in results and 'male' in results[rate] and 'female' in results[rate]:
            male_dur = results[rate]['male']['duration']
            female_dur = results[rate]['female']['duration']
            diff_percent = ((male_dur - female_dur) / female_dur) * 100
            
            print(f"Rate {rate}x: Male voice is {diff_percent:+.1f}% {'slower' if diff_percent > 0 else 'faster'}")
    
    return results

if __name__ == "__main__":
    print("🔬 Voice Timing Analysis")
    print("Testing male vs female voice timing differences...")
    
    results = asyncio.run(test_voice_timing())
