#!/usr/bin/env python3
"""
Test script for AI-Enhanced Story Generator
"""

import sys
import os
from dotenv import load_dotenv

def test_imports():
    """Test all required imports for AI functionality."""
    # Load environment variables first
    load_dotenv()
    
    print("🧪 Testing AI-Enhanced Story Generator Setup...")
    print("=" * 50)
    
    # Test basic imports
    try:
        import praw
        print("✅ PRAW (Reddit API): Available")
    except ImportError as e:
        print(f"❌ PRAW: {e}")
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI: Available")
        
        # Test Google Gemini API configuration
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if api_key and api_key.startswith("AIza"):
            print(f"✅ Google Gemini API Key: Configured")
            ai_available = True
        else:
            print("❌ Google Gemini API Key: Not configured or invalid")
            ai_available = False
            
    except ImportError as e:
        print(f"❌ Google Generative AI: {e}")
        ai_available = False
    
    try:
        from ai_story_generator import AIStoryGenerator
        print("✅ AI Story Generator: Available")
    except ImportError as e:
        print(f"❌ AI Story Generator: {e}")
    
    # Test environment
    print("\n🔧 Environment Configuration:")
    print("=" * 50)
    
    env_file = ".env"
    if os.path.exists(env_file):
        print("✅ .env file: Found")
    else:
        print("❌ .env file: Not found (copy from .env.example)")
    
    # Test story modes
    print("\n🎯 Story Generation Modes:")
    print("=" * 50)
    
    if ai_available:
        print("✅ AI-Enhanced Mode: Available")
        print("   - Original story creation from Reddit inspiration")
        print("   - Multi-subreddit source blending") 
        print("   - Copyright-safe content generation")
        print("   - Using Google Gemini 1.5 Flash model")
    else:
        print("❌ AI-Enhanced Mode: Not Available")
        print("   - Install: pip install google-generativeai")
        print("   - Configure GOOGLE_API_KEY in .env")
        print("   - Get API key from: https://aistudio.google.com/app/apikey")
    
    print("✅ Direct Reddit Mode: Available")
    print("   - Direct Reddit post usage")
    print("   - Original app functionality")
    
    print("\n🎬 Video Generation:")
    print("=" * 50)
    
    background_video = "static/minecraft_background.mp4"
    if os.path.exists(background_video):
        print("✅ Background video: Found")
    else:
        print(f"❌ Background video: Missing ({background_video})")
    
    music_file = "static/music/the-journey-home-by-tokyo-music-walker.mp3"
    if os.path.exists(music_file):
        print("✅ Background music: Found")
    else:
        print(f"❌ Background music: Missing ({music_file})")
    
    print("\n" + "=" * 50)
    print("🎯 SETUP SUMMARY:")
    print("=" * 50)
    
    if ai_available:
        print("🤖 AI-Enhanced Mode: READY")
        print("   Using Google Gemini 1.5 Flash model")
    else:
        print("📱 Reddit Direct Mode: READY") 
        print("   Install google-generativeai and configure Google Gemini API for AI features")
    
    print("\nTo get started:")
    print("1. Configure .env file with Reddit API credentials")
    print("2. (Optional) Add Google Gemini API key for AI mode")
    print("3. Run: python app.py")

if __name__ == "__main__":
    test_imports()
