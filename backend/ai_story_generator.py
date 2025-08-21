"""
AI-powered story generation from Reddit content.
Transforms multiple Reddit stories into original, engaging narratives using Google Gemini.
"""

import os
import json
import random
import google.generativeai as genai
from typing import List, Dict, Optional
import re
from datetime import datetime

class AIStoryGenerator:
    """AI-powered story generator that transforms Reddit content into original narratives using Google Gemini."""
    
    def __init__(self):
        # Configure Google Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.transformation_prompt = """
Read the following Reddit stories. Your task is to combine creative elements, themes, and ideas from each post to generate a completely new, original story. Do not copy or closely paraphrase any of the original text. Instead, invent new characters, settings, and a unique plot that draws inspiration from all the input stories.

Requirements:
- Blend character traits, situations, or dilemmas from at least two stories
- Create a distinct narrative arc with its own beginning, middle, and end
- Add a twist or unexpected resolution not present in the originals
- Use clear, engaging language suitable for an audience aged 16–35
- Keep the story concise: aim for a length that fits a 30–60 second video script (roughly 120–200 words)
- Write in first person perspective for maximum engagement
- Include emotional stakes and relatable conflicts
- End with a compelling question or moral dilemma

Here are the source stories:
{source_stories}

Provide only the remixed story. Do not reference the original stories or include a title. Start directly with the story content.
"""
    
    def scrape_multiple_stories(self, reddit_client, subreddits: List[str], stories_per_sub: int = 3) -> List[Dict]:
        """
        Scrape multiple stories from different subreddits for AI transformation.
        
        Args:
            reddit_client: PRAW Reddit instance
            subreddits: List of subreddit names to scrape from
            stories_per_sub: Number of stories to fetch per subreddit
            
        Returns:
            List of story dictionaries with title, body, and subreddit info
        """
        print(f"🔍 Scraping stories from {len(subreddits)} subreddits...")
        
        all_stories = []
        
        for subreddit_name in subreddits:
            try:
                print(f"  📱 Fetching from r/{subreddit_name}...")
                subreddit = reddit_client.subreddit(subreddit_name)
                
                # Get hot posts with text content
                posts = []
                for post in subreddit.hot(limit=stories_per_sub * 3):  # Get extra to filter
                    if (post.selftext and 
                        len(post.selftext) > 100 and  # Minimum length
                        len(post.selftext) < 2000 and  # Maximum length for processing
                        not post.over_18 and  # Skip NSFW content
                        post.score > 10):  # Minimum engagement
                        
                        posts.append({
                            'title': post.title,
                            'body': post.selftext,
                            'subreddit': subreddit_name,
                            'score': post.score,
                            'url': post.url,
                            'word_count': len(post.selftext.split())
                        })
                        
                        if len(posts) >= stories_per_sub:
                            break
                
                all_stories.extend(posts)
                print(f"    ✅ Got {len(posts)} stories from r/{subreddit_name}")
                
            except Exception as e:
                print(f"    ❌ Error fetching from r/{subreddit_name}: {e}")
                continue
        
        print(f"📚 Total stories collected: {len(all_stories)}")
        return all_stories
    
    def select_source_stories(self, all_stories: List[Dict], count: int = 3) -> List[Dict]:
        """
        Select the best source stories for AI transformation.
        
        Args:
            all_stories: All scraped stories
            count: Number of stories to select (2-4 optimal for blending)
            
        Returns:
            Selected stories for transformation
        """
        if len(all_stories) <= count:
            return all_stories
        
        # Filter for quality stories
        quality_stories = []
        for story in all_stories:
            word_count = story['word_count']
            
            # Quality criteria: good length, engagement, and variety
            if (50 <= word_count <= 400 and  # Optimal length for processing
                story['score'] > 10 and       # Some engagement
                not any(banned in story['title'].lower() for banned in 
                       ['update', 'edit', 'meta', 'final', 'conclusion'])):  # Original posts only
                quality_stories.append(story)
        
        # If not enough quality stories, use all available
        if len(quality_stories) < count:
            quality_stories = all_stories
        
        # Sort by score and select diverse stories
        quality_stories.sort(key=lambda x: x['score'], reverse=True)
        
        # Select stories from different subreddits when possible for variety
        selected = []
        used_subreddits = set()
        
        # First pass: one story per subreddit
        for story in quality_stories:
            if story['subreddit'] not in used_subreddits and len(selected) < count:
                selected.append(story)
                used_subreddits.add(story['subreddit'])
        
        # Second pass: fill remaining slots with best stories
        for story in quality_stories:
            if len(selected) >= count:
                break
            if story not in selected:
                selected.append(story)
        
        # If still not enough, add random selection
        if len(selected) < count:
            remaining = [s for s in quality_stories if s not in selected]
            additional = random.sample(remaining, min(count - len(selected), len(remaining)))
            selected.extend(additional)
        
        print(f"🎯 Selected {len(selected)} source stories from subreddits: {[s['subreddit'] for s in selected]}")
        return selected
    
    def format_source_stories(self, stories: List[Dict]) -> str:
        """
        Format source stories for the AI prompt.
        
        Args:
            stories: Selected source stories
            
        Returns:
            Formatted string for AI prompt
        """
        formatted_stories = []
        
        for i, story in enumerate(stories, 1):
            # Truncate very long stories
            body = story['body']
            if len(body) > 800:
                body = body[:800] + "..."
            
            formatted_story = f"Story {i} (from r/{story['subreddit']}):\nTitle: {story['title']}\nContent: {body}"
            formatted_stories.append(formatted_story)
        
        return "\n\n---\n\n".join(formatted_stories)
    
    def generate_original_story(self, source_stories: List[Dict]) -> Optional[str]:
        """
        Generate an original story using AI based on source material.
        
        Args:
            source_stories: Source stories for inspiration
            
        Returns:
            Generated original story text
        """
        if not source_stories:
            print("❌ No source stories provided")
            return None
        
        print(f"🤖 Generating original story from {len(source_stories)} source stories...")
        
        # Format source stories for prompt
        source_text = self.format_source_stories(source_stories)
        
        # Create the full prompt
        full_prompt = self.transformation_prompt.format(source_stories=source_text)
        
        try:
            # Generate story using Google Gemini
            response = self.model.generate_content(full_prompt)
            
            if response.text:
                generated_story = response.text.strip()
            else:
                print("❌ Gemini returned empty response")
                return None
            
            # Clean up the story
            generated_story = self.clean_generated_story(generated_story)
            
            # Validate story length (aim for 120-200 words for video)
            word_count = len(generated_story.split())
            print(f"✅ Generated original story ({word_count} words)")
            
            if word_count < 80:
                print("⚠️ Story might be too short for video content")
            elif word_count > 250:
                print("⚠️ Story might be too long for video content")
            
            return generated_story
            
        except Exception as e:
            print(f"❌ AI generation failed: {e}")
            return None
    
    def clean_generated_story(self, story: str) -> str:
        """Clean and format the AI-generated story."""
        # Remove any meta-text about being AI generated
        story = re.sub(r'(as an ai|i\'m an ai|here\'s a story|here is a story)', '', story, flags=re.IGNORECASE)
        
        # Remove any quotation marks around the whole story
        story = story.strip('\'"')
        
        # Remove title if accidentally included
        lines = story.split('\n')
        if len(lines) > 1 and len(lines[0]) < 100 and ':' not in lines[0]:
            # First line might be a title, remove it
            story = '\n'.join(lines[1:]).strip()
        
        # Ensure it starts with a capital letter
        if story and not story[0].isupper():
            story = story[0].upper() + story[1:]
        
        # Clean up multiple spaces and newlines
        story = re.sub(r'\n+', '\n\n', story)
        story = re.sub(r' +', ' ', story)
        
        # Remove any reference to original stories
        story = re.sub(r'(inspired by|based on|from the stories above)', '', story, flags=re.IGNORECASE)
        
        return story.strip()
    
    def save_generation_log(self, source_stories: List[Dict], generated_story: str):
        """Save a log of the generation process for reference."""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'source_count': len(source_stories),
            'source_stories': [
                {
                    'title': s['title'][:100] + '...' if len(s['title']) > 100 else s['title'],
                    'subreddit': s['subreddit'],
                    'score': s['score'],
                    'word_count': s['word_count']
                } for s in source_stories
            ],
            'generated_story': generated_story,
            'generated_word_count': len(generated_story.split()),
            'subreddits_used': list(set(s['subreddit'] for s in source_stories))
        }
        
        log_file = 'ai_generation_log.json'
        
        # Load existing log or create new
        existing_log = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    existing_log = json.load(f)
            except:
                existing_log = []
        
        existing_log.append(log_data)
        
        # Keep only last 20 generations to prevent file from getting too large
        if len(existing_log) > 20:
            existing_log = existing_log[-20:]
        
        with open(log_file, 'w') as f:
            json.dump(existing_log, f, indent=2)
        
        print(f"📝 Generation logged to {log_file}")

def create_ai_enhanced_story(reddit_client, target_subreddits: List[str] = None) -> Optional[str]:
    """
    Main function to create an AI-enhanced original story.
    
    Args:
        reddit_client: PRAW Reddit instance
        target_subreddits: List of subreddits to scrape (uses defaults if None)
        
    Returns:
        Generated original story text
    """
    # Default subreddits for story inspiration
    if not target_subreddits:
        target_subreddits = [
            'AmItheAsshole',
            'relationship_advice', 
            'tifu',
            'offmychest',
            'entitledparents',
            'maliciouscompliance',
            'pettyrevenge',
            'choosingbeggars'
        ]
    
    generator = AIStoryGenerator()
    
    try:
        # Step 1: Scrape multiple stories
        all_stories = generator.scrape_multiple_stories(reddit_client, target_subreddits, stories_per_sub=3)
        
        if len(all_stories) < 2:
            print("❌ Need at least 2 stories for blending")
            return None
        
        # Step 2: Select best source stories (3-4 for optimal blending)
        story_count = min(4, max(2, len(all_stories)))
        source_stories = generator.select_source_stories(all_stories, count=story_count)
        
        if len(source_stories) < 2:
            print("❌ Not enough quality stories found")
            return None
        
        # Step 3: Generate original story
        generated_story = generator.generate_original_story(source_stories)
        
        if generated_story:
            # Step 4: Log the generation
            generator.save_generation_log(source_stories, generated_story)
            
            print("\n" + "="*60)
            print("🎉 AI-GENERATED ORIGINAL STORY:")
            print("="*60)
            print(generated_story)
            print("="*60)
            print(f"📊 Blended from {len(source_stories)} stories across subreddits:")
            for story in source_stories:
                print(f"   • r/{story['subreddit']}: {story['title'][:50]}...")
            print("="*60)
        
        return generated_story
        
    except Exception as e:
        print(f"❌ AI story generation failed: {e}")
        return None
