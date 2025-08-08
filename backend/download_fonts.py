import requests
import os

def download_google_fonts():
    """Download Poppins Bold font for text styling."""
    fonts_dir = "fonts"
    os.makedirs(fonts_dir, exist_ok=True)
    
    font_name = "Poppins-Bold.ttf"
    font_url = "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf"
    font_path = os.path.join(fonts_dir, font_name)
    
    if os.path.exists(font_path):
        print(f"✅ {font_name} already exists")
        return
        
    try:
        print(f"📥 Downloading {font_name}...")
        response = requests.get(font_url, timeout=30)
        response.raise_for_status()
        
        with open(font_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded {font_name}")
        
    except Exception as e:
        print(f"❌ Failed to download {font_name}: {e}")

if __name__ == "__main__":
    download_google_fonts()
