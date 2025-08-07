import requests
import os

def download_google_fonts():
    """Download popular Google Fonts for better text styling."""
    fonts_dir = "fonts"
    os.makedirs(fonts_dir, exist_ok=True)
    
    # Google Fonts direct download URLs (these are static URLs for popular fonts)
    fonts_to_download = {
        "Oswald-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/oswald/Oswald-Bold.ttf",
        "Oswald-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/oswald/Oswald-Regular.ttf",
        "Montserrat-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Bold.ttf",
        "Montserrat-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Regular.ttf",
        "Inter-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/inter/Inter-Bold.ttf",
        "Poppins-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf"
    }
    
    for font_name, url in fonts_to_download.items():
        font_path = os.path.join(fonts_dir, font_name)
        
        if os.path.exists(font_path):
            print(f"✅ {font_name} already exists")
            continue
            
        try:
            print(f"📥 Downloading {font_name}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(font_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded {font_name}")
            
        except Exception as e:
            print(f"❌ Failed to download {font_name}: {e}")

if __name__ == "__main__":
    download_google_fonts()
