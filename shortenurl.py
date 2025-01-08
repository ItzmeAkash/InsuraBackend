import pyshorteners

# URL to shorten
long_url = "https://chatgpt.com"

# Initialize the shortener
shortener = pyshorteners.Shortener()

# Shorten the URL
short_url = shortener.tinyurl.short(long_url)

print(f"Shortened URL: {short_url}")
