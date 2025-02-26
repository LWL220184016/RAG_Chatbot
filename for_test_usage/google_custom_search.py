import os
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup

# Environment variables for sensitive data
api_key = os.environ.get('GOOGLE_CUSTOM_SEARCH_API_KEY')
search_engine_id = os.environ.get('SEARCH_ENGINE_ID_cx')

# Build a service object for interacting with the API:
service = build("customsearch", "v1", developerKey=api_key)

def google_search(query, **kwargs):
    # Execute the search request
    res = service.cse().list(q=query, cx=search_engine_id, **kwargs).execute()
    return res.get('items', [])

def fetch_page_content(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        page_content = ' '.join([para.get_text() for para in paragraphs])
        
        # Save content to a file
        os.makedirs('fetch_from_internet', exist_ok=True)
        with open(f'fetch_from_internet/{filename}', 'w', encoding='utf-8') as file:
            file.write("url: " + url + "\n\n" + page_content)
        
        return page_content
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

# Example search
search_query = "2020 news"
results = google_search(search_query)

# Save results to individual text files
for i, result in enumerate(results):
    filename = f"result_{i+1}.txt"
    content = fetch_page_content(result['link'], filename)
    if content:
        print(f"Content saved to fetch_from_internet/{filename}")
    else:
        print(f"Failed to save content for {result['link']}")