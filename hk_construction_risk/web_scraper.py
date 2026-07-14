"""从香港劳工处网站，安全警示部分获取工作安全警示标题"""

import requests
import time
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

max_retries = 3
for attempt in range(max_retries):
    try:
        response = requests.get(
            "https://www.labour.gov.hk/eng/news/work_safety_alert_2024.htm", #分别收集2012-2024年的标题
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        break
    except requests.exceptions.RequestException as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(2)
        else:
            print("All retries failed. Exiting.")
            exit()

soup = BeautifulSoup(response.text, "html.parser")

for a in soup.find_all('a', class_='externalUrl'):
    title = a.get_text(strip=True)
    title = title.replace('This link will open in a new window', '').strip()
    if title:
        print(title)
