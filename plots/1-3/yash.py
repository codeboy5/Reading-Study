import requests
from bs4 import BeautifulSoup


def trade_spider(max_pages):
    page = 1
    while(page <= max_pages):
        url = "http://books.toscrape.com/catalogue/page-" + str(page) + ".html"
        source_code = requests.get(url)
        plain_text = source_code.text
        soup = BeautifulSoup(plain_text, 'html.parser')
        for link in soup.findAll('p', {'class': 'star-rating Five'}):
            print(link)
        page += 1


trade_spider(1)
