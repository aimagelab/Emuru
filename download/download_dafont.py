import requests
from pathlib import Path
from bs4 import BeautifulSoup
import string
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def download_fonts_urls(char='a', page=1, fpp=200):
    url = f'https://www.dafont.com/it/alpha.php?lettre={char}&fpp={fpp}&page={page}'
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=header)
    soup = BeautifulSoup(response.text, 'html.parser')

    categories = []
    for el in soup.find_all('div', {'class': 'lv1right'}):
        cats = el.find_all('a')
        assert len(cats) == 2
        categories.append(f'{cats[0].text}/{cats[1].text}'.replace(' ', '_'))

    download_urls = []
    for link in soup.find_all('a', {'class': 'dl'}):
        href = link.get('href')
        if href and href.startswith('//dl.dafont.com/dl/?f='):
            font_name = href.split('?f=')[-1] + '.zip'
            font_url = 'https:' + href
            download_urls.append((font_name, font_url))
    
    assert len(categories) == len(download_urls)
    return [(cat, fname, furl) for cat, (fname, furl) in zip(categories, download_urls)]

def download(url, path):
    try:
        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=header)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        print(f'Error downloading {url}: {e}')

folder = Path('/home/vpippi/Emuru/files/font_square/new_fonts/dafont')
folder.mkdir(parents=True, exist_ok=True)

for char in list(string.ascii_lowercase) + ['%23']:
    download_urls = []
    page = 1
    pbar = tqdm(desc=f'Collecting urls for char "{char}"', unit='urls')
    while True:
        res = download_fonts_urls(char=char, page=page)
        download_urls.extend(res)
        pbar.update(len(res))
        if len(res) < 200:
            break
        page += 1
    pbar.close()
    
    with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for cat, fname, furl in download_urls:
                path = folder / cat / fname
                if not path.exists():
                    futures.append(executor.submit(download, furl, path))

            for future in tqdm(futures, desc='Downloading', unit='font'):
                future.result()
                

