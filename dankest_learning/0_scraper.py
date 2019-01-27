import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from bs4 import BeautifulSoup
from tqdm import tqdm


def scrape_template(progress, template_href, path, meme_name, num_captions):
    for k in range(num_captions):
        r = requests.get(
            f'https://memegenerator.net{template_href}/images/popular/alltime/page/{k + 1}')
        soup = BeautifulSoup(r.text, 'html.parser')
        captions = soup.find_all(class_='char-img')
        with open(f'{path.joinpath(meme_name)}.txt', 'a') as f:
            for c in captions:
                top_text = c.find('div', class_="optimized-instance-text0").text
                bottom_text = c.find('div', class_="optimized-instance-text1").text
                f.write(f'{top_text} <SEP> {bottom_text}\n')
                progress.update()


def scrape_memes(save_dir, num_templates, num_captions):
    path = Path(save_dir)
    path.mkdir(exist_ok=True, parents=True)
    progress = tqdm(total=num_captions * 15 * num_templates * 15, desc="Memes downloaded")
    with ThreadPoolExecutor() as pool:
        for i in range(num_templates):
            url = f'https://memegenerator.net/memes/popular/alltime/page/{i + 1}'

            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            templates = soup.find_all(class_='char-img')
            templates_page = [t.find('a') for t in templates]
            for p in templates_page:
                template_href = p['href']
                template_img = p.find('img')
                template_img_url = template_img['src']
                template_verbose_name = template_img['alt']
                file_name = Path(template_img_url).stem
                try:
                    with requests.get(template_img_url, stream=True) as r:  # Streaming request for images
                        Image.open(BytesIO(r.content)).save(f'{path.joinpath(file_name)}.png')
                    pool.submit(scrape_template, progress, template_href, path, file_name, num_captions)
                except OSError:  # Some images are bugged :(
                    pass


def main():
    parser = argparse.ArgumentParser(description='Scrapes memegenerator.net for the most popular meme formats. ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--save_dir', default='../memes', type=str,
                        help='Directory to save memes in.')
    parser.add_argument('-t', '--num_templates', type=int, default=200,
                        help='Number of different templates (formats) to download times 15.')
    parser.add_argument('-c', '--num_captions', type=int, default=100,
                        help='Number of captions to download per template times 15.')
    # parser.add_argument('-o', '--overwrite', type=bool, default=True,
    #                     help='Overwrite current save_dir.')
    args = parser.parse_args()

    # if args.overwrite:
    #     Path(args.save_dir).rmtree()

    scrape_memes(args.save_dir, args.num_templates, args.num_captions)


if __name__ == '__main__':
    main()
