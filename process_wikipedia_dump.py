# !/usr/bin/env python3
"""
Convert Wikipedia XML dump to TSV format for SEAL FM-index building.
This script processes Wikipedia dumps and outputs TSV files suitable for SEAL.
"""

import xml.etree.ElementTree as ET
import bz2
import re
import sys
import argparse
from tqdm import tqdm
import mwparserfromhell
import ftfy


def clean_wikitext(text):
    """Remove wiki markup and clean text."""
    try:
        # Parse wikitext
        wikicode = mwparserfromhell.parse(text)

        # Remove templates, tables, and other markup
        plain_text = wikicode.strip_code()

        # Clean up whitespace
        plain_text = re.sub(r'\s+', ' ', plain_text)

        # Fix text encoding issues
        plain_text = ftfy.fix_text(plain_text)

        # Remove section markers that SEAL strips anyway
        plain_text = plain_text.replace('BULLET::::', '')
        plain_text = plain_text.replace('SECTION::::', '')

        return plain_text.strip()
    except:
        # If parsing fails, do basic cleanup
        text = re.sub(r'\{\{[^}]+\}\}', '', text)  # Remove templates
        text = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', text)  # Clean links
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def process_wikipedia_dump(input_file, output_file, max_articles=None):
    """
    Process Wikipedia XML dump and convert to TSV format.

    Args:
        input_file: Path to Wikipedia XML dump (can be .bz2)
        output_file: Path to output TSV file
        max_articles: Maximum number of articles to process (None for all)
    """

    # Open input file (handle bz2 compression)
    if input_file.endswith('.bz2'):
        f = bz2.BZ2File(input_file, 'r')
    else:
        f = open(input_file, 'rb')

    # Open output file
    out = open(output_file, 'w', encoding='utf-8')

    # Parse XML iteratively
    article_count = 0
    redirect_count = 0

    # Create iterator
    context = ET.iterparse(f, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)

    # Get namespace from root element
    namespace = re.match(r'\{.*\}', root.tag).group() if root.tag.startswith('{') else ''

    for event, elem in tqdm(context, desc="Processing articles"):
        if event == 'end' and elem.tag == f'{namespace}page':
            # Extract page data
            title_elem = elem.find(f'{namespace}title')
            text_elem = elem.find(f'{namespace}revision/{namespace}text')
            id_elem = elem.find(f'{namespace}id')
            redirect_elem = elem.find(f'{namespace}redirect')

            if title_elem is not None and text_elem is not None and id_elem is not None:
                title = title_elem.text
                page_id = id_elem.text

                # Skip redirects
                if redirect_elem is not None:
                    redirect_count += 1
                    elem.clear()
                    root.clear()
                    continue

                # Skip non-article namespaces (User:, Talk:, etc.)
                if ':' in title and not title.startswith('Category:'):
                    elem.clear()
                    root.clear()
                    continue

                text = text_elem.text or ''

                # Clean the text
                cleaned_text = clean_wikitext(text)

                # Skip very short articles
                if len(cleaned_text) < 100:
                    elem.clear()
                    root.clear()
                    continue

                # Write to TSV: id, title, text
                # Make sure no tabs in the content
                title = title.replace('\t', ' ')
                cleaned_text = cleaned_text.replace('\t', ' ')

                out.write(f"{page_id}\t{title}\t{cleaned_text}\n")

                article_count += 1

                if max_articles and article_count >= max_articles:
                    break

            # Clear memory
            elem.clear()
            root.clear()

    f.close()
    out.close()

    print(f"\nProcessed {article_count} articles")
    print(f"Skipped {redirect_count} redirects")


def split_tsv_file(input_file, chunk_size=1000000):
    """Split large TSV file into smaller chunks for parallel processing."""
    with open(input_file, 'r', encoding='utf-8') as f:
        chunk_num = 0
        current_chunk = []

        for line in tqdm(f, desc="Splitting file"):
            current_chunk.append(line)

            if len(current_chunk) >= chunk_size:
                output_file = f"{input_file}.chunk{chunk_num:04d}"
                with open(output_file, 'w', encoding='utf-8') as out:
                    out.writelines(current_chunk)
                print(f"Written chunk {chunk_num} with {len(current_chunk)} articles")
                current_chunk = []
                chunk_num += 1

        # Write remaining articles
        if current_chunk:
            output_file = f"{input_file}.chunk{chunk_num:04d}"
            with open(output_file, 'w', encoding='utf-8') as out:
                out.writelines(current_chunk)
            print(f"Written final chunk {chunk_num} with {len(current_chunk)} articles")


def main():
    parser = argparse.ArgumentParser(description='Convert Wikipedia XML dump to TSV for SEAL')
    parser.add_argument('input', help='Input Wikipedia XML dump file (can be .bz2)')
    parser.add_argument('output', help='Output TSV file')
    parser.add_argument('--max-articles', type=int, default=None,
                        help='Maximum number of articles to process')
    parser.add_argument('--split', action='store_true',
                        help='Split output into chunks after processing')
    parser.add_argument('--chunk-size', type=int, default=1000000,
                        help='Number of articles per chunk when splitting')

    args = parser.parse_args()

    # Install required packages if needed
    try:
        import mwparserfromhell
        import ftfy
        from tqdm import tqdm
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "mwparserfromhell", "ftfy", "tqdm"])
        import mwparserfromhell
        import ftfy
        from tqdm import tqdm

    # Process the dump
    process_wikipedia_dump(args.input, args.output, args.max_articles)

    # Optionally split the output
    if args.split:
        split_tsv_file(args.output, args.chunk_size)


if __name__ == '__main__':
    main()
