"""Normalize excessive Arial superscript rise in saved SVG figure exports.

Only simple superscript glyph y-coordinates change. Curves, text content, font
sizes, axes and canvas dimensions are preserved. No analysis is rerun.
"""
from collections import Counter
from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess

from lxml import etree as ET
from PIL import Image

SVG = 'http://www.w3.org/2000/svg'
NS = {'s': SVG}
TARGET_RISE_EM = .34


def excessive_superscripts(tree):
    findings = []
    for text in tree.findall('.//s:text', NS):
        glyphs = []
        for span in text.findall('s:tspan', NS):
            size = re.search(r'font-size:\s*([\d.]+)px', span.get('style', ''))
            try:
                glyphs.append((span, float(size[1]), float(span.get('y'))))
            except (TypeError, ValueError):
                continue
        if not glyphs:
            continue
        base = max(g[1] for g in glyphs)
        baseline = Counter(g[2] for g in glyphs if g[1] == base).most_common(1)[0][0]
        for span, size, y in glyphs:
            if (.60 < size / base < .80 and (baseline - y) / base > .50
                    and re.fullmatch(r'[\d+\-\u2212]+', span.text or '')):
                findings.append((span, baseline - TARGET_RISE_EM * base))
    return findings


def normalize_tree(tree):
    changed = []
    for span, y in excessive_superscripts(tree):
        changed.append({'glyph': span.text, 'old_y': float(span.get('y')), 'new_y': y})
        span.set('y', f'{y:.6f}')
    return changed


def inkscape_executable():
    binary = shutil.which('inkscape') or shutil.which('inkscape.com')
    if binary:
        return binary
    windows = Path('C:/Program Files/Inkscape/bin/inkscape.com')
    if windows.is_file():
        return str(windows)
    raise FileNotFoundError('Inkscape is required to render the corrected SVG exports.')


def render_companions(path):
    binary = inkscape_executable()
    png = path.with_suffix('.png')
    with Image.open(png) as image:
        width, height = image.size
    for ext in ['png', 'pdf']:
        command = [binary, str(path), '--export-area-page', '--export-background=white',
                   '--export-background-opacity=1', '--export-filename='+str(path.with_suffix('.'+ext))]
        if ext == 'png':
            command.extend([f'--export-width={width}', f'--export-height={height}'])
        else:
            command.append('--export-text-to-path')
        subprocess.run(command, check=True, capture_output=True, timeout=180)
    with Image.open(png) as image:
        assert image.size == (width, height)


def normalize_exports(folder, apply=False):
    records = []
    for path in sorted(Path(folder).rglob('*.svg')):
        tree = ET.parse(str(path))
        changed = normalize_tree(tree)
        if not changed:
            continue
        if apply:
            inkscape_executable()
            assert path.with_suffix('.png').is_file(), path
            tree.write(str(path), encoding='utf-8', xml_declaration=True)
            render_companions(path)
        records.append({'svg': str(path), 'glyphs': changed})
    # Keep the combined atlas consistent with its individual page exports.
    atlas = Path(folder)/'figures/si_candidates/six_weighted_descriptor_pairs'
    if apply and any(Path(r['svg']).parent == atlas for r in records):
        from pypdf import PdfWriter
        writer = PdfWriter()
        pages = sorted(atlas.glob('[0-9][0-9]_*.pdf'))
        assert len(pages) == 15
        for path in pages:
            writer.append(path)
        with (atlas/'all_15_pairs.pdf').open('wb') as handle:
            writer.write(handle)
        writer.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', type=Path, nargs='+')
    parser.add_argument('--apply', action='store_true', help='Correct SVG and re-export existing PNG/PDF companions.')
    args = parser.parse_args()
    records = [row for folder in args.folders for row in normalize_exports(folder, apply=args.apply)]
    print(json.dumps({'applied': args.apply, 'figures': records}, indent=2))
    if records and not args.apply:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
