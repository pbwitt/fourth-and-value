"""Consistent metadata for original articles; preserve their public URLs and bodies."""
import argparse
import html
import json
from pathlib import Path
import re

BASE='https://fourthandvalue.com'

def metadata(item):
    url=BASE+item['url'];title=item['title'];excerpt=item['excerpt']
    published=item.get('published_at') or item['date']
    modified=item.get('modified_at') or published
    schema={'@context':'https://schema.org','@type':'Article','headline':title,'description':excerpt,
        'url':url,'mainEntityOfPage':{'@type':'WebPage','@id':url},'datePublished':published,'dateModified':modified,
        'author':{'@type':'Organization','name':'Fourth & Value','url':BASE+'/'},
        'publisher':{'@type':'Organization','name':'Fourth & Value','url':BASE},
        'articleSection':item.get('sport','Sports'),'inLanguage':'en-US'}
    # Only use an actual story image, never pretend a logo illustrates the story.
    image=item.get('image')
    if image and (image.startswith('/') or image.startswith(BASE+'/')):schema['image']=BASE+image if image.startswith('/') else image
    esc=lambda s:html.escape(str(s),quote=True)
    tags=[f'<title>{esc(title)} | Fourth &amp; Value</title>',f'<meta name="description" content="{esc(excerpt)}">',
        f'<link rel="canonical" href="{esc(url)}">','<meta name="author" content="Fourth &amp; Value">',
        '<meta name="robots" content="index,follow,max-image-preview:large">']
    values={'og:title':title,'og:description':excerpt,'og:url':url,'og:type':'article','og:site_name':'Fourth & Value',
        'article:published_time':published,'article:modified_time':modified,'article:section':item.get('sport','Sports')}
    if 'image' in schema:values['og:image']=schema['image']
    tags += [f'<meta property="{key}" content="{esc(value)}">' for key,value in values.items()]
    tags += [f'<meta name="twitter:card" content="{"summary_large_image" if "image" in schema else "summary"}">',
        f'<meta name="twitter:title" content="{esc(title)}">',f'<meta name="twitter:description" content="{esc(excerpt)}">']
    if 'image' in schema:tags.append(f'<meta name="twitter:image" content="{esc(schema["image"])}">')
    encoded=json.dumps(schema,ensure_ascii=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    tags.append('<script type="application/ld+json">'+encoded+'</script>')
    return '\n'.join(tags)


def update_page(page,item):
    head,body=page.split('</head>',1)
    head=re.sub(r'<title>.*?</title>','',head,flags=re.S|re.I)
    head=re.sub(r'<link\b[^>]*\brel=["\']canonical["\'][^>]*>','',head,flags=re.I)
    head=re.sub(r'<meta\b[^>]*(?:name|property)=["\'](?:description|author|robots|og:[^"\']+|twitter:[^"\']+|article:[^"\']+)["\'][^>]*>','',head,flags=re.I)
    head=re.sub(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>.*?</script>','',head,flags=re.S|re.I)
    head=re.sub(r'\n\s*\n','\n',head).rstrip()
    return head+'\n'+metadata(item)+'</head>'+body


def backfill(root):
    root=Path(root);catalog=json.loads((root/'docs/editorial/published.json').read_text());changed=0
    for item in catalog:
        if not item.get('url','').startswith('/editorial/articles/') or item.get('kind')!='Analysis':continue
        path=root/'docs'/item['url'].lstrip('/')
        if not path.exists():continue
        old=path.read_text();new=update_page(old,item)
        if new!=old:path.write_text(new);changed+=1
    print(f'Updated article metadata on {changed} pages; article bodies and URLs preserved')

if __name__=='__main__':
    backfill(Path(__file__).resolve().parents[1])
