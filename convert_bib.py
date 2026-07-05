import re

def format_authors(author_str):
    author_str = author_str.strip()
    if author_str.endswith('.'):
        author_str = author_str[:-1]
        
    parts = re.split(r',\s+and\s+|\s+and\s+|,\s+', author_str)
    formatted_parts = []
    
    for p in parts:
        p = p.strip()
        if not p: continue
        # Replace ~ with space for easier parsing
        p_clean = p.replace('~', ' ')
        m = re.match(r'^(.*?)\s+([^\s]+)$', p_clean)
        if m:
            first = m.group(1).strip()
            last = m.group(2).strip()
            formatted = f"{last}, {first}"
        else:
            formatted = p_clean
        formatted_parts.append(formatted)
        
    if len(formatted_parts) == 1:
        return formatted_parts[0]
    elif len(formatted_parts) == 2:
        return f"{formatted_parts[0]}; and {formatted_parts[1]}"
    else:
        return "; ".join(formatted_parts[:-1]) + f"; and {formatted_parts[-1]}"

def extract_last_name(author_str):
    author_str = author_str.replace('~', ' ').strip()
    m = re.match(r'^(.*?)\s+([^\s]+)$', author_str)
    if m:
        return m.group(2).replace('.', '')
    return author_str.replace('.', '')

with open('final (1).tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.strip() == '\\begin{thebibliography}{10}':
        start_idx = i
    elif line.strip() == '\\end{thebibliography}':
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    bib_text = ''.join(lines[start_idx+1:end_idx])
    entries = re.split(r'\\bibitem\{([^}]+)\}', bib_text)[1:]
    
    new_lines = []
    
    for i in range(0, len(entries), 2):
        key = entries[i]
        text = entries[i+1]
        
        text = text.replace('\\newblock', ' ')
        text_clean = re.sub(r'\s+', ' ', text).strip()
        
        year_match = re.search(r'\b(19\d\d|20\d\d)\b', text_clean)
        year = year_match.group(1) if year_match else ""
        
        lines_text = [l.strip() for l in text.strip().split('\n') if l.strip()]
        original_author_line = lines_text[0]
        
        rest = " ".join(lines_text[1:])
        rest = rest.replace('\\newblock', ' ')
        rest = re.sub(r'\s+', ' ', rest).strip()
        
        if year:
            rest = re.sub(r',?\s*' + year + r'\.$', '.', rest)
            rest = re.sub(r',?\s*' + year + r'$', '.', rest)
        
        if not rest.endswith('.'):
            rest += '.'
            
        authors_formatted = format_authors(original_author_line)
        
        author_parts = re.split(r',\s+and\s+|\s+and\s+|,\s+', original_author_line)
        if len(author_parts) > 2:
            cite_author = f"{extract_last_name(author_parts[0])} \\bgroup et al.\\egroup "
        elif len(author_parts) == 2:
            cite_author = f"{extract_last_name(author_parts[0])} and {extract_last_name(author_parts[1])}"
        else:
            cite_author = extract_last_name(author_parts[0])

        new_bibitem = f"\\bibitem[\\protect\\citeauthoryear{{{cite_author}}}{{{year}}}]{{{key}}}\n{authors_formatted}. {year}. \n\\newblock {rest}\n"
        new_lines.append(new_bibitem)
        
    with open('new_bib.tex', 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(new_lines))
    print("Successfully wrote new_bib.tex")
