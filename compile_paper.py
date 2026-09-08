import os
import re
import base64
import subprocess
import markdown

WORKSPACE_DIR = r"c:\Users\Ronak Daniel\Downloads\RL_PAPER"
INPUT_MD = os.path.join(WORKSPACE_DIR, "paper_draft.md")
OUTPUT_HTML = os.path.join(WORKSPACE_DIR, "paper.html")
OUTPUT_PDF = os.path.join(WORKSPACE_DIR, "paper.pdf")

def read_markdown(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def convert_images_to_base64(html_content, base_dir):
    # Regex to find img src="..."
    def replacer(match):
        rel_path = match.group(1)
        # Check if file exists
        full_path = os.path.normpath(os.path.join(base_dir, rel_path))
        if os.path.exists(full_path):
            with open(full_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode("utf-8")
                ext = os.path.splitext(full_path)[1].lower().replace(".", "")
                mime = "image/png" if ext == "png" else "image/jpeg"
                return f'<img src="data:{mime};base64,{encoded}"'
        return match.group(0)
    
    return re.sub(r'<img\s+src="([^"]+)"', replacer, html_content)

def build_html():
    md_content = read_markdown(INPUT_MD)
    
    # Pre-process math so markdown parser doesn't mangle underscores or asterisks in math blocks
    math_blocks = []
    def save_display_math(match):
        idx = len(math_blocks)
        math_blocks.append(match.group(0))
        return f"MATHBLOCKDISPLAY{idx}END"
    
    def save_inline_math(match):
        idx = len(math_blocks)
        math_blocks.append(match.group(0))
        return f"MATHBLOCKINLINE{idx}END"
    
    # Protect display math $$ ... $$
    protected_md = re.sub(r'\$\$[\s\S]*?\$\$', save_display_math, md_content)
    # Protect inline math $ ... $
    protected_md = re.sub(r'\$([^\$\n]+?)\$', save_inline_math, protected_md)
    
    # Convert markdown to HTML
    html_body = markdown.markdown(protected_md, extensions=['tables', 'fenced_code'])
    
    # Restore math
    for idx, math in enumerate(math_blocks):
        html_body = html_body.replace(f"MATHBLOCKDISPLAY{idx}END", math)
        html_body = html_body.replace(f"MATHBLOCKINLINE{idx}END", math)
    
    # Replace relative images with base64 data URIs
    html_body = convert_images_to_base64(html_body, WORKSPACE_DIR)
    
    # Styled template
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Computational Analogues of Post-Decision and Self-Evaluative Biases in Reinforcement Learning Agents</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  svg: {{
    fontCache: 'global'
  }}
}};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>
<style>
@page {{
    size: letter;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-right {{
        content: counter(page);
    }}
}}

body {{
    font-family: "Charter", "Georgia", "Cambria", serif;
    color: #1a1a1a;
    line-height: 1.6;
    font-size: 11pt;
    max-width: 900px;
    margin: 0 auto;
    padding: 30px;
    background-color: #ffffff;
}}

h1 {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 24pt;
    font-weight: 700;
    line-height: 1.25;
    color: #0f172a;
    text-align: center;
    margin-bottom: 8px;
}}

.authors-block {{
    text-align: center;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 11pt;
    color: #475569;
    margin-bottom: 25px;
}}

.authors-block strong {{
    color: #1e293b;
    font-size: 12pt;
}}

.venue-tag {{
    display: inline-block;
    background: #e2e8f0;
    color: #334155;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 9.5pt;
    font-weight: 600;
    margin-top: 6px;
}}

hr {{
    border: 0;
    height: 1px;
    background: #cbd5e1;
    margin: 25px 0;
}}

h2 {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 15pt;
    font-weight: 700;
    color: #1e293b;
    border-bottom: 1.5px solid #0284c7;
    padding-bottom: 4px;
    margin-top: 30px;
    margin-bottom: 12px;
}}

h3 {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 12pt;
    font-weight: 600;
    color: #334155;
    margin-top: 20px;
    margin-bottom: 8px;
}}

h4 {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 11pt;
    font-weight: 600;
    color: #475569;
    margin-top: 14px;
    margin-bottom: 6px;
}}

p {{
    margin-bottom: 12px;
    text-align: justify;
    hyphens: auto;
}}

blockquote {{
    border-left: 4px solid #0284c7;
    margin: 15px 0;
    padding: 10px 18px;
    background-color: #f0f9ff;
    color: #0369a1;
    font-style: italic;
}}

ul, ol {{
    margin-top: 6px;
    margin-bottom: 14px;
    padding-left: 24px;
}}

li {{
    margin-bottom: 5px;
}}

pre, code {{
    font-family: "Consolas", "Courier New", monospace;
    font-size: 9.5pt;
}}

pre {{
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 12px;
    overflow-x: auto;
    margin-bottom: 15px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0 10px 0;
    font-size: 9.5pt;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}

th, td {{
    padding: 8px 10px;
    text-align: left;
    border-bottom: 1px solid #cbd5e1;
}}

th {{
    background-color: #f1f5f9;
    font-weight: 600;
    color: #1e293b;
    border-top: 2px solid #334155;
    border-bottom: 2px solid #334155;
}}

tr:last-child td {{
    border-bottom: 2px solid #334155;
}}

img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 16px auto 8px auto;
    border-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}}

em {{
    color: #334155;
}}

.figure-caption {{
    font-size: 9pt;
    color: #475569;
    text-align: center;
    margin-bottom: 20px;
    line-height: 1.4;
}}

@media print {{
    body {{
        padding: 0;
        max-width: 100%;
        font-size: 10pt;
    }}
    h1 {{ font-size: 20pt; }}
    h2 {{ font-size: 13pt; page-break-after: avoid; }}
    h3 {{ font-size: 11pt; page-break-after: avoid; }}
    table, figure, img {{
        page-break-inside: avoid;
    }}
    .no-print {{
        display: none;
    }}
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Generated HTML: {OUTPUT_HTML} ({len(html_doc)} bytes)")

def compile_pdf():
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not os.path.exists(edge_path):
        print("Edge not found at standard path.")
        return False
    
    cmd = [
        edge_path,
        "--headless=new",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=5000",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUTPUT_PDF}",
        OUTPUT_HTML
    ]
    
    print(f"Compiling PDF using Edge headless...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(OUTPUT_PDF) and os.path.getsize(OUTPUT_PDF) > 1000:
        print(f"Successfully compiled PDF: {OUTPUT_PDF} ({os.path.getsize(OUTPUT_PDF)} bytes)")
        return True
    else:
        print(f"PDF compilation failed or output file is empty: {res.stderr}")
        return False

if __name__ == "__main__":
    build_html()
    compile_pdf()
