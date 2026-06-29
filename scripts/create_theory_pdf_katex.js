const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");
const katex = require("../tools/mathpdf/node_modules/katex");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..");
const source = path.join(root, "THEORY_CHANGES.md");
const outDir = path.join(root, "output", "pdf");
const outHtml = path.join(outDir, "guds_edl_theory_changes.html");
const outPdf = path.join(outDir, "guds_edl_theory_changes.pdf");
const katexCss = path.join(root, "tools", "mathpdf", "node_modules", "katex", "dist", "katex.min.css");

function esc(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function inline(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function isMathBlock(code, lang) {
  if (lang !== "tex" && lang !== "latex") return false;
  if (/\\section|\\subsection|\\label\{/.test(code)) return false;
  return /\\frac|\\sum|\\partial|\\alpha|\\beta|\\gamma|\\Delta|\\nabla|\\operatorname|\\begin\{|[_^]|=|<|>/.test(code);
}

function normalizeTex(code) {
  let s = code.trim();
  s = s.replace(/\\begin\{equation\*?\}/g, "").replace(/\\end\{equation\*?\}/g, "");
  s = s.replace(/\\begin\{split\}/g, "\\begin{aligned}").replace(/\\end\{split\}/g, "\\end{aligned}");
  s = s.replace(/\\text\{if \}/g, "\\text{if }");
  return s.trim();
}

function renderMath(code) {
  const tex = normalizeTex(code);
  try {
    return `<div class="math-block">${katex.renderToString(tex, {
      displayMode: true,
      throwOnError: false,
      strict: false,
      trust: true,
      maxSize: Infinity,
      maxExpand: 2000,
    })}</div>`;
  } catch (err) {
    return `<pre class="code-fallback">${esc(code)}</pre>`;
  }
}

function renderMarkdown(md) {
  const lines = md.split(/\r?\n/);
  const parts = [];
  let para = [];
  let inCode = false;
  let codeLang = "";
  let code = [];

  function flushPara() {
    if (!para.length) return;
    const text = para.join(" ").trim();
    if (text) parts.push(`<p>${inline(text)}</p>`);
    para = [];
  }

  for (const line of lines) {
    const trimmed = line.trim();
    const fence = trimmed.match(/^```(\w+)?/);
    if (fence) {
      if (inCode) {
        const raw = code.join("\n");
        if (isMathBlock(raw, codeLang)) parts.push(renderMath(raw));
        else parts.push(`<pre class="code-fallback">${esc(raw)}</pre>`);
        inCode = false;
        code = [];
        codeLang = "";
      } else {
        flushPara();
        inCode = true;
        codeLang = fence[1] || "";
      }
      continue;
    }
    if (inCode) {
      code.push(line);
      continue;
    }
    if (!trimmed) {
      flushPara();
      continue;
    }
    if (trimmed.startsWith("# ")) {
      flushPara();
      parts.push(`<h1>${inline(trimmed.slice(2))}</h1>`);
      parts.push(`<p class="subtitle">Bản ghi chú lý thuyết, công thức và chứng minh liên quan đến GUDS-EDL / MDEP.</p>`);
      continue;
    }
    if (trimmed.startsWith("## ")) {
      flushPara();
      parts.push(`<h2>${inline(trimmed.slice(3))}</h2>`);
      continue;
    }
    if (trimmed.startsWith("### ")) {
      flushPara();
      parts.push(`<h3>${inline(trimmed.slice(4))}</h3>`);
      continue;
    }
    if (trimmed.startsWith("- ")) {
      flushPara();
      parts.push(`<li>${inline(trimmed.slice(2))}</li>`);
      continue;
    }
    para.push(line);
  }
  flushPara();
  return parts.join("\n");
}

function htmlDoc(body) {
  const cssHref = pathToFileURL(katexCss).href;
  return `<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>GUDS-EDL Theory Changes</title>
<link rel="stylesheet" href="${cssHref}">
<style>
  @page { size: Letter; margin: 14mm 14mm 15mm 14mm; }
  body {
    font-family: Arial, "Helvetica Neue", sans-serif;
    color: #20262e;
    font-size: 10.2px;
    line-height: 1.38;
    counter-reset: page;
  }
  h1 {
    color: #16324f;
    font-size: 22px;
    text-align: center;
    margin: 0 0 6px 0;
    line-height: 1.15;
  }
  .subtitle {
    text-align: center;
    color: #536579;
    margin: 0 0 14px 0;
    padding-bottom: 9px;
    border-bottom: 1px solid #cbd5e1;
  }
  h2 {
    color: #1f4e79;
    font-size: 13px;
    margin: 11px 0 4px 0;
    page-break-after: avoid;
  }
  h3 {
    color: #1f4e79;
    font-size: 11px;
    margin: 8px 0 3px 0;
    page-break-after: avoid;
  }
  p {
    margin: 0 0 5px 0;
    text-align: justify;
  }
  li {
    margin: 0 0 3px 15px;
  }
  code {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 0.94em;
    background: #eef4fa;
    border-radius: 3px;
    padding: 0 2px;
  }
  .math-block {
    margin: 5px 0 7px 0;
    padding: 7px 8px;
    background: #f7fafc;
    border: 1px solid #c7d3e0;
    border-radius: 5px;
    overflow: visible;
    page-break-inside: avoid;
  }
  .math-block .katex-display {
    margin: 0;
    overflow: visible;
  }
  .math-block .katex {
    font-size: 1.02em;
    line-height: 1.25;
    white-space: normal;
  }
  pre.code-fallback {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 8.5px;
    line-height: 1.28;
    white-space: pre-wrap;
    margin: 5px 0 7px 0;
    padding: 7px 8px;
    background: #f7fafc;
    border: 1px solid #c7d3e0;
    border-radius: 5px;
  }
</style>
</head>
<body>
${body}
</body>
</html>`;
}

async function main() {
  fs.mkdirSync(outDir, { recursive: true });
  const md = fs.readFileSync(source, "utf8");
  const html = htmlDoc(renderMarkdown(md));
  fs.writeFileSync(outHtml, html, "utf8");

  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  });
  const page = await browser.newPage();
  await page.goto(pathToFileURL(outHtml).href, { waitUntil: "networkidle" });
  await page.pdf({
    path: outPdf,
    format: "Letter",
    printBackground: true,
    margin: { top: "14mm", right: "14mm", bottom: "15mm", left: "14mm" },
  });
  await browser.close();
  console.log(outPdf);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
