#!/bin/sh
# Builds docs/ml-compiler-guide.pdf (and .html) from the chapters in this directory.
# Needs pandoc and Google Chrome.
set -e
cd "$(dirname "$0")"
cat > /tmp/mlc_guide.css <<'CSS'
body { font: 11pt/1.5 -apple-system, "Helvetica Neue", Arial, sans-serif; max-width: 46em; margin: 2em auto; padding: 0 1em; color: #1a1a1a; }
h1 { font-size: 1.7em; border-bottom: 2px solid #999; padding-bottom: .2em; margin-top: 2.2em; page-break-before: always; }
h1:first-of-type { page-break-before: avoid; }
h2 { font-size: 1.25em; margin-top: 1.6em; } h3 { font-size: 1.05em; }
code, pre { font: 9pt/1.35 Menlo, Consolas, monospace; }
pre { background: #f4f4f4; padding: .7em .9em; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; page-break-inside: avoid; }
code { background: #f4f4f4; padding: 0 .2em; } pre code { padding: 0; }
table { border-collapse: collapse; margin: 1em 0; font-size: .92em; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: .3em .6em; vertical-align: top; } th { background: #eee; }
CSS
pandoc README.md 0*.md 1*.md -f gfm -t html5 -s --toc --toc-depth=2 \
  --metadata title="The ml-compiler guide" -c /tmp/mlc_guide.css --embed-resources -o ../ml-compiler-guide.html
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=../ml-compiler-guide.pdf "file://$(cd .. && pwd)/ml-compiler-guide.html" 2>/dev/null
echo "wrote docs/ml-compiler-guide.pdf"
