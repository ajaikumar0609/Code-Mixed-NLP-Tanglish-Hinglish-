#!/bin/bash
# Render preview: swaps Calibri/Cambria for metric-compatible Carlito/Caladea (copy only), exports via Keynote.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"; SRC="$1"; OUTPDF="$2"; TMP=$(mktemp -d)
cp "$SRC" "$TMP/preview.pptx"; cd "$TMP"; mkdir x; cd x; unzip -q ../preview.pptx
python3 - <<'PY'
import glob,re
for f in glob.glob('ppt/**/*.xml',recursive=True):
    s=open(f,encoding='utf8').read(); t=s.replace('typeface="Calibri"','typeface="Carlito"').replace('typeface="Cambria"','typeface="Caladea"')
    if t!=s: open(f,'w',encoding='utf8').write(t)
PY
zip -qr ../preview_fonts.pptx . ; cd ..
"$DIR/render_doc.sh" "$TMP/preview_fonts.pptx" "$OUTPDF"
