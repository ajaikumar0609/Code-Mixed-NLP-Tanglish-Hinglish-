#!/bin/bash
# usage: render_doc.sh input.docx|pptx output.pdf   (uses Pages / Keynote export)
IN="$1"; OUT="$2"
case "$IN" in
  *.docx) APP="Pages";;
  *.pptx) APP="Keynote";;
esac
rm -f "$OUT"
osascript <<OSA
tell application "$APP"
  activate
  set d to open POSIX file "$IN"
  delay 4
  export d to POSIX file "$OUT" as PDF
  close d saving no
end tell
OSA
