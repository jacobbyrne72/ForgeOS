#!/usr/bin/env bash
# Enumerate channel video TITLES only (cheap metadata), filter deterministically,
# then emit just the matching video ids. Fetching transcripts for everything a
# channel ever posted would be exactly the waste this harness exists to prevent.
set -u
R=/c/Users/byrne/Downloads/hive/research
OUT=$R/matched_ids.txt
RAW=$R/enumerated.tsv
: > "$RAW"

# Priority order per the ranked corpus: harness-design channels first.
CHANNELS='
https://www.youtube.com/@indydevdan/videos
https://www.youtube.com/@colemedin/videos
https://www.youtube.com/@daveebbelaar/videos
https://www.youtube.com/@AIJasonZ/videos
https://www.youtube.com/@AllAboutAI/videos
https://www.youtube.com/@jamesbriggs/videos
https://www.youtube.com/@MervinPraison/videos
https://www.youtube.com/@davidondrej/videos
https://www.youtube.com/@AndrejKarpathy/videos
https://www.youtube.com/playlist?list=PLinedj3B30sCzJnjhtEZBpKGtC1Xk7z5Z
'

INCLUDE='agent|multi-agent|swarm|orchestrat|coding agent|claude code|codex|gemini cli|opencode|aider|\bmcp\b|\bacp\b|context engineer|prompt engineer|memory|\brag\b|eval|routing|model select|token|subagent|worktree|sandbox|automation|n8n|langgraph|pydantic ai|crewai|openrouter|litellm|cheap|cost|spec|skill|harness|parallel'
EXCLUDE='will change everything|is insane|breaking news|destroys|make money|top 100|daily ai news|reaction|rumou?r|shocking|you won.t believe'

for ch in $CHANNELS; do
  [ -z "$ch" ] && continue
  name=$(echo "$ch" | sed -E 's#.*/@([^/]+)/.*#\1#; s#.*list=#playlist-#')
  n=$(yt-dlp --flat-playlist --playlist-end 60 \
        --print "%(id)s\t%(title)s" "$ch" 2>/dev/null \
      | sed "s#^#${name}\t#" | tee -a "$RAW" | wc -l)
  printf '%-24s enumerated %s\n' "$name" "$n"
done

# Filter: must match an include keyword, must not match an exclude keyword.
awk -F'\t' 'NF>=3' "$RAW" \
  | grep -Ei "$INCLUDE" \
  | grep -Eiv "$EXCLUDE" \
  | awk -F'\t' '{print $2"\t"$1"\t"$3}' \
  | sort -u > "$R/matched.tsv"

cut -f1 "$R/matched.tsv" | sed 's#^#https://youtu.be/#' | sort -u > "$OUT"

echo
echo "=== enumeration summary ==="
echo "enumerated total : $(wc -l < "$RAW")"
echo "matched (kept)   : $(wc -l < "$OUT")"
echo "dropped as noise : $(( $(wc -l < "$RAW") - $(wc -l < "$OUT") ))"
echo
echo "kept per channel:"
cut -f2 "$R/matched.tsv" | sort | uniq -c | sort -rn
echo
echo "ids -> $OUT"
