#!/usr/bin/env bash
# Deep enumeration across the full ranked channel list.
# TITLES ONLY — metadata is cheap, transcripts are not. Filtering happens before
# a single transcript is fetched, which is the whole point.
set -u
R=/c/Users/byrne/Downloads/hive/research
RAW=$R/enumerated_deep.tsv
: > "$RAW"
DEPTH=${DEPTH:-200}

# Tier 1 (harness design) first, then tier 2 (discovery / adjacent engineering).
CHANNELS='
indydevdan
colemedin
daveebbelaar
AIJasonZ
AllAboutAI
jamesbriggs
MervinPraison
davidondrej
rileybrownai
AndrejKarpathy
hyperautomationlabs1045
matthew_berman
LangChain
mreflow
WorldofAI
SkillLeapAI
aiadvantage
futurepedia
Aitrepreneur
samwitteveenai
AssemblyAI
'

for ch in $CHANNELS; do
  [ -z "$ch" ] && continue
  n=$(yt-dlp --flat-playlist --playlist-end "$DEPTH" \
        --print "%(id)s|||%(title)s" "https://www.youtube.com/@${ch}/videos" 2>/dev/null \
      | sed "s#^#${ch}|||#" | tee -a "$RAW" | wc -l)
  printf '%-26s %s\n' "$ch" "$n"
done

echo
echo "raw rows: $(wc -l < "$RAW")"
echo "-> now run: python $R/filter_titles.py --deep"
