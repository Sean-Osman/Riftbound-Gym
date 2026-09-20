"""Convert a tab of the 'Riftbound Collection' spreadsheet into card JSON.

    python3 import_sheet.py "~/Downloads/Riftbound Collection.xlsx" "Kaisa Deck" card_data/kaisa_deck.json

Stdlib only (reads the .xlsx XML directly). The output loads with
cards.load_card_pool(). Known gaps in the sheet are patched via OVERRIDES.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from cards import Keyword, card_from_dict

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

DOMAIN_LETTERS = {"Fury": "R", "Calm": "G", "Mind": "B", "Body": "O", "Chaos": "P", "Order": "Y"}

# "Card Type" column -> (types, supertypes, is_token)
TYPE_COLUMN = {
    "Unit": (["unit"], [], False),
    "Champion Unit": (["unit"], ["champion"], False),
    "Signature Unit": (["unit"], ["signature"], False),
    "Token Unit": (["unit"], [], True),
    "Spell": (["spell"], [], False),
    "Signature Spell": (["spell"], ["signature"], False),
    "Gear": (["gear"], [], False),
    "Basic Rune": (["rune"], [], False),
    "Battlefield": (["battlefield"], [], False),
    "Legend": (["legend"], [], False),
    "Champion Legend": (["legend"], [], False),
}

# The sheet's Domain column holds a single domain, so dual-domain cards
# (every legend) lose their second one. Patch by card_id.
OVERRIDES: dict[str, dict[str, Any]] = {
    "OGN-247": {"domains": ["R", "B"]},    # Kai'Sa, Daughter of the Void: Fury / Mind
}

# 164.2: basic runes have no printed rules text in the sheet
BASIC_RUNE_TEXT = "[E]: [Reaction] — Add [1]. Recycle this: [Reaction] — Add [C]."

_LEADING_KEYWORD = re.compile(
    r"\s*\[(?P<kw>[A-Za-z\-]+)(?: (?P<n>\d+))?\]\s*(?P<dash>—)?\s*(?:\([^)]*\))?\s*")


# --- xlsx reading ------------------------------------------------------------

def read_sheet(xlsx: Path, sheet_name: str) -> list[dict[str, str]]:
    """Rows of a sheet as {header: value} dicts (empty cells omitted)."""
    with zipfile.ZipFile(xlsx) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels}
        for sheet in workbook.iter(f"{{{_NS['m']}}}sheet"):
            if sheet.get("name") == sheet_name:
                target = targets[sheet.get(_REL_NS)].lstrip("/")
                path = target if target.startswith("xl/") else f"xl/{target}"
                root = ET.fromstring(z.read(path))
                break
        else:
            raise ValueError(f"no sheet named {sheet_name!r}")

    grid: list[dict[str, str]] = []
    for row in root.iter(f"{{{_NS['m']}}}row"):
        cells: dict[str, str] = {}
        for c in row.findall("m:c", _NS):
            col = re.match(r"[A-Z]+", c.get("r")).group()
            v, inline = c.find("m:v", _NS), c.find("m:is", _NS)
            if c.get("t") == "s":
                value = shared[int(v.text)]
            elif c.get("t") == "inlineStr":
                value = "".join(t.text or "" for t in inline.iter(f"{{{_NS['m']}}}t"))
            else:
                value = v.text if v is not None else ""
            if value:
                cells[col] = value
        grid.append(cells)
    header, *body = grid
    return [{header[col]: val for col, val in row.items() if col in header} for row in body if row]


# --- field conversion --------------------------------------------------------

def clean_text(text: str) -> str:
    """Turn the sheet's icon names back into rules shorthand (135.2.e)."""
    text = text.replace("_", "")
    text = re.sub(r"(\d+) Energy(" + "|".join(DOMAIN_LETTERS) + ")",
                  lambda m: f"[{m[1]}][{DOMAIN_LETTERS[m[2]]}]", text)
    text = re.sub(r"(\d+) Energy", r"[\1]", text)
    text = text.replace("Rune Rainbow", "[A]")
    text = re.sub(r"Rune (" + "|".join(DOMAIN_LETTERS) + ")", lambda m: f"[{DOMAIN_LETTERS[m[1]]}]", text)
    text = re.sub(r"\bExhaust:", "[E]:", text)
    text = re.sub(r"([):])(?=[A-Z\[])", r"\1 ", text)  # reminder text ran into the next ability
    return text.strip()


def leading_keywords(text: str) -> dict[str, int | None]:
    """Keywords printed at the start of the card's text, e.g. '[Accelerate] (...)'.

    Stops at the first non-keyword, so 'Give a unit [Assault 3]' on Cleave is
    not mistaken for Cleave having Assault. A dependent / trigger keyword
    followed by '—' (Legion, Deathknell) ends the scan: the rest is its ability.
    """
    found: dict[str, int | None] = {}
    pos = 0
    while m := _LEADING_KEYWORD.match(text, pos):
        try:
            kw = Keyword.parse(m["kw"])
        except ValueError:
            break
        found[kw.value] = int(m["n"]) if m["n"] else None
        if m["dash"]:
            break
        pos = m.end()
    return found


def split_name(full: str) -> tuple[str, str | None]:
    """'Kai'Sa, Survivor' / 'Ahri - Nine-Tailed Fox' -> (short name, subtitle)."""
    full = re.sub(r"\s*\((Signature|Overnumbered|Showcase)\)$", "", full)
    for sep in (", ", " - "):
        if sep in full:
            short, subtitle = full.split(sep, 1)
            return short, subtitle
    return full, None


def row_to_card(row: dict[str, str]) -> dict[str, Any]:
    card_id = row["Card Number"].split("/")[0]
    short, subtitle = split_name(row["Card Name"])
    types, supertypes, is_token = TYPE_COLUMN[row["Card Type"]]
    domain = DOMAIN_LETTERS.get(row.get("Domain", "Colorless"))
    domains = [domain] if domain else []
    tags = [t.strip() for t in row.get("Tags", "").split(",") if t.strip()]

    champion_tag = None
    if "champion" in supertypes or "signature" in supertypes or "legend" in types:
        if short in tags:
            champion_tag = short
        elif len(tags) == 1:
            champion_tag = tags[0]

    raw_text = row.get("Ability", "")
    card: dict[str, Any] = {
        "card_id": card_id,
        "short_name": short,
        "subtitle": subtitle,
        "types": types,
        "supertypes": supertypes,
        "tags": tags,
        "champion_tag": champion_tag,
        "domains": domains,
        "keywords": leading_keywords(raw_text),
        "rules_text": clean_text(raw_text) or (BASIC_RUNE_TEXT if row["Card Type"] == "Basic Rune" else ""),
        "is_token": is_token,
        "art_url": row.get("Image URL", ""),
    }
    if "unit" in types or "gear" in types or "spell" in types:
        card["energy"] = int(float(row.get("Energy", 0)))
        # the sheet gives only a count of power symbols; they match the card's domain
        n_power = int(float(row.get("Power", 0)))
        if n_power:
            if len(domains) != 1:
                raise ValueError(f"{card_id}: can't infer power symbols for domains {domains}")
            card["power"] = f"{n_power}{domains[0]}"
    if "Might" in row:
        card["might"] = int(float(row["Might"]))
    card.update(OVERRIDES.get(card_id, {}))
    return {k: v for k, v in card.items() if v not in (None, [], {}, "", False)}


def main(xlsx: str, sheet: str, out: str) -> None:
    rows = read_sheet(Path(xlsx).expanduser(), sheet)
    cards = [row_to_card(r) for r in rows if r.get("Card Name")]
    for c in cards:
        card_from_dict(c)                      # validate against the rules model
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(cards, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(cards)} cards to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
