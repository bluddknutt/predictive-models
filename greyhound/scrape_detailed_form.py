"""
Enhanced greyhound form scraper - fetches detailed per-runner data:
  - Past 6 race starts (time, placing, box, weight, distance, grade, margin, sectional, SP, PIR)
  - Box draw history (starts per distance band)
  - Best track times
  - Race page data: speedmap sectional, track/dist record, running trait
"""

import re
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

BASE_URL = "https://www.thedogs.com.au"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
}
AEST = timezone(timedelta(hours=11))


def fetch(url):
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt == 2:
                print(f"  WARN: Failed {url}: {e}")
                return None
            time.sleep(1)


def get_todays_venues(date_str):
    html = fetch(f"{BASE_URL}/racing")
    if not html:
        return []
    pattern = rf'meetings-venues__name"><a[^>]*href="(/racing/([^/]+)/{date_str}\?trial=false)">([^<]+)</a>'
    matches = re.findall(pattern, html)
    venues = []
    seen = set()
    for path, slug, name in matches:
        if slug not in seen:
            seen.add(slug)
            venues.append({"url": f"{BASE_URL}{path}", "slug": slug, "name": name.strip(), "state": ""})
    return venues


def get_venue_races(venue_url, venue_info, cutoff_time):
    html = fetch(venue_url)
    if not html:
        return [], venue_info
    state_match = re.search(r'meeting-header__venue__state">([^<]+)', html)
    if state_match:
        venue_info["state"] = state_match.group(1).strip()

    upcoming = re.findall(
        r'<a class="race-box" href="([^"]+)"[^>]*data-race-box="([^"]+)"[^>]*>.*?'
        r'<div class="race-box__number">([^<]+)</div>',
        html, re.DOTALL
    )
    races = []
    for path, race_time_str, race_num in upcoming:
        try:
            race_time = datetime.fromisoformat(race_time_str)
        except ValueError:
            continue
        if race_time >= cutoff_time:
            races.append({
                "url": f"{BASE_URL}{path}",
                "race_number": int(race_num.replace("R", "")),
                "race_time": race_time_str,
                "venue": venue_info["name"],
                "state": venue_info["state"],
            })
    return races, venue_info


def parse_race_page_runners(race_url, race_info):
    """Extract enriched runner data from the race page (speedmap, track/dist, trait, dog_id)."""
    html = fetch(race_url)
    if not html:
        return []

    # Race metadata
    race_name = ""
    m = re.search(r'race-header__info__name[^>]*>([^<]+)', html)
    if m:
        race_name = m.group(1).strip()

    grade_distance = ""
    m = re.search(r'race-header__info__grade[^>]*>([^<]+)', html)
    if m:
        grade_distance = m.group(1).strip()
    dist_m = re.search(r'(\d+)m', grade_distance)
    distance = dist_m.group(0) if dist_m else ""
    grade = grade_distance.replace(distance, "").strip() if distance else grade_distance

    # Parse each runner row - capture class + content so we can filter scratched
    runner_pattern = re.compile(
        r'<tr[^>]*class="([^"]*accordion__anchor\s+race-runner[^"]*)"[^>]*>(.*?)</tr>',
        re.DOTALL
    )
    raw_rows = runner_pattern.findall(html)
    rows = [content for cls, content in raw_rows if "scratched" not in cls]

    # Also extract content-urls for expanded data (dog profile links)
    content_urls = re.findall(r'data-content-url="(/dogs/runner/\d+)"', html)

    runners = []
    for i, row in enumerate(rows):
        # Box
        box_match = re.search(r'rug_(\d+)', row)
        box = int(box_match.group(1)) if box_match else 0

        # Dog name and dog_id
        dog_id_match = re.search(r'data-dog-id="(\d+)"', row)
        dog_id = int(dog_id_match.group(1)) if dog_id_match else 0
        dog_match = re.search(r'race-runners__name__dog"[^>]*>(?:<a[^>]*>)?([^<]+)', row)
        dog_name = dog_match.group(1).strip() if dog_match else ""
        dog_name = dog_name.replace("&#39;", "'").replace("&amp;", "&")

        # Best time
        time_match = re.search(r'race-runners__name__time"[^>]*>([^<]+)', row)
        best_time = time_match.group(1).strip() if time_match else ""

        # Trainer
        trainer_match = re.search(r'race-runners__name__trainer"[^>]*>T:\s*([^R<]+)', row)
        trainer = trainer_match.group(1).strip() if trainer_match else ""

        # Running trait (e.g. FR=front runner, RR=railer, MW=midtrack wide)
        trait_match = re.search(r'race-runners__track-sa-trait"[^>]*>([^<]+)', row)
        trait = trait_match.group(1).strip() if trait_match else ""

        # Grade
        grade_match = re.search(r'race-runners__grade"[^>]*>([^<]+)', row)
        runner_grade = grade_match.group(1).strip() if grade_match else ""

        # Last 4
        last4_match = re.search(r'race-runners__last-four"[^>]*>([^<]+)', row)
        last_4 = last4_match.group(1).strip() if last4_match else ""

        # Track/dist record (e.g. "28: 2-2-4" = 28 starts, 2W-2P-4S at this track/dist)
        td_match = re.search(r'race-runners__track-dist"[^>]*>([^<]+)', row)
        track_dist_record = td_match.group(1).strip() if td_match else ""

        # Speedmap (1st sectional)
        speed_match = re.search(r'race-runners__speedmap"[^>]*><div>([^<]+)', row)
        speedmap_sec = speed_match.group(1).strip() if speed_match else ""

        # Last start info
        last_start_match = re.search(
            r'race-runners__last-start"[^>]*>.*?<a[^>]*class="runner-result-cell"[^>]*>'
            r'.*?<div>(\d+\w*/\d+)</div>.*?<div>(\d+m)</div>.*?<div>([^<]+)</div>.*?<div>([^<]+)</div>',
            row, re.DOTALL
        )
        last_start_place = last_start_match.group(1) if last_start_match else ""
        last_start_dist = last_start_match.group(2) if last_start_match else ""
        last_start_date = last_start_match.group(3).strip() if last_start_match else ""
        last_start_track = last_start_match.group(4).strip() if last_start_match else ""

        # Dog profile URL for detailed past starts
        dog_slug = re.search(r'href="/dogs/\d+/([^"]+)"', row)
        dog_profile_url = f"{BASE_URL}/dogs/{dog_id}/{dog_slug.group(1)}" if dog_slug and dog_id else ""

        if dog_name:
            runners.append({
                "venue": race_info["venue"],
                "state": race_info["state"],
                "race_number": race_info["race_number"],
                "race_name": race_name,
                "race_time": race_info["race_time"],
                "distance": distance,
                "grade": grade,
                "box": box,
                "dog_name": dog_name,
                "dog_id": dog_id,
                "trainer": trainer,
                "best_time": best_time,
                "last_4_starts": last_4,
                "trait": trait,
                "runner_grade": runner_grade,
                "track_dist_record": track_dist_record,
                "speedmap_sectional": speedmap_sec,
                "last_start_place": last_start_place,
                "last_start_dist": last_start_dist,
                "last_start_date": last_start_date,
                "last_start_track": last_start_track,
                "dog_profile_url": dog_profile_url,
            })
    return runners


def _extract_text(html_fragment):
    """Strip HTML tags and return clean text."""
    return re.sub(r'<[^>]+>', '', html_fragment).strip()


def parse_dog_profile(dog_profile_url):
    """Fetch dog profile page and extract past 6 races + box history."""
    html = fetch(dog_profile_url)
    if not html:
        return [], {}, []

    # ── Past race form ──
    # The show-runner-form table has <thead> (header) and <tbody> (data rows)
    past_races = []
    form_table = re.search(
        r'<table[^>]*class="[^"]*show-runner-form[^"]*"[^>]*>(.*?)</table>',
        html, re.DOTALL
    )
    if form_table:
        tbody = re.search(r'<tbody>(.*?)</tbody>', form_table.group(1), re.DOTALL)
        if tbody:
            rows = re.findall(r'<tr>(.*?)</tr>', tbody.group(1), re.DOTALL)
            for row in rows[:6]:
                def cell(cls):
                    m = re.search(rf'{cls}"[^>]*>(.*?)</td>', row, re.DOTALL)
                    return _extract_text(m.group(1)) if m else ""

                # Extract all fields from cells by their class
                placing = cell("runner-form__finish-position")
                box_m = re.search(r'rug_(\d+)', row)
                box = int(box_m.group(1)) if box_m else 0
                weight = cell("runner-form__weight").replace("kg", "")
                dist = cell("runner-form__distance")
                track = cell("runner-form__track")
                grade_r = cell("runner-form__grade")
                margin = cell("runner-form__margin").replace("\u2014", "0").replace("—", "0")
                pir = cell("runner-form__in-running-places")
                sp = cell("runner-form__starting-price").replace("$", "")

                # Multiple runner-form__time cells: race_time, win_time, bon_time, 1st_sec
                time_cells = re.findall(r'runner-form__time"[^>]*>(.*?)</td>', row, re.DOTALL)
                race_time_v = _extract_text(time_cells[0]) if len(time_cells) > 0 else ""
                win_time = _extract_text(time_cells[1]) if len(time_cells) > 1 else ""
                bon_time = _extract_text(time_cells[2]) if len(time_cells) > 2 else ""
                sec1 = _extract_text(time_cells[3]) if len(time_cells) > 3 else ""

                # Date from timestamp
                ts_match = re.search(r'data-timestamp="(\d+)"', row)
                date_str = ""
                if ts_match:
                    date_str = datetime.fromtimestamp(int(ts_match.group(1)), tz=AEST).strftime("%Y-%m-%d")

                # Winner
                winner_m = re.search(r'runner-form__winner"[^>]*>(.*?)</td>', row, re.DOTALL)
                winner = _extract_text(winner_m.group(1)) if winner_m else ""

                past_races.append({
                    "placing": placing, "box": box, "weight": weight,
                    "distance": int(dist) if dist.isdigit() else 0,
                    "date": date_str, "track": track, "grade": grade_r,
                    "race_time": race_time_v, "win_time": win_time,
                    "bon_time": bon_time, "first_sectional": sec1,
                    "margin": margin, "winner": winner, "pir": pir, "sp": sp,
                })

    # ── Box history ──
    # Second box-history table: header row with rug sprites (box numbers),
    # then rows: Starts, Wins, Places, Win%, Place% - each with values per box
    box_history = {}
    bh_tables = re.findall(r'<table[^>]*class="[^"]*box-history"[^>]*>(.*?)</table>', html, re.DOTALL)
    if len(bh_tables) >= 2:
        bh_html = bh_tables[1]
        bh_rows = re.findall(r'<tr[^>]*>(.*?)</tr>', bh_html, re.DOTALL)
        # First row has the box numbers as rug sprites in <th> tags
        box_nums = []
        data_rows = {}
        for bh_row in bh_rows:
            rugs = re.findall(r'rug_(\d+)', bh_row)
            if rugs and not box_nums:
                box_nums = [int(r) for r in rugs]
                continue
            label_m = re.search(r'box-history__title"[^>]*>([^<]+)', bh_row)
            if label_m:
                label = label_m.group(1).strip().lower()
                # Values are in box-history__box cells after the title cell
                vals = re.findall(r'box-history__box[^>]*>(\d+)', bh_row)
                data_rows[label] = [int(v) for v in vals]

        if box_nums and "starts" in data_rows:
            for i, b in enumerate(box_nums):
                if i < len(data_rows.get("starts", [])):
                    box_history[b] = {
                        "starts": data_rows["starts"][i] if i < len(data_rows.get("starts", [])) else 0,
                        "wins": data_rows["wins"][i] if i < len(data_rows.get("wins", [])) else 0,
                        "places": data_rows["places"][i] if i < len(data_rows.get("places", [])) else 0,
                    }

    # ── Best track times ──
    best_times = []
    bt_table = re.search(r'<table[^>]*class="[^"]*best-track-times"[^>]*>(.*?)</table>', html, re.DOTALL)
    if bt_table:
        bt_rows = re.findall(r'<tr>(.*?)</tr>', bt_table.group(1), re.DOTALL)
        for bt_row in bt_rows:
            track_m = re.search(r'best-track-times__track"[^>]*>(.*?)</td>', bt_row, re.DOTALL)
            time_m = re.search(r'best-track-times__time"[^>]*>([^<]+)', bt_row)
            box_m = re.search(r'rug_(\d+)', bt_row)
            dist_m = re.search(r'best-track-times__distance"[^>]*>([^<]+)', bt_row)
            if track_m and time_m:
                best_times.append({
                    "track": _extract_text(track_m.group(1)),
                    "time": time_m.group(1).strip(),
                    "box": int(box_m.group(1)) if box_m else 0,
                    "distance": dist_m.group(1).strip() if dist_m else "",
                })

    return past_races, box_history, best_times


def scrape_all_detailed(date_str, cutoff_time):
    """Scrape full detailed form for all runners in upcoming races."""
    print(f"Fetching venues for {date_str}...")
    venues = get_todays_venues(date_str)
    print(f"Found {len(venues)} venues: {', '.join(v['name'] for v in venues)}")

    all_runners = []
    total_races = 0

    for venue in venues:
        print(f"\n{'='*60}")
        print(f"  {venue['name']}")
        print(f"{'='*60}")
        races, venue = get_venue_races(venue["url"], venue, cutoff_time)
        print(f"  {len(races)} upcoming races")
        total_races += len(races)

        for race in races:
            print(f"  R{race['race_number']} ({race['race_time'][:16]})...", end="", flush=True)
            runners = parse_race_page_runners(race["url"], race)
            print(f" {len(runners)} runners", end="", flush=True)

            for runner in runners:
                # Fetch detailed form for each dog
                if runner["dog_profile_url"]:
                    past_races, box_hist, best_times = parse_dog_profile(runner["dog_profile_url"])

                    # Flatten past 6 races into columns
                    for j, pr in enumerate(past_races[:6], 1):
                        runner[f"pr{j}_placing"] = pr["placing"]
                        runner[f"pr{j}_box"] = pr["box"]
                        runner[f"pr{j}_weight"] = pr["weight"]
                        runner[f"pr{j}_dist"] = pr["distance"]
                        runner[f"pr{j}_date"] = pr["date"]
                        runner[f"pr{j}_track"] = pr["track"]
                        runner[f"pr{j}_grade"] = pr["grade"]
                        runner[f"pr{j}_time"] = pr["race_time"]
                        runner[f"pr{j}_win_time"] = pr["win_time"]
                        runner[f"pr{j}_bon_time"] = pr["bon_time"]
                        runner[f"pr{j}_sec1"] = pr["first_sectional"]
                        runner[f"pr{j}_margin"] = pr["margin"]
                        runner[f"pr{j}_winner"] = pr["winner"]
                        runner[f"pr{j}_pir"] = pr["pir"]
                        runner[f"pr{j}_sp"] = pr["sp"]

                    # Box history for today's box
                    today_box = runner["box"]
                    if today_box in box_hist:
                        bh = box_hist[today_box]
                        runner["box_starts"] = bh["starts"]
                        runner["box_wins"] = bh["wins"]
                        runner["box_places"] = bh["places"]
                        total = bh["starts"] if bh["starts"] > 0 else 1
                        runner["box_win_pct"] = round(bh["wins"] / total * 100, 1)
                        runner["box_place_pct"] = round(bh["places"] / total * 100, 1)
                    else:
                        runner["box_starts"] = 0
                        runner["box_wins"] = 0
                        runner["box_places"] = 0
                        runner["box_win_pct"] = 0
                        runner["box_place_pct"] = 0

                    # Best time at today's track
                    race_track = venue["slug"]
                    track_best = None
                    for bt in best_times:
                        if bt["track"].lower().startswith(race_track[:4].lower()):
                            try:
                                t = float(bt["time"])
                                if track_best is None or t < track_best:
                                    track_best = t
                            except ValueError:
                                pass
                    runner["track_best_time"] = track_best if track_best else ""

                    time.sleep(0.3)  # Polite delay

                all_runners.append(runner)
            print(" OK")

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(venues)} venues, {total_races} races, {len(all_runners)} runners")
    return all_runners


if __name__ == "__main__":
    today = datetime.now(AEST).strftime("%Y-%m-%d")
    cutoff = datetime(2026, 3, 18, 18, 56, 0, tzinfo=AEST)

    all_data = scrape_all_detailed(today, cutoff)

    if all_data:
        df = pd.DataFrame(all_data)
        output_path = f"greyhound/data/detailed_form_{today}.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(df)} rows x {len(df.columns)} cols to {output_path}")
        print(f"Venues: {df['venue'].nunique()}")
        print(f"Races: {df.groupby(['venue', 'race_number']).ngroups}")
        print(f"\nColumns: {df.columns.tolist()}")
    else:
        print("\nNo data found.")
