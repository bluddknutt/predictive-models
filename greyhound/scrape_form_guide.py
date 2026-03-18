"""
Scrape greyhound race form guide data from thedogs.com.au.
Fetches all upcoming races for today and extracts field/runner data.
"""

import re
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

BASE_URL = "https://www.thedogs.com.au"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
}
AEST = timezone(timedelta(hours=11))


def fetch(url):
    """Fetch a URL with retries."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt == 2:
                print(f"  Failed to fetch {url}: {e}")
                return None
            time.sleep(1)


def get_todays_venues(date_str):
    """Get list of venues racing today from the main racing page."""
    html = fetch(f"{BASE_URL}/racing")
    if not html:
        return []

    # Venues are in table rows: <td class="meetings-venues__name"><a href="...">Name</a></td>
    pattern = rf'meetings-venues__name"><a[^>]*href="(/racing/([^/]+)/{date_str}\?trial=false)">([^<]+)</a>'
    matches = re.findall(pattern, html)

    venues = []
    seen = set()
    for path, slug, name in matches:
        if slug not in seen:
            seen.add(slug)
            venues.append({
                "url": f"{BASE_URL}{path}",
                "slug": slug,
                "name": name.strip(),
                "state": "",  # Will be populated from venue page
            })
    return venues


def get_venue_races(venue_url, venue_info, cutoff_time):
    """Get upcoming races for a venue (races with times after cutoff)."""
    html = fetch(venue_url)
    if not html:
        return [], venue_info

    # Extract state from the venue page header
    state_match = re.search(r'meeting-header__venue__state">([^<]+)', html)
    if state_match:
        venue_info["state"] = state_match.group(1).strip()

    # Find upcoming race boxes (have data-race-box timestamp, no race-box--result class)
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


def get_race_fields(race_url, race_info):
    """Extract all runner data from a race page."""
    html = fetch(race_url)
    if not html:
        return []

    # Extract race name and grade/distance
    race_name = ""
    match = re.search(r'race-header__info__name[^>]*>([^<]+)', html)
    if match:
        race_name = match.group(1).strip()

    grade_distance = ""
    match = re.search(r'race-header__info__grade[^>]*>([^<]+)', html)
    if match:
        grade_distance = match.group(1).strip()

    # Parse distance from grade string (e.g. "Grade 5 Heat 525m")
    dist_match = re.search(r'(\d+m)', grade_distance)
    distance = dist_match.group(1) if dist_match else ""
    grade = grade_distance.replace(distance, "").strip() if distance else grade_distance

    # Extract each runner's row
    # Pattern: find each accordion row with runner data
    runner_pattern = re.compile(
        r'<tr[^>]*class="[^"]*accordion__anchor\s+race-runner[^"]*"[^>]*>(.*?)</tr>',
        re.DOTALL
    )
    rows = runner_pattern.findall(html)

    runners = []
    for row in rows:
        # Box number from rug sprite
        box_match = re.search(r'rug_(\d+)', row)
        box = int(box_match.group(1)) if box_match else 0

        # Dog name
        dog_match = re.search(r'race-runners__name__dog"[^>]*>([^<]+)', row)
        dog_name = dog_match.group(1).strip() if dog_match else ""
        # Decode HTML entities
        dog_name = dog_name.replace("&#39;", "'").replace("&amp;", "&")

        # Best time
        time_match = re.search(r'race-runners__name__time"[^>]*>([^<]+)', row)
        best_time = time_match.group(1).strip() if time_match else ""

        # Trainer
        trainer_match = re.search(r'race-runners__name__trainer"[^>]*>T:\s*([^R<]+)', row)
        trainer = trainer_match.group(1).strip() if trainer_match else ""

        # Grade
        grade_match = re.search(r'race-runners__grade"[^>]*>([^<]+)', row)
        runner_grade = grade_match.group(1).strip() if grade_match else ""

        # Last 4 starts
        last4_match = re.search(r'race-runners__last-four"[^>]*>([^<]+)', row)
        last_4 = last4_match.group(1).strip() if last4_match else ""

        # Last start info
        last_start_match = re.search(r'race-runners__last-start"[^>]*>.*?<a[^>]*>([^<]+)</a>', row, re.DOTALL)
        last_start = last_start_match.group(1).strip() if last_start_match else ""

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
                "trainer": trainer,
                "best_time": best_time,
                "last_4_starts": last_4,
                "last_start": last_start,
            })

    return runners


def scrape_all_upcoming(date_str, cutoff_time):
    """Scrape all upcoming race data for today."""
    print(f"Fetching venues for {date_str}...")
    venues = get_todays_venues(date_str)
    print(f"Found {len(venues)} venues: {', '.join(v['name'] for v in venues)}")

    all_runners = []
    total_races = 0

    for venue in venues:
        print(f"\nFetching races for {venue['name']}...")
        races, venue = get_venue_races(venue["url"], venue, cutoff_time)
        print(f"  {len(races)} upcoming races")
        total_races += len(races)

        for race in races:
            print(f"  Fetching R{race['race_number']} ({race['race_time']})...")
            runners = get_race_fields(race["url"], race)
            print(f"    {len(runners)} runners")
            all_runners.extend(runners)
            time.sleep(0.5)  # Be polite

    print(f"\n{'='*60}")
    print(f"Total: {len(venues)} venues, {total_races} races, {len(all_runners)} runners")
    return all_runners


if __name__ == "__main__":
    today = datetime.now(AEST).strftime("%Y-%m-%d")
    # Cutoff: 6:56 PM AEST (user's current time)
    cutoff = datetime(2026, 3, 18, 18, 56, 0, tzinfo=AEST)

    all_data = scrape_all_upcoming(today, cutoff)

    if all_data:
        df = pd.DataFrame(all_data)
        output_path = f"greyhound/data/form_guide_{today}.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(df)} rows to {output_path}")
        print(f"\nVenues: {df['venue'].nunique()}")
        print(f"Races: {df.groupby(['venue', 'race_number']).ngroups}")
        print(f"\nSample data:")
        print(df.head(10).to_string(index=False))
    else:
        print("\nNo upcoming race data found.")
