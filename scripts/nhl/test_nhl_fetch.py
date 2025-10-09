import requests
import sys
from datetime import datetime

import pytest

def fetch_schedule():
    """Fetch schedule data from NHL API."""
    print("=" * 60)
    print("TEST 1: Schedule API")
    print("=" * 60)

    url = "https://api-web.nhle.com/v1/schedule/2025-10-07/2025-10-08"
    print(f"URL: {url}")
    print("Fetching...", end="", flush=True)

    try:
        r = requests.get(url, timeout=15)
        print(f" Status: {r.status_code}")

        if r.status_code == 200:
            data = r.json()
            total_games = 0
            for week in data.get('gameWeek', []):
                games = week.get('games', [])
                total_games += len(games)
                print(f"  Found {len(games)} games")
                for game in games[:3]:
                    print(
                        f"    {game['id']}: "
                        f"{game['awayTeam']['abbrev']} @ {game['homeTeam']['abbrev']} "
                        f"({game.get('gameState')})"
                    )

            return data
        else:
            print(f"ERROR: {r.text[:200]}")
            return None

    except Exception as e:
        print(f" FAILED: {e}")
        return None

def fetch_boxscore(game_id):
    """Fetch boxscore data for a specific game."""
    print("\n" + "=" * 60)
    print(f"TEST 2: Boxscore API for game {game_id}")
    print("=" * 60)

    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    print(f"URL: {url}")
    print("Fetching...", end="", flush=True)

    try:
        r = requests.get(url, timeout=15)
        print(f" Status: {r.status_code}")

        if r.status_code == 200:
            data = r.json()

            # Count players
            home_forwards = len(data.get('homeTeam', {}).get('forwards', []))
            home_defense = len(data.get('homeTeam', {}).get('defense', []))
            away_forwards = len(data.get('awayTeam', {}).get('forwards', []))
            away_defense = len(data.get('awayTeam', {}).get('defense', []))

            total = home_forwards + home_defense + away_forwards + away_defense
            print(f"  Players: {total} total")
            print(f"    Home: {home_forwards} F, {home_defense} D")
            print(f"    Away: {away_forwards} F, {away_defense} D")

            # Show sample player
            if data.get('homeTeam', {}).get('forwards'):
                player = data['homeTeam']['forwards'][0]
                print(f"\n  Sample player:")
                print(f"    Name: {player.get('name', {}).get('default')}")
                print(f"    Goals: {player.get('goals', 0)}")
                print(f"    Assists: {player.get('assists', 0)}")
                print(f"    Shots: {player.get('sog', 0)}")

            return True
        else:
            print(f"ERROR: {r.text[:200]}")
            return False

    except Exception as e:
        print(f" FAILED: {e}")
        return False


@pytest.fixture(scope="module")
def schedule_data():
    data = fetch_schedule()
    if not data:
        pytest.skip("Schedule request failed")
    return data


@pytest.fixture(scope="module")
def game_id(schedule_data):
    games = []
    for week in schedule_data.get('gameWeek', []):
        games.extend(week.get('games', []))

    if not games:
        pytest.skip("No games available in schedule response")

    return games[0]['id']


def test_schedule(schedule_data):
    assert schedule_data.get('gameWeek'), "Schedule response missing gameWeek"


def test_boxscore(game_id):
    assert fetch_boxscore(game_id), "Boxscore request failed"

if __name__ == "__main__":
    print("\nNHL API Test")
    print("=" * 60)
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    # Test 1: Schedule
    schedule_data = fetch_schedule()

    if not schedule_data:
        print("\n❌ Schedule test failed. Exiting.")
        sys.exit(1)

    # Test 2: Boxscore for first game
    games = []
    for week in schedule_data.get('gameWeek', []):
        games.extend(week.get('games', []))

    if games:
        first_game_id = games[0]['id']
        success = fetch_boxscore(first_game_id)

        if success:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Boxscore test failed.")
            sys.exit(1)
    else:
        print("\n❌ No games found in schedule.")
        sys.exit(1)
