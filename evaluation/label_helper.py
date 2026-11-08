"""Print timestamped, numbered screenplay lines to help create gold labels.

Usage: python3 evaluation/label_helper.py /path/to/episodes 12
Nothing is written to disk unless the caller redirects terminal output.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser.txt_parser import parse_directory
from llm_selector import build_lines


def format_episode(folder, episode_number):
    episodes = parse_directory(str(Path(folder).expanduser()))
    episode = next((ep for ep in episodes if ep.number == int(episode_number)), None)
    if episode is None:
        raise ValueError(f"Episode {episode_number} not found or could not be parsed in {folder}")
    lines = build_lines(episode)
    output = [f"EPISODE {episode.number} — {episode.filename}"]
    for line in lines:
        start = line["start"]
        end = line["end"]
        mins1, secs1 = divmod(start, 60)
        mins2, secs2 = divmod(end, 60)
        output.append(f'{line["id"]} [{int(mins1):02d}:{secs1:05.2f}-{int(mins2):02d}:{secs2:05.2f}] {line["text"]}')
    return "\n".join(output)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="Folder containing episode TXT files")
    ap.add_argument("episode", type=int, help="Episode number to print")
    args = ap.parse_args()
    try:
        print(format_episode(args.folder, args.episode))
    except ValueError as exc:
        ap.error(str(exc))

if __name__ == "__main__":
    main()
