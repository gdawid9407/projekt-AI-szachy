import argparse
import bz2
import io
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional
import zstandard as zstd
from tqdm import tqdm

from src.config import DATA_DIR

PGN_MENTOR_URLS = {
    "kasparov": "https://www.pgnmentor.com/players/Kasparov.zip",
    "carlsen": "https://www.pgnmentor.com/players/Carlsen.zip",
}
FICS_BASE_URL = "https://www.ficsgames.org"
LICHESS_BASE_URL = "https://database.lichess.org/standard"


def download_pgn_mentor(player: str = "kasparov", target_dir: Path = DATA_DIR) -> Path:
    """
    Downloads curated master games from PGN Mentor.
    """
    key = player.lower()
    if key not in PGN_MENTOR_URLS:
        raise ValueError(f"Unknown player '{player}'. Available: {list(PGN_MENTOR_URLS.keys())}")

    target_dir.mkdir(parents=True, exist_ok=True)
    pgn_path = target_dir / f"{key.capitalize()}.pgn"

    if pgn_path.is_file() and pgn_path.stat().st_size > 0:
        print(f"PGN Mentor archive exists at: {pgn_path} ({pgn_path.stat().st_size / 1024:.1f} KB)")
        return pgn_path

    url = PGN_MENTOR_URLS[key]
    print(f"Fetching master archive from {url}...")
    headers = {"User-Agent": "ChessAI-Downloader/1.0"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as resp:
        content = resp.read()

    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        archive.extractall(target_dir)

    print(f"Extracted to: {pgn_path}")
    return pgn_path


def download_fics_games(
    year: int = 2020,
    month: int = 1,
    rating_over_2000: bool = False,
    target_dir: Path = DATA_DIR,
) -> Path:
    """
    Queries the FICS archive for community/amateur games and streams decompression.
    Args:
        year: Target year (e.g., 2020).
        month: Target month (1-12).
        rating_over_2000: If True, queries games with avg rating > 2000; if False, all ratings.
        target_dir: Output directory.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    rating_tag = "standard_gt2000" if rating_over_2000 else "standard_all"
    output_pgn = target_dir / f"fics_{year}_{month:02d}_{rating_tag}.pgn"

    if output_pgn.is_file() and output_pgn.stat().st_size > 0:
        print(f"FICS dataset already exists: {output_pgn} ({output_pgn.stat().st_size / (1024*1024):.1f} MB)")
        return output_pgn

    # gametype 3 = Standard all ratings; gametype 4 = Standard > 2000
    gametype = "4" if rating_over_2000 else "3"
    post_payload = urllib.parse.urlencode({
        "action": "Download",
        "gametype": gametype,
        "year": str(year),
        "month": str(month),
        "movetimes": "0",
        "download": "Download",
    }).encode("utf-8")

    cgi_url = f"{FICS_BASE_URL}/cgi-bin/download.cgi"
    print(f"Querying FICS archive metadata for {year}-{month:02d} ({rating_tag})...")
    req = urllib.request.Request(cgi_url, data=post_payload, headers={"User-Agent": "ChessAI/1.0"})

    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("iso-8859-1")

    match = re.search(r'href="(/dl/[^"]+\.bz2)"', html)
    if not match:
        raise RuntimeError("Failed to resolve bz2 download URL from FICS service.")

    relative_dl = match.group(1)
    direct_url = f"{FICS_BASE_URL}{relative_dl}"
    print(f"Streaming and decompressing BZ2 archive from {direct_url}...")

    # Stream download and decompress simultaneously using chunked buffers
    dl_req = urllib.request.Request(direct_url, headers={"User-Agent": "ChessAI/1.0"})
    decompressor = bz2.BZ2Decompressor()

    with urllib.request.urlopen(dl_req) as response, open(output_pgn, "wb") as out_f:
        total_size = response.headers.get("Content-Length")
        total_bytes = int(total_size) if total_size else None
        progress = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="FICS BZ2 stream")

        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            progress.update(len(chunk))
            decompressed = decompressor.decompress(chunk)
            if decompressed:
                out_f.write(decompressed)

        progress.close()

    print(f"FICS dataset extracted to: {output_pgn} ({output_pgn.stat().st_size / (1024*1024):.1f} MB)")
    return output_pgn


def download_lichess_games(
    year_month: str = "2013-01",
    target_dir: Path = DATA_DIR,
    max_mb: Optional[int] = 100,
) -> Path:
    """
    Downloads and streams decompressing a monthly Lichess open database rated games dump (.pgn.zst).
    Args:
        year_month: String in format 'YYYY-MM' (e.g. '2013-01').
        target_dir: Output directory.
        max_mb: Optional maximum uncompressed megabytes to extract to prevent disk exhaustion.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    output_pgn = target_dir / f"lichess_{year_month}.pgn"

    if output_pgn.is_file() and output_pgn.stat().st_size > 0:
        print(f"Lichess dataset already exists: {output_pgn} ({output_pgn.stat().st_size / (1024*1024):.1f} MB)")
        return output_pgn

    url = f"{LICHESS_BASE_URL}/lichess_db_standard_rated_{year_month}.pgn.zst"
    print(f"Streaming Lichess Zstandard archive from {url}...")

    req = urllib.request.Request(url, headers={"User-Agent": "ChessAI-Downloader/1.0"})
    dctx = zstd.ZstdDecompressor()

    max_bytes = max_mb * 1024 * 1024 if max_mb else None
    written_bytes = 0

    with urllib.request.urlopen(req) as resp, open(output_pgn, "wb") as out_f:
        total_size = resp.headers.get("Content-Length")
        total_bytes = int(total_size) if total_size else None
        progress = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Lichess ZST stream")

        # Streaming chunk decompressor
        reader = dctx.stream_reader(resp)
        while True:
            chunk = reader.read(65536)
            if not chunk:
                break
            out_f.write(chunk)
            written_bytes += len(chunk)
            progress.update(len(chunk))

            if max_bytes is not None and written_bytes >= max_bytes:
                print(f"\nReached max uncompressed limit ({max_mb} MB). Stopping stream.")
                break

        progress.close()

    print(f"Lichess dataset extracted to: {output_pgn} ({output_pgn.stat().st_size / (1024*1024):.1f} MB)")
    return output_pgn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download chess datasets from multiple sources.")
    parser.add_argument(
        "--source",
        type=str,
        choices=["kasparov", "carlsen", "fics", "lichess"],
        default="fics",
        help="Dataset source: 'kasparov' / 'carlsen' (PGN Mentor), 'fics', or 'lichess'.",
    )
    parser.add_argument("--fics-year", type=int, default=2020)
    parser.add_argument("--fics-month", type=int, default=1)
    parser.add_argument("--fics-over-2000", action="store_true", help="Filter FICS games > 2000 Elo")
    parser.add_argument("--lichess-month", type=str, default="2013-01", help="Lichess year-month (YYYY-MM)")

    args = parser.parse_args()

    if args.source in ("kasparov", "carlsen"):
        download_pgn_mentor(args.source)
    elif args.source == "fics":
        download_fics_games(
            year=args.fics_year,
            month=args.fics_month,
            rating_over_2000=args.fics_over_2000,
        )
    elif args.source == "lichess":
        download_lichess_games(year_month=args.lichess_month)
