"""
Download all accepted papers from an OpenReview venue (API v2).

Install:  pip install openreview-py
Usage:    python download_accepted.py --venue-id "colmweb.org/COLM/2025/Workshop/Social_Sim" --out SocialSim2025

Find your venue id in the URL of your venue page:
  https://openreview.net/group?id=<VENUE_ID>
"""

import argparse
import getpass
import os
import re

import openreview
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def slugify(text, max_len=80):
    text = re.sub(r"[^\w\s-]", "", text).strip()
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:max_len]


def get_value(content, key, default=""):
    """API v2 stores fields as {'value': ...}."""
    field = content.get(key)
    if isinstance(field, dict):
        return field.get("value", default)
    return field if field is not None else default


def accepted_by_venueid(client, venue_id):
    """Works once the post-decision stage has stamped accepted papers with the venue id."""
    return client.get_all_notes(content={"venueid": venue_id})


def accepted_by_decision(client, venue_id, submission_name="Submission"):
    """Fallback: read Decision notes directly (if post-decision hasn't been run yet)."""
    submissions = client.get_all_notes(
        invitation=f"{venue_id}/-/{submission_name}", details="replies"
    )
    accepted = []
    for note in submissions:
        for reply in note.details.get("replies", []):
            if any(inv.endswith("/-/Decision") for inv in reply.get("invitations", [])):
                decision = reply["content"].get("decision", {}).get("value", "")
                if "accept" in decision.lower():
                    accepted.append(note)
                    break
    return accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue-id", required=True)
    parser.add_argument("--out", default="accepted_papers")
    parser.add_argument("--username", default=os.environ.get("OPENREVIEW_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("OPENREVIEW_PASSWORD"))
    parser.add_argument("--supplementary", action="store_true",
                        help="Also download supplementary material if present")
    args = parser.parse_args()

    username = args.username or input("OpenReview username (email): ")
    password = args.password or getpass.getpass("OpenReview password: ")

    client = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=username,
        password=password,
    )

    notes = accepted_by_venueid(client, args.venue_id)
    if not notes:
        print("No papers found via venueid; falling back to Decision notes...")
        notes = accepted_by_decision(client, args.venue_id)

    print(f"Found {len(notes)} accepted papers.")
    os.makedirs(args.out, exist_ok=True)

    index_lines = ["number,title,authors,forum_url,file"]
    for note in sorted(notes, key=lambda n: n.number or 0):
        title = get_value(note.content, "title", note.id)
        authors = "; ".join(get_value(note.content, "authors", []) or [])
        fname = f"{note.number:03d}_{slugify(title)}.pdf" if note.number else f"{note.id}.pdf"
        path = os.path.join(args.out, fname)

        if get_value(note.content, "pdf"):
            try:
                pdf = client.get_attachment(id=note.id, field_name="pdf")
                with open(path, "wb") as f:
                    f.write(pdf)
                print(f"  ✓ {fname}")
            except Exception as e:
                print(f"  ✗ {title}: {e}")
                fname = ""
        else:
            print(f"  - No PDF for: {title}")
            fname = ""

        if args.supplementary and get_value(note.content, "supplementary_material"):
            try:
                supp = client.get_attachment(id=note.id, field_name="supplementary_material")
                ext = os.path.splitext(get_value(note.content, "supplementary_material"))[1] or ".zip"
                with open(os.path.join(args.out, fname.replace(".pdf", "") + "_supp" + ext), "wb") as f:
                    f.write(supp)
            except Exception as e:
                print(f"    (supplementary failed: {e})")

        safe = lambda s: '"' + str(s).replace('"', '""') + '"'
        index_lines.append(",".join([
            str(note.number), safe(title), safe(authors),
            f"https://openreview.net/forum?id={note.forum}", fname,
        ]))

    with open(os.path.join(args.out, "index.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))
    print(f"\nDone. PDFs and index.csv saved to ./{args.out}/")


if __name__ == "__main__":
    main()
