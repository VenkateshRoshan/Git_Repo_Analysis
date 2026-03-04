import argparse
import requests
import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import time
import os
import numpy as np

import aiohttp
import asyncio

from dotenv import load_dotenv
load_dotenv()

DEFAULT_DAYS = 90
DEFAULT_REPO = "PostHog/posthog"
DEFAULT_TOP_N = 5
DEFAULT_SHIPPING_WEIGHT = 0.4
DEFAULT_COLLAB_WEIGHT = 0.35
DEFAULT_INFLUENCE_WEIGHT = 0.25
BASE = "https://api.github.com"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyzing Github PRs to identify top contributors based on merged PRs, reviews, and influence."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Number of past days to analyze (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Repository name (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top contributors to the repository (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--shipping-weight",
        type=float,
        default=DEFAULT_SHIPPING_WEIGHT,
        help=f"Weight for shipping score ( b/w 0 and 1 ) (default: {DEFAULT_SHIPPING_WEIGHT})",
    )
    parser.add_argument(
        "--collab-weight",
        type=float,
        default=DEFAULT_COLLAB_WEIGHT,
        help=f"Weight for collaboration score ( b/w 0 and 1 ) (default: {DEFAULT_COLLAB_WEIGHT})",
    )
    parser.add_argument(
        "--influence-weight",
        type=float,
        default=DEFAULT_INFLUENCE_WEIGHT,
        help=f"Weight for influence score ( b/w 0 and 1 ) (default: {DEFAULT_INFLUENCE_WEIGHT})",
    )
    return parser.parse_args()

# Giving equality to all features so that no single feature is going to dominate the other features.
def normalize(vals):
    arr = np.array(vals, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros(len(vals)).tolist()
    return ((arr - lo) / (hi - lo) * 100).round(1).tolist()

def get_repo_prs(repo, headers, cutoff):
    all_prs = []
    
    page = 1
    # Github API wont give all PRs in one time, it splits them into chunks of 100 which called as pages.
    while True:
        r = requests.get(
            f"{BASE}/repos/{repo}/pulls",
            headers=headers,
            params={
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": 100,
                "page": page,
            },
        )

        r.raise_for_status()
        prs = r.json()

        if not prs:
            break

        for pr in prs:
            if pr.get("merged_at"):
                merged = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
                if merged >= cutoff:
                    all_prs.append(pr)

        last_updated = datetime.fromisoformat(prs[-1]["updated_at"].replace("Z", "+00:00"))
        if last_updated < cutoff or len(prs) < 100:
            break

        page += 1
        time.sleep(0.1)

        print(f"\rFetched page {page} with {len(prs)} PRs (Total merged PRs so far: {len(all_prs)})", end="")

    return all_prs

async def get_pr_reviews(pr, repo, headers, session, eng):
    author = pr["user"]["login"]
    try:
        async with session.get(
            f"{BASE}/repos/{repo}/pulls/{pr['number']}/reviews"
        ) as r:
            reviews = await r.json()

        seen_reviewers = set()
        for rev in reviews:
            reviewer = rev["user"]["login"]
            if reviewer != author and reviewer not in seen_reviewers:
                eng[reviewer]["reviews_given"] += 1
                seen_reviewers.add(reviewer)
    except Exception:
        pass

async def main():

    """
        1. Fetch merged PRs data
        2. Extract the data
        3. Calculate scores for each contributor based on:
            - Shipping Score (40%): Based on the number of PRs merged and their sizes.
            - Collaboration Score (35%): Based on the number of reviews given to others' PRs.
            - Influence Score (25%): Based on the number of comments received on their PRs.
        4. Normalize the scores to a 0-100 scale.
        5. Output the top N contributors with their scores and recent PR details.
    """

    args = parse_args()
    token = os.getenv("GITHUB_ACCESS_TOKEN")

    # Validating the token
    if not token:
        raise ValueError("GITHUB_ACCESS_TOKEN not found. Add it to your .env file.")
    
    days = args.days
    repo = args.repo
    top_n = args.top_n
    shipping_weight = args.shipping_weight
    collab_weight = args.collab_weight
    influence_weight = args.influence_weight

    total_weight = shipping_weight + collab_weight + influence_weight
    if not np.isclose(total_weight, 1.0):
        raise ValueError("The scoring weights must sum to 1.0.")

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # converting days to a datetime
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    all_prs = []

    print(f"Fetching merged PRs for {repo} (last {days} days)...")

    all_prs = get_repo_prs(repo, headers, cutoff)

    print(f"\n{len(all_prs)} merged PRs found\n")

    eng = defaultdict(
        lambda: {
            "prs_merged": 0,
            "additions": 0,
            "deletions": 0,
            "reviews_given": 0,
            "pr_sizes": [],
            "comments_received": 0,
            "recent_prs": [],
        }
    )

    # considering each PR's contribution like this:
    # 1. PRs merged
    # 2. additions and deletions in the code / doc
    # 3. comments / discussions on the PR
    # 4. reviewer who reviewed

    for i, pr in enumerate(all_prs):
        author = pr["user"]["login"]
        adds = pr.get("additions", 0)
        dels = pr.get("deletions", 0)

        eng[author]["prs_merged"] += 1
        eng[author]["additions"] += adds # no of lines added in the code
        eng[author]["deletions"] += dels # no of lines deleted in the code
        eng[author]["pr_sizes"].append(adds + dels) # size of the PR = additions + deletions
        eng[author]["comments_received"] += pr.get("review_comments", 0)
        eng[author]["recent_prs"].append(
            {
                "number": pr["number"],
                "title": pr["title"][:80],
                "merged_at": pr["merged_at"],
                "additions": adds,
                "deletions": dels,
                "review_comments": pr.get("review_comments", 0),
            }
        )

        # Sequential approach
        # # To know who reviewed the PR
        # try:
        #     reviews = requests.get(
        #         f"{BASE}/repos/{repo}/pulls/{pr['number']}/reviews",
        #         headers=headers,
        #     ).json()

        #     seen_reviewers = set()
        #     for rev in reviews:
        #         reviewer = rev["user"]["login"]
        #         if reviewer != author and reviewer not in seen_reviewers:
        #             eng[reviewer]["reviews_given"] += 1
        #             seen_reviewers.add(reviewer)

        # except Exception:
        #     pass

        # if (i + 1) % 100 == 0:
        #     print(f"\rProcessed {i + 1}/{len(all_prs)} PRs", end="")
        #     time.sleep(0.3)

    # Async approach
    print("Fetching reviews for all PRs...")
    async with aiohttp.ClientSession(headers=headers) as session:
        await asyncio.gather(*[
            get_pr_reviews(pr, repo, headers, session, eng)
            for pr in all_prs
        ])
    print("Reviews fetched\n")

    logins = list(eng.keys())
    ship_raw = [eng[login]["prs_merged"] for login in logins]
    collab_raw = [eng[login]["reviews_given"] for login in logins]
    infl_raw = [eng[login]["comments_received"] + eng[login]["prs_merged"] for login in logins]

    # normalizing values to a 0-100 scale, to give equality to all features.
    ship_n = normalize(ship_raw)
    collab_n = normalize(collab_raw)
    infl_n = normalize(infl_raw)

    for i, login in enumerate(logins):
        engineer = eng[login]
        engineer["login"] = login
        engineer["shipping_score"] = round(ship_n[i], 2)
        engineer["collab_score"] = round(collab_n[i], 2)
        engineer["influence_score"] = round(infl_n[i], 2)

        # Updating the total score by weights
        engineer["total_score"] = round(
            ship_n[i] * shipping_weight
            + collab_n[i] * collab_weight
            + infl_n[i] * influence_weight,
            2
        )
        engineer["avg_pr_size"] = (
            round(sum(engineer["pr_sizes"]) / len(engineer["pr_sizes"])) if engineer["pr_sizes"] else 0
        )
        engineer.pop("pr_sizes")
        engineer["recent_prs"] = sorted(
            engineer["recent_prs"], key=lambda item: item["review_comments"], reverse=True
        )[:5]

    top_contributors = sorted(eng.values(), key=lambda item: item["total_score"], reverse=True)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days_analyzed": days,
        "repo": repo,
        "top_n": top_n,
        "weights": {
            "shipping": shipping_weight,
            "collaboration": collab_weight,
            "influence": influence_weight,
        },
        "total_prs": len(all_prs),
        "total_contributors": len(eng),
        "topN": top_contributors,
    }

    with open("data.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nUpdated the data.json file with the latest analysis results.\n")
    for engineer in top_contributors:
        print(f"Engineer Name: {engineer['login']} | Total Score: {engineer['total_score']} | Shipping Score: {engineer['shipping_score']} | Collaboration Score: {engineer['collab_score']} | Influence Score: {engineer['influence_score']}")

if __name__ == "__main__":
    asyncio.run(main())
