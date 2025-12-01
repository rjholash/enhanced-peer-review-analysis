#The scrip requires a CSV file with columns for student emails, group names, and URLs.
#Email, Group, URL (are the expected columns, but the script can adapt to variations in header names).

import csv
import os
import random
from collections import defaultdict
from tkinter import Tk
from tkinter.filedialog import askopenfilename

RANDOM_SEED = 213  # change this if you want a different pseudo-random pattern


def choose_columns(fieldnames):
    """
    Given a list of cleaned header names, pick:
    - email_col: header containing 'email'
    - group_col: header containing 'group'
    - url_col:  the remaining column (for your current file: the URL column)
    """
    lower = [h.lower() for h in fieldnames]

    # Email column: must contain 'email'
    email_candidates = [h for h in fieldnames if "email" in h.lower()]
    if not email_candidates:
        raise KeyError(f"No email-like column found in headers: {fieldnames}")
    email_col = email_candidates[0]

    # Group column: must contain 'group'
    group_candidates = [h for h in fieldnames if "group" in h.lower()]
    if not group_candidates:
        raise KeyError(f"No group-like column found in headers: {fieldnames}")
    group_col = group_candidates[0]

    # URL column: any remaining column
    remaining = [h for h in fieldnames if h not in (email_col, group_col)]
    if len(remaining) != 1:
        raise KeyError(
            f"Could not uniquely identify URL column. Headers were: {fieldnames}. "
            f"Email: {email_col}, Group: {group_col}, Remaining: {remaining}"
        )
    url_col = remaining[0]

    return email_col, group_col, url_col


def main():
    random.seed(RANDOM_SEED)

    # --- GUI file picker (Finder dialog) ---
    root = Tk()
    root.withdraw()  # hide the root window

    csv_path = askopenfilename(
        title="Select the email/group/URL CSV file",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    if not csv_path:
        print("No file selected. Exiting.")
        return

    csv_path = os.path.expanduser(csv_path)
    print(f"Reading input from: {csv_path}")

    # --- Load students ---
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV appears to have no header row.")

        # Clean header names (strip BOM and whitespace)
        fieldnames = [h.replace("\ufeff", "").strip() for h in reader.fieldnames]
        reader.fieldnames = fieldnames

        email_col, group_col, url_col = choose_columns(fieldnames)

        students = []
        for row in reader:
            if not any(row.values()):
                continue
            students.append({
                "email": row[email_col].strip(),
                "group": row[group_col].strip(),
                "url":   row[url_col].strip(),
            })

    if not students:
        raise ValueError("No student rows found in the CSV.")

    # --- Build group → URL mapping ---
    group_to_url = {}
    for s in students:
        g = s["group"]
        u = s["url"]
        if g and u:
            group_to_url[g] = u

    groups = sorted(group_to_url.keys())
    num_students = len(students)
    num_groups = len(groups)
    total_reviews = 2 * num_students

    print(f"\nLoaded {num_students} students and {num_groups} groups.")
    print(f"Total review slots: {total_reviews}")

    # --- Balanced target counts per group ---
    base = total_reviews // num_groups
    remainder = total_reviews % num_groups

    # Bucket: each group appears base or base+1 times
    bucket = []
    for i, g in enumerate(groups):
        target_count = base + (1 if i < remainder else 0)
        bucket.extend([g] * target_count)

    if len(bucket) != total_reviews:
        raise RuntimeError(
            f"Bucket size mismatch: expected {total_reviews}, got {len(bucket)}."
        )

    random.shuffle(bucket)

    # --- Assign reviews ---
    student_indices = list(range(num_students))
    random.shuffle(student_indices)

    for idx in student_indices:
        s = students[idx]
        own_group = s["group"]

        # First review group: cannot be own_group
        candidates_idx = [i for i, g in enumerate(bucket) if g != own_group]
        if not candidates_idx:
            raise RuntimeError("No valid group available for first review.")
        chosen_i = random.choice(candidates_idx)
        g1 = bucket.pop(chosen_i)

        # Second review group: cannot be own_group or g1
        candidates_idx = [i for i, g in enumerate(bucket)
                          if g != own_group and g != g1]
        if not candidates_idx:
            # Fallback: relax g1 constraint (still avoid own_group)
            candidates_idx = [i for i, g in enumerate(bucket) if g != own_group]
        if not candidates_idx:
            raise RuntimeError("No valid group available for second review.")
        chosen_j = random.choice(candidates_idx)
        g2 = bucket.pop(chosen_j)

        s["review_group_1"] = g1
        s["review_group_2"] = g2

    # --- Map groups → URLs for review targets ---
    for s in students:
        s["review_url_1"] = group_to_url[s["review_group_1"]]
        s["review_url_2"] = group_to_url[s["review_group_2"]]

    # --- Write output CSV next to input ---
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_name = f"{base_name}-with-reviews.csv"
    output_path = os.path.join(os.path.dirname(csv_path), output_name)

    fieldnames_out = [
        "Email",
        "Product Review Groups",
        "OriginalURL",
        "ReviewGroup1",
        "ReviewURL1",
        "ReviewGroup2",
        "ReviewURL2",
    ]

    # Try to keep your original column names where possible
    group_header = None
    url_header = None
    for h in fieldnames:
        if "group" in h.lower():
            group_header = h
        if "http" in h.lower() or "url" in h.lower() or "grade" in h.lower():
            url_header = h
    if group_header is None:
        group_header = "Product Review Groups"
    if url_header is None:
        url_header = "URL"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_out)
        writer.writeheader()
        for s in students:
            writer.writerow({
                "Email":                s["email"],
                "Product Review Groups": s["group"],
                "OriginalURL":          s["url"],
                "ReviewGroup1":         s["review_group_1"],
                "ReviewURL1":           s["review_url_1"],
                "ReviewGroup2":         s["review_group_2"],
                "ReviewURL2":           s["review_url_2"],
            })

    # --- Summary of distribution ---
    counts = defaultdict(int)
    for s in students:
        counts[s["review_group_1"]] += 1
        counts[s["review_group_2"]] += 1

    print("\nReview counts per group:")
    for g in groups:
        print(f"{g}: {counts[g]}")

    print(f"\nDone. Output written to:\n{output_path}")


if __name__ == "__main__":
    main()