# this file relys on the `instructor_report` module to read the Excel files and build the report.
# gui_launcher.py  ──  run with:  python gui_launcher.py
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import logging
from instructor_report import read_review, read_roster, build_report  # import core

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def pick_file(title: str) -> Path | None:
    """Thin wrapper around askopenfilename → pathlib.Path."""
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    return Path(path) if path else None


def main() -> None:
    root = tk.Tk()
    root.withdraw()                 # hide the empty Tk window

    reviews = pick_file("Select the **Peer‑Review** Excel file")
    if not reviews:
        messagebox.showwarning("Canceled", "No review file selected.")
        return

    roster = pick_file("Select the **Student‑List** Excel file")
    if not roster:
        messagebox.showwarning("Canceled", "No roster file selected.")
        return

    # choose where to save the report
    out_path = filedialog.asksaveasfilename(
        title="Save report as…",
        defaultextension=".html",
        filetypes=[("HTML report", "*.html")],
    )
    if not out_path:
        messagebox.showwarning("Canceled", "No output location chosen.")
        return
    out_path = Path(out_path)

    try:
        LOG.info("Reading workbooks…")
        review_df  = read_review(reviews)
        roster_df  = read_roster(roster)

        LOG.info("Building report…")
        html = build_report(review_df, roster_df)

        out_path.write_text(html, encoding="utf‑8")
        messagebox.showinfo("Done", f"Report saved to\n{out_path}")
    except Exception as exc:
        LOG.exception(exc)
        messagebox.showerror("Error", str(exc))


if __name__ == "__main__":
    main()
