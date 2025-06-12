#!/usr/bin/env python3

"""
Enhanced GUI launcher for peer-evaluation report with reliability metrics
"""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import logging
from .enhanced_instructor_report import read_review, read_roster, build_enhanced_report

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def pick_file(title: str) -> Path | None:
    """Thin wrapper around askopenfilename → pathlib.Path."""
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    return Path(path) if path else None


def create_progress_window():
    """Create a simple progress window."""
    progress_window = tk.Toplevel()
    progress_window.title("Processing...")
    progress_window.geometry("300x100")
    progress_window.transient()
    progress_window.grab_set()
    
    tk.Label(progress_window, text="Calculating reliability metrics...").pack(pady=20)
    
    progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
    progress_bar.pack(pady=10, padx=20, fill='x')
    progress_bar.start()
    
    return progress_window, progress_bar


def main() -> None:
    root = tk.Tk()
    root.withdraw()  # hide the empty Tk window

    # Show info dialog about enhanced features
    info_msg = """Enhanced Peer Review Report Generator

This tool now includes:
• Inter-rater reliability (ICC) calculations
• Cronbach's alpha for internal consistency  
• Individual reviewer quality scores
• Bias and consistency metrics

The enhanced report will provide detailed reliability 
assessments for both groups and individual reviewers.

Click OK to continue..."""
    
    messagebox.showinfo("Enhanced Features", info_msg)

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
        title="Save enhanced report as…",
        defaultextension=".html",
        filetypes=[("HTML report", "*.html")],
        initialfile="enhanced_peer_report.html"
    )
    if not out_path:
        messagebox.showwarning("Canceled", "No output location chosen.")
        return
    out_path = Path(out_path)

    try:
        # Show progress window
        progress_window, progress_bar = create_progress_window()
        root.update()

        LOG.info("Reading workbooks…")
        review_df = read_review(reviews)
        roster_df = read_roster(roster)

        LOG.info("Building enhanced report with reliability metrics…")
        html = build_enhanced_report(review_df, roster_df)

        # Close progress window
        progress_bar.stop()
        progress_window.destroy()

        out_path.write_text(html, encoding="utf‑8")
        
        success_msg = f"""Enhanced report generated successfully!

Saved to: {out_path}

The report includes:
✓ Traditional group statistics
✓ Inter-rater reliability (ICC) 
✓ Cronbach's alpha values
✓ Individual reviewer quality scores
✓ Bias and consistency metrics

Open the HTML file in your browser to view the results."""

        messagebox.showinfo("Success", success_msg)
        
    except Exception as exc:
        # Close progress window if still open
        try:
            progress_bar.stop()
            progress_window.destroy()
        except:
            pass
            
        LOG.exception(exc)
        error_msg = f"""Error generating report:

{str(exc)}

Please check that:
• Excel files are not open in another program
• Files contain the expected column names
• At least 2 reviewers per group for reliability calculations"""
        
        messagebox.showerror("Error", error_msg)


if __name__ == "__main__":
    main()
