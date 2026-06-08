#!/usr/bin/env python3

"""
GUI launcher for the instructor report and grade exports.
"""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import logging
from instructor_report import (
    read_review,
    read_roster,
    build_enhanced_report,
    build_consolidated_gradebook,
    build_d2l_grade_import,
    build_word_mail_merge_export,
    write_word_mail_merge_template,
)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def pick_file(title: str) -> Path | None:
    """Thin wrapper around askopenfilename → pathlib.Path."""
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Spreadsheet files", "*.xlsx *.xls *.csv"), ("All files", "*.*")]
    )
    return Path(path) if path else None


def pick_folder(title: str) -> Path | None:
    """Pick a folder path."""
    folder = filedialog.askdirectory(
        title=title
    )
    return Path(folder) if folder else None


def find_available_output_path(path: Path) -> Path:
    """Return the first available sibling path by appending a numeric suffix."""
    if not path.exists():
        return path

    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find an available output filename for {path}")


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


def pick_optional_written_report_source() -> Path | None:
    """Pick optional written report grade input."""
    selected = filedialog.askopenfilename(
        title="Optional: select written report grade file (.xlsx/.xls/.csv). Cancel to choose a folder of feedback_group HTML files.",
        filetypes=[("Spreadsheet files", "*.xlsx *.xls *.csv"), ("All files", "*.*")],
    )
    if selected:
        return Path(selected)

    folder = filedialog.askdirectory(
        title="Optional: select folder containing feedback_group_*.html files. Cancel to skip consolidated grade export."
    )
    return Path(folder) if folder else None


def main() -> None:
    root = tk.Tk()
    root.withdraw()  # hide the empty Tk window

    # Show info dialog about enhanced features
    info_msg = """Peer Review Report Generator

This tool now includes:
• Inter-rater reliability (ICC) calculations
• Cronbach's alpha for internal consistency  
• Individual reviewer quality scores
• Bias and consistency metrics

The enhanced report will provide detailed reliability 
assessments for both groups and individual reviewers.

Click OK to continue..."""
    
    messagebox.showinfo("Report Generator", info_msg)

    reviews = pick_file("Select the peer-review file from Microsoft Forms")
    if not reviews:
        messagebox.showwarning("Canceled", "No review file selected.")
        return

    roster = pick_file("Select the D2L grading export / student roster file")
    if not roster:
        messagebox.showwarning("Canceled", "No roster file selected.")
        return

    # choose where to save the report
    out_path = filedialog.asksaveasfilename(
        title="Save instructor report as…",
        defaultextension=".html",
        filetypes=[("HTML report", "*.html")],
        initialfile="Instructor_Report.html"
    )
    if not out_path:
        messagebox.showwarning("Canceled", "No output location chosen.")
        return
    out_path = Path(out_path)

    export_consolidated = messagebox.askyesno(
        "Consolidated Grade Export",
        "Do you want to create a consolidated grade export for mail merge and D2L?"
    )

    written_report_source = None
    enhanced_group_report_source = None
    gradebook_path = None
    if export_consolidated:
        written_report_source = pick_folder(
            "Select the folder containing feedback_Group-#.html files for written report grades"
        )
        if written_report_source is None:
            written_report_source = pick_optional_written_report_source()

        enhanced_group_report_source = pick_folder(
            "Select the folder containing Group_#Peer_Review_Report.html files"
        )

        gradebook_out = filedialog.asksaveasfilename(
            title="Save consolidated gradebook as…",
            defaultextension=".xlsx",
            filetypes=[("Excel workbook", "*.xlsx")],
            initialfile="Consolidated_Grades.xlsx"
        )
        if not gradebook_out:
            messagebox.showwarning("Canceled", "No consolidated gradebook location chosen.")
            return
        gradebook_path = Path(gradebook_out)

    try:
        # Show progress window
        progress_window, progress_bar = create_progress_window()
        root.update()

        LOG.info("Reading workbooks…")
        review_df = read_review(reviews)
        roster_df = read_roster(roster)

        LOG.info("Building instructor report with reliability metrics…")
        html = build_enhanced_report(review_df, roster_df)

        # Close progress window
        progress_bar.stop()
        progress_window.destroy()

        out_path.write_text(html, encoding="utf-8")

        export_notes = ""
        if export_consolidated and gradebook_path is not None:
            consolidated = build_consolidated_gradebook(
                review_df,
                roster_df,
                written_report_source,
                enhanced_group_report_source,
            )
            consolidated.to_excel(gradebook_path, index=False)

            d2l_path = gradebook_path.with_name(f"{gradebook_path.stem}_D2L_Import.csv")
            d2l_import = build_d2l_grade_import(consolidated)
            d2l_import.to_csv(d2l_path, index=False)

            mail_merge_csv = gradebook_path.with_suffix(".csv")
            consolidated.to_csv(mail_merge_csv, index=False)

            word_merge_path = gradebook_path.with_name(f"{gradebook_path.stem}_Word_Mail_Merge.xlsx")
            word_merge = build_word_mail_merge_export(consolidated)
            word_merge.to_excel(word_merge_path, index=False)

            word_template_path = gradebook_path.with_name(f"{gradebook_path.stem}_mail_merge.docx")
            word_template_note = ""
            try:
                write_word_mail_merge_template(word_template_path)
            except PermissionError:
                fallback_template_path = find_available_output_path(word_template_path)
                write_word_mail_merge_template(fallback_template_path)
                word_template_note = (
                    f"\nNote: {word_template_path.name} was open, so the mail merge template was saved as "
                    f"{fallback_template_path.name} instead."
                )
                word_template_path = fallback_template_path

            export_notes = f"""

Additional exports:
✓ Detailed gradebook: {gradebook_path}
✓ Mail merge CSV: {mail_merge_csv}
✓ Word mail merge workbook: {word_merge_path}
✓ Word mail merge template: {word_template_path}
✓ D2L import CSV: {d2l_path}
{word_template_note}

The D2L CSV uses the column header `Final Assignment Grade Points Grade`.
Rename that header if your D2L grade item uses a different exact name."""
        
        success_msg = f"""Instructor report generated successfully!

Saved to: {out_path}

The report includes:
✓ Traditional group statistics
✓ Inter-rater reliability (ICC) 
✓ Cronbach's alpha values
✓ Individual reviewer quality scores
✓ Bias and consistency metrics
{export_notes}

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
