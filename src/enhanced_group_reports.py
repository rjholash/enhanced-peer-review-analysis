#!/usr/bin/env python3

"""
Enhanced Group Report Generator with Reliability Metrics
Integrates group reliability data into individual group reports
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Import the enhanced analysis functions
sys.path.append(str(Path(__file__).parent))
from .enhanced_instructor_report import (
    read_review, read_roster, calculate_reviewer_scores, 
    calculate_cronbach_alpha, calculate_icc
)

def create_enhanced_group_reports():
    """Create individual group reports with reliability metrics."""
    
    # Create the root Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog box to select the Excel file for the DataFrame
    file_path = filedialog.askopenfilename(
        filetypes=[('Excel Files', '*.xlsx')],
        title='Select the Downloaded Peer Review File'
    )

    # Check if a file was selected
    if not file_path:
        print("No file selected. Exiting the program.")
        return

    # Open a file dialog box to select the student roster file
    roster_path = filedialog.askopenfilename(
        filetypes=[('Excel Files', '*.xlsx')],
        title='Select the Student Roster File'
    )

    if not roster_path:
        print("No roster file selected. Exiting the program.")
        return

    try:
        # Read the files
        df = read_review(Path(file_path))
        roster = read_roster(Path(roster_path))
        
        # Merge data
        merged = df.merge(
            roster[["email_clean", "student_id_clean", "Group"]],
            on="email_clean",
            how="left"
        )
        
        # Calculate duration
        merged["DurationMin"] = (
            pd.to_datetime(merged["Completion time"]) -
            pd.to_datetime(merged["Start time"])
        ).dt.total_seconds() / 60
        
        # Rating columns
        rating_cols = [
            'Video_Quality', 'Presenters', 'Explanation', 'Mechanism', 
            'Side_Effects', 'Bias', 'Critical_review', 'Study_Quality', 
            'Study_participants'
        ]
        
        # Calculate reliability metrics
        reviewer_scores = calculate_reviewer_scores(merged, rating_cols)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error reading files: {str(e)}")
        return

    # Display a message box to prompt the user to select the destination folder
    messagebox.showinfo(
        'Select Destination Folder', 
        'Please select a destination folder for the enhanced HTML reports.'
    )

    # Open a file dialog box to select the destination folder
    folder_path = filedialog.askdirectory()

    # Check if a folder path was selected
    if not folder_path:
        print("No folder path selected. Group reports not saved.")
        return

    # Group the data
    grouped = merged.groupby('Group#_reviewing')

    # Iterate through each group
    for group, data in grouped:
        try:
            # Calculate traditional statistics
            means = data[rating_cols].agg(['mean', 'std']).round(2)
            score = means.loc['mean']
            total_score = score.sum()
            mark = round(total_score / 90 * 100, 2)
            
            # Create D2L marks
            final_mark = round(mark * 0.4, 2)
            total_mark = round(final_mark + 20, 2)

            # Calculate reliability metrics for this group
            group_reliability = {}
            if len(data) >= 2:
                group_matrix = data[rating_cols].values
                
                # Cronbach's alpha
                alpha = calculate_cronbach_alpha(pd.DataFrame(group_matrix))
                group_reliability['cronbach_alpha'] = round(alpha, 3) if not pd.isna(alpha) else 'N/A'
                
                # ICC
                if len(rating_cols) >= 2:
                    icc_data = pd.DataFrame(group_matrix.T)
                    icc_val, icc_p = calculate_icc(icc_data)
                    group_reliability['icc_value'] = round(icc_val, 3) if not pd.isna(icc_val) else 'N/A'
                    group_reliability['icc_p_value'] = round(icc_p, 3) if not pd.isna(icc_p) else 'N/A'
                else:
                    group_reliability['icc_value'] = 'N/A'
                    group_reliability['icc_p_value'] = 'N/A'
                
                # Number of reviewers
                group_reliability['num_reviewers'] = len(data)
                
                # Average review time
                group_reliability['avg_review_time'] = round(data['DurationMin'].mean(), 1)
                
                # Standard deviation of scores
                group_reliability['score_std'] = round(data[rating_cols].mean(axis=1).std(), 2)
                
            else:
                group_reliability = {
                    'cronbach_alpha': 'N/A',
                    'icc_value': 'N/A', 
                    'icc_p_value': 'N/A',
                    'num_reviewers': len(data),
                    'avg_review_time': round(data['DurationMin'].mean(), 1) if len(data) > 0 else 0,
                    'score_std': 'N/A'
                }

            # Get reviewer details for this group (calculate averages only)
            group_reviewer_stats = {
                'avg_reliability_score': 0,
                'num_reliable_reviewers': 0,
                'total_reviewers': 0
            }
            
            if not reviewer_scores.empty:
                group_reviewer_data = reviewer_scores[
                    reviewer_scores['reviewer_email'].isin(data['email_clean'])
                ]
                if not group_reviewer_data.empty:
                    reliability_scores = group_reviewer_data['reliability_score'].tolist()
                    group_reviewer_stats['avg_reliability_score'] = round(np.mean(reliability_scores), 1)
                    group_reviewer_stats['num_reliable_reviewers'] = len([s for s in reliability_scores if s >= 75])
                    group_reviewer_stats['total_reviewers'] = len(reliability_scores)

            # Prepare feedback data (convert Series to lists)
            feedback_data = {
                'Comments_ProductionQuality': [],
                'Comment_Information': [],
                'Comment_Research': []
            }
            
            # Safely extract feedback comments
            try:
                if 'Comments_ProductionQuality' in data.columns:
                    feedback_data['Comments_ProductionQuality'] = [
                        str(comment) for comment in data['Comments_ProductionQuality'].dropna() 
                        if str(comment) != 'nan' and str(comment).strip()
                    ]
                if 'Comment_Information' in data.columns:
                    feedback_data['Comment_Information'] = [
                        str(comment) for comment in data['Comment_Information'].dropna() 
                        if str(comment) != 'nan' and str(comment).strip()
                    ]
                if 'Comment_Research' in data.columns:
                    feedback_data['Comment_Research'] = [
                        str(comment) for comment in data['Comment_Research'].dropna() 
                        if str(comment) != 'nan' and str(comment).strip()
                    ]
            except Exception as feedback_error:
                print(f"Warning: Error processing feedback for Group {group}: {feedback_error}")

            # Build HTML content using f-strings (no emojis, no templates)
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Group {group} Report</title>
    <style>
        body {{
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            font-family: 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #007acc, #0056b3);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .metric-box {{
            background: #f8f9fa;
            border-left: 4px solid #007acc;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .score-highlight {{
            font-size: 1.2em;
            font-weight: bold;
            color: #007acc;
        }}
        .reliability-good {{ color: #28a745; }}
        .reliability-moderate {{ color: #ffc107; }}
        .reliability-poor {{ color: #dc3545; }}
        .feedback-section {{
            background: #e7f3ff;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Group {group} - Enhanced Performance Report</h1>
        <p>Comprehensive analysis including peer review quality metrics</p>
    </div>

    <div class="metric-box">
        <h2>Final Scores</h2>
        <p class="score-highlight">Peer Review Grade: {mark}%</p>
        <p>Weighted Score (40%): {final_mark}%</p>
        <p>Completion Bonus (20%): 20%</p>
        <p class="score-highlight">Total D2L Mark: {total_mark}/60</p>
    </div>

    <div class="metric-box">
        <h2>Review Quality Analysis</h2>"""
            
            if group_reliability['num_reviewers'] >= 2:
                sample_size_text = "Good sample size" if group_reliability['num_reviewers'] >= 3 else "Minimum for analysis"
                
                # Determine reliability interpretations
                alpha_interp = "Insufficient data"
                if group_reliability['cronbach_alpha'] != 'N/A':
                    if group_reliability['cronbach_alpha'] >= 0.8:
                        alpha_interp = "Excellent consistency"
                    elif group_reliability['cronbach_alpha'] >= 0.7:
                        alpha_interp = "Good consistency"
                    else:
                        alpha_interp = "Needs improvement"
                
                icc_interp = "Insufficient data"
                if group_reliability['icc_value'] != 'N/A':
                    if group_reliability['icc_value'] >= 0.75:
                        icc_interp = "High agreement"
                    elif group_reliability['icc_value'] >= 0.5:
                        icc_interp = "Moderate agreement"
                    else:
                        icc_interp = "Low agreement"
                
                time_interp = "Adequate consideration" if group_reliability['avg_review_time'] >= 2 else "May need more time"
                variance_interp = "Low variance - high agreement" if group_reliability['score_std'] != 'N/A' and group_reliability['score_std'] < 5 else "Higher variance in ratings"

                html_content += f"""
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>Number of Reviewers</td>
                <td>{group_reliability['num_reviewers']}</td>
                <td>{sample_size_text}</td>
            </tr>
            <tr>
                <td>Cronbach's Alpha</td>
                <td>{group_reliability['cronbach_alpha']}</td>
                <td>{alpha_interp}</td>
            </tr>
            <tr>
                <td>Inter-Rater Reliability (ICC)</td>
                <td>{group_reliability['icc_value']}</td>
                <td>{icc_interp}</td>
            </tr>
            <tr>
                <td>Average Review Time</td>
                <td>{group_reliability['avg_review_time']} minutes</td>
                <td>{time_interp}</td>
            </tr>
            <tr>
                <td>Score Variability</td>
                <td>{group_reliability['score_std']}</td>
                <td>{variance_interp}</td>
            </tr>
        </table>"""
            else:
                plural = "s" if group_reliability['num_reviewers'] != 1 else ""
                html_content += f"""
        <p>Limited reliability analysis available (only {group_reliability['num_reviewers']} reviewer{plural})</p>"""
            
            html_content += """
    </div>"""

            # Add average reviewer quality information (no individual IDs)
            if group_reviewer_stats['total_reviewers'] > 0:
                # Determine overall quality level
                avg_score = group_reviewer_stats['avg_reliability_score']
                if avg_score >= 85:
                    quality_level = "Excellent"
                    quality_class = "reliability-good"
                elif avg_score >= 75:
                    quality_level = "Very Good"
                    quality_class = "reliability-good"
                elif avg_score >= 65:
                    quality_level = "Good"
                    quality_class = "reliability-moderate"
                elif avg_score >= 55:
                    quality_level = "Satisfactory"
                    quality_class = "reliability-moderate"
                else:
                    quality_level = "Needs Improvement"
                    quality_class = "reliability-poor"
                
                reliable_percentage = round((group_reviewer_stats['num_reliable_reviewers'] / group_reviewer_stats['total_reviewers']) * 100, 1)
                
                html_content += f"""
    <div class="metric-box">
        <h2>Reviewer Quality Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Assessment</th>
            </tr>
            <tr>
                <td>Average Reviewer Reliability</td>
                <td>{avg_score}/100</td>
                <td class="{quality_class}">{quality_level}</td>
            </tr>
            <tr>
                <td>High-Quality Reviewers</td>
                <td>{group_reviewer_stats['num_reliable_reviewers']} of {group_reviewer_stats['total_reviewers']}</td>
                <td>{reliable_percentage}% rated as reliable (75+ score)</td>
            </tr>
        </table>
        <p><em>This reflects the overall quality of peer reviews your group received, based on reviewer consistency and thoroughness.</em></p>
    </div>"""

            # Detailed Section Scores
            html_content += """
    <div class="metric-box">
        <h2>Detailed Section Scores</h2>
        <table>
            <thead>
                <tr>
                    <th>Assessment Section</th>
                    <th>Mean Score</th>
                    <th>Standard Deviation</th>
                </tr>
            </thead>
            <tbody>"""
            
            for column in means.columns:
                section_name = column.replace('_', ' ').title()
                mean_score = means.loc['mean', column]
                std_score = means.loc['std', column]
                html_content += f"""
                <tr>
                    <td>{section_name}</td>
                    <td>{mean_score}/10</td>
                    <td>{std_score}</td>
                </tr>"""
            
            html_content += """
            </tbody>
        </table>
    </div>

    <div class="feedback-section">
        <h2>Peer Feedback</h2>
        
        <h3>Presentation Quality & Production</h3>"""
            
            if feedback_data['Comments_ProductionQuality']:
                for feedback in feedback_data['Comments_ProductionQuality']:
                    html_content += f"        <p>• {feedback}</p>\n"
            else:
                html_content += "        <p><em>No feedback provided for this section.</em></p>"

            html_content += """
        
        <h3>Content & Information Quality</h3>"""
            
            if feedback_data['Comment_Information']:
                for feedback in feedback_data['Comment_Information']:
                    html_content += f"        <p>• {feedback}</p>\n"
            else:
                html_content += "        <p><em>No feedback provided for this section.</em></p>"

            html_content += """
        
        <h3>Research Quality & Evidence</h3>"""
            
            if feedback_data['Comment_Research']:
                for feedback in feedback_data['Comment_Research']:
                    html_content += f"        <p>• {feedback}</p>\n"
            else:
                html_content += "        <p><em>No feedback provided for this section.</em></p>"

            html_content += f"""
    </div>

    <div class="metric-box">
        <h2>Understanding Your Report</h2>
        <p><strong>Cronbach's Alpha:</strong> Measures how consistently reviewers rated different aspects. Higher values (>0.7) indicate reviewers agreed on what constitutes quality.</p>
        <p><strong>ICC (Inter-Rater Reliability):</strong> Measures overall agreement between reviewers. Values >0.75 indicate strong consensus.</p>
        <p><strong>Reliability Scores:</strong> Individual reviewer quality based on consistency with peers and time spent reviewing.</p>
        <p><strong>Grade Calculation:</strong> Your peer review score ({mark}%) × 40% + completion bonus (20%) = {total_mark}/60 total points.</p>
    </div>

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; text-align: center;">
        <p>Generated by Enhanced Peer Review Analysis System | Report includes advanced reliability metrics</p>
    </footer>
</body>
</html>"""

            # Save the file
            file_path = f"{folder_path}/enhanced_group_{group}_report.html"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"Enhanced Group {group} report saved at {file_path}")
            
        except Exception as e:
            print(f"Error processing Group {group}: {str(e)}")
            continue

    messagebox.showinfo("Success", f"Enhanced group reports generated successfully!\n\nSaved to: {folder_path}")


if __name__ == "__main__":
    create_enhanced_group_reports()
