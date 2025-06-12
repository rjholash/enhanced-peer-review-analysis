#!/usr/bin/env python3

"""
Enhanced Instructor peer‑evaluation report generator with reliability metrics
Includes inter-rater reliability (ICC) and Cronbach's alpha calculations
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from jinja2 import Template
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ‑ %(levelname)s ‑ %(message)s",
)
LOG = logging.getLogger(__name__)


# ──────────────────────────────── Reliability Calculations ──────────────────────────────── #

def calculate_cronbach_alpha(data: pd.DataFrame) -> float:
    """
    Calculate Cronbach's alpha for internal consistency.
    
    Args:
        data: DataFrame with items as columns and observations as rows
        
    Returns:
        Cronbach's alpha value (0-1, higher is better consistency)
    """
    if data.shape[1] < 2:
        return np.nan
    
    # Remove rows with any missing values
    data_clean = data.dropna()
    if data_clean.empty or data_clean.shape[0] < 2:
        return np.nan
    
    k = data_clean.shape[1]  # number of items
    item_vars = data_clean.var(axis=0, ddof=1)
    total_var = data_clean.sum(axis=1).var(ddof=1)
    
    if total_var == 0:
        return np.nan
    
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return max(0, alpha)  # Alpha can be negative, but we'll floor at 0


def calculate_icc(data: pd.DataFrame, icc_type: str = 'ICC(2,1)') -> tuple[float, float]:
    """
    Calculate Intraclass Correlation Coefficient for inter-rater reliability.
    
    Args:
        data: DataFrame with raters as columns and subjects as rows
        icc_type: Type of ICC to calculate
        
    Returns:
        Tuple of (ICC value, F-statistic p-value)
    """
    if data.shape[1] < 2 or data.shape[0] < 2:
        return np.nan, np.nan
    
    # Remove rows with any missing values
    data_clean = data.dropna()
    if data_clean.empty or data_clean.shape[0] < 2:
        return np.nan, np.nan
    
    n_subjects = data_clean.shape[0]
    n_raters = data_clean.shape[1]
    
    # Calculate means
    subject_means = data_clean.mean(axis=1)
    rater_means = data_clean.mean(axis=0)
    grand_mean = data_clean.values.mean()
    
    # Sum of squares calculations
    ss_total = np.sum((data_clean.values - grand_mean) ** 2)
    ss_between_subjects = n_raters * np.sum((subject_means - grand_mean) ** 2)
    ss_between_raters = n_subjects * np.sum((rater_means - grand_mean) ** 2)
    ss_error = ss_total - ss_between_subjects - ss_between_raters
    
    # Mean squares
    ms_between_subjects = ss_between_subjects / (n_subjects - 1)
    ms_between_raters = ss_between_raters / (n_raters - 1)
    ms_error = ss_error / ((n_subjects - 1) * (n_raters - 1))
    
    # ICC(2,1) - Two-way random effects, single measurement, absolute agreement
    if ms_error == 0:
        return np.nan, np.nan
    
    icc = (ms_between_subjects - ms_error) / (ms_between_subjects + (n_raters - 1) * ms_error)
    
    # F-test for significance
    f_stat = ms_between_subjects / ms_error
    df1 = n_subjects - 1
    df2 = (n_subjects - 1) * (n_raters - 1)
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    
    return max(0, icc), p_value


def calculate_reviewer_scores(merged_df: pd.DataFrame, rating_cols: list) -> pd.DataFrame:
    """
    Calculate reviewer quality scores based on inter-rater reliability metrics.
    Now includes review count tracking and more nuanced bias assessment.
    
    Args:
        merged_df: DataFrame with review data
        rating_cols: List of column names containing ratings
        
    Returns:
        DataFrame with reviewer scores and metrics
    """
    reviewer_scores = []
    
    # Group by the item being reviewed (each group presentation)
    for group_reviewed, group_data in merged_df.groupby('Group#_reviewing'):
        if len(group_data) < 2:  # Need at least 2 reviewers
            continue
            
        # Create matrix: reviewers × rating dimensions
        reviewer_ids = group_data['email_clean'].tolist()
        rating_matrix = group_data[rating_cols].values
        review_times = group_data['DurationMin'].values
        
        # Calculate Cronbach's alpha for this group's reviews
        alpha = calculate_cronbach_alpha(pd.DataFrame(rating_matrix))
        
        # Calculate ICC for inter-rater reliability
        # Transpose so subjects are rows, raters are columns
        if len(rating_cols) >= 2:
            icc_data = pd.DataFrame(rating_matrix.T, columns=reviewer_ids)
            icc_value, icc_p = calculate_icc(icc_data)
        else:
            icc_value, icc_p = np.nan, np.nan
        
        # Calculate individual reviewer deviation from group mean
        group_means = np.mean(rating_matrix, axis=0)
        overall_group_mean = np.mean(group_means)
        
        for i, reviewer_id in enumerate(reviewer_ids):
            reviewer_ratings = rating_matrix[i, :]
            review_time = review_times[i]
            
            # Calculate reviewer-specific metrics
            deviation_from_mean = np.mean(np.abs(reviewer_ratings - group_means))
            consistency_score = 1 / (1 + deviation_from_mean)  # Higher is better
            
            # Apply time penalty only for very short reviews (but don't penalize reasonable variation)
            if review_time < 1.0:  # Per review, not total
                consistency_score *= 0.7  # Reduce consistency score for rushed individual reviews
            
            # Severity bias (tendency to rate higher or lower than group)
            severity_bias = np.mean(reviewer_ratings) - overall_group_mean
            
            # Halo effect (variance in ratings - lower variance suggests halo effect)
            halo_score = np.var(reviewer_ratings) if np.var(reviewer_ratings) > 0 else 0
            
            reviewer_scores.append({
                'reviewer_email': reviewer_id,
                'group_reviewed': group_reviewed,
                'cronbach_alpha': alpha,
                'icc_value': icc_value,
                'icc_p_value': icc_p,
                'consistency_score': consistency_score,
                'severity_bias': severity_bias,
                'halo_score': halo_score,
                'deviation_from_mean': deviation_from_mean,
                'review_time': review_time
            })
    
    scores_df = pd.DataFrame(reviewer_scores)
    
    if not scores_df.empty:
        # Calculate overall reviewer quality scores
        reviewer_summary = scores_df.groupby('reviewer_email').agg({
            'consistency_score': 'mean',
            'severity_bias': lambda x: np.abs(x).mean(),  # Average absolute bias
            'halo_score': 'mean',
            'deviation_from_mean': 'mean',
            'cronbach_alpha': 'mean',
            'icc_value': 'mean',
            'review_time': 'sum'  # Total time across all reviews
        }).reset_index()
        
        # Add review count
        review_counts = scores_df.groupby('reviewer_email').size().reset_index(name='review_count')
        reviewer_summary = reviewer_summary.merge(review_counts, on='reviewer_email')
        
        # Calculate z-scores for severity bias to identify truly extreme outliers
        if len(reviewer_summary) > 1:
            severity_values = scores_df.groupby('reviewer_email')['severity_bias'].mean()
            severity_mean = severity_values.mean()
            severity_std = severity_values.std()
            if severity_std > 0:
                reviewer_summary['severity_bias_raw'] = severity_values.values
                reviewer_summary['severity_bias_zscore'] = (severity_values - severity_mean) / severity_std
            else:
                reviewer_summary['severity_bias_raw'] = severity_values.values
                reviewer_summary['severity_bias_zscore'] = 0
        else:
            reviewer_summary['severity_bias_raw'] = reviewer_summary['severity_bias']
            reviewer_summary['severity_bias_zscore'] = 0
        
        # Composite reliability score (0-100 scale)
        # Weight: 40% consistency, 30% low bias, 20% appropriate variance, 10% ICC
        # Apply penalty only for incomplete reviews or extreme statistical outliers
        incomplete_penalty = reviewer_summary['review_count'].apply(lambda x: 0.5 if x < 2 else 1.0)
        extreme_bias_penalty = reviewer_summary['severity_bias_zscore'].apply(
            lambda x: 0.8 if abs(x) > 2.0 else 1.0  # Only penalize extreme outliers (>2 SD)
        )
        total_time_penalty = reviewer_summary['review_time'].apply(
            lambda x: 0.7 if x < 2.0 else 1.0  # Penalty for insufficient total time for 2 reviews
        )
        
        reviewer_summary['reliability_score'] = (
            reviewer_summary['consistency_score'] * 40 +
            (1 / (1 + reviewer_summary['severity_bias'])) * 30 +
            np.clip(reviewer_summary['halo_score'] * 10, 0, 20) +  # Cap halo contribution
            reviewer_summary['icc_value'].fillna(0.5) * 10
        ) * incomplete_penalty * extreme_bias_penalty * total_time_penalty
        
        return reviewer_summary
    else:
        return pd.DataFrame()

# ──────────────────────────────── Helpers ──────────────────────────────── #

def clean(series: pd.Series, pad: int | None = None) -> pd.Series:
    """Strip, lowercase; optionally zero‑pad numeric IDs."""
    s = series.astype(str).str.strip().str.lower()
    if pad:
        s = s.str.zfill(pad)
    return s


def read_review(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["email_clean"] = clean(df["Email"])
    df["student_id_clean"] = clean(df["StudentID"])
    return df


def read_roster(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["email_clean"] = clean(df["Email"])
    df["student_id_clean"] = clean(df["OrgDefinedId"])
    # pull last two chars as group #
    df["Group"] = df["Product Review Groups"].astype(str).str[-2:]
    return df


def build_enhanced_report(review: pd.DataFrame, roster: pd.DataFrame) -> str:
    """Return enhanced HTML report with reliability metrics."""
    # ── merge on email ───────────────────────────────────────────────
    merged = review.merge(
        roster[["email_clean", "student_id_clean", "Group"]],
        on="email_clean",
        how="left",
        indicator=True,
    )

    # ── submission pattern flags ────────────────────────────────────
    counts = merged.groupby("email_clean")["Group#_reviewing"].count()
    one_only = counts[counts == 1].index.tolist()
    three_plus = counts[counts > 2].index.tolist()
    submitted = set(merged["email_clean"])
    no_submit = sorted(set(roster["email_clean"]) - submitted)

    # map email → ID for nicer lists
    email_to_id = roster.set_index("email_clean")["student_id_clean"].to_dict()
    tag = lambda lst: [f"{email_to_id.get(e,'?')} ({e})" for e in lst]

    # ── duration (minutes) ──────────────────────────────────────────
    merged["DurationMin"] = (
        pd.to_datetime(merged["Completion time"]) -
        pd.to_datetime(merged["Start time"])
    ).dt.total_seconds() / 60

    # ── rating columns: numeric only, exclude helper cols ───────────
    rating_cols = [
        "Video_Quality", "Presenters", "Explanation",
        "Mechanism", "Side_Effects", "Bias",
        "Critical_review", "Study_Quality", "Study_participants",
    ]
    merged["OverallPct"] = merged[rating_cols].mean(axis=1) * 10  # 0‑100 %

    # ── ENHANCED: Calculate reliability metrics ─────────────────────
    LOG.info("Calculating inter-rater reliability metrics...")
    reviewer_scores = calculate_reviewer_scores(merged, rating_cols)
    
    # ── group stats: mean % and SD (%), plus count ──────────────────
    group_stats = (
        merged.groupby("Group#_reviewing")["OverallPct"]
        .agg(Mean="mean", SD="std", Reviews="count")
        .round({"Mean": 1, "SD": 1})
        .reset_index(names="Group")
        .sort_values("Group")
    )

    # Add reliability metrics to group stats if available
    if not reviewer_scores.empty:
        group_reliability = []
        for group_id in group_stats['Group']:
            group_reviews = merged[merged['Group#_reviewing'] == group_id]
            if len(group_reviews) >= 2:
                group_matrix = group_reviews[rating_cols].values
                alpha = calculate_cronbach_alpha(pd.DataFrame(group_matrix))
                
                # ICC calculation for group
                if len(rating_cols) >= 2:
                    icc_data = pd.DataFrame(group_matrix.T)
                    icc_val, _ = calculate_icc(icc_data)
                else:
                    icc_val = np.nan
                    
                group_reliability.append({
                    'Group': group_id,
                    'Cronbach_Alpha': round(alpha, 3) if not np.isnan(alpha) else 'N/A',
                    'ICC': round(icc_val, 3) if not np.isnan(icc_val) else 'N/A'
                })
            else:
                group_reliability.append({
                    'Group': group_id,
                    'Cronbach_Alpha': 'N/A',
                    'ICC': 'N/A'
                })
        
        reliability_df = pd.DataFrame(group_reliability)
        group_stats = group_stats.merge(reliability_df, on='Group', how='left')

    html_group = group_stats.to_html(
        index=False, classes="table", border=0,
        justify="left", float_format="%.1f"
    )

    # ── Enhanced time table with reliability scores ─────────────────
    time_data = merged[["email_clean", "Group#_reviewing", "DurationMin"]].copy()
    
    # Add reviewer reliability scores if available
    if not reviewer_scores.empty:
        time_data = time_data.merge(
            reviewer_scores[['reviewer_email', 'reliability_score']],
            left_on='email_clean',
            right_on='reviewer_email',
            how='left'
        )
        time_data['reliability_score'] = time_data['reliability_score'].round(1)
    else:
        time_data['reliability_score'] = 'N/A'

    html_time = (
        time_data[["email_clean", "Group#_reviewing", "DurationMin", "reliability_score"]]
        .sort_values("email_clean")
        .rename(columns={
            "email_clean": "Email",
            "Group#_reviewing": "Group",
            "DurationMin": "Minutes",
            "reliability_score": "Reliability Score"
        })
        .to_html(index=False, classes="table", border=0,
                 justify="left", float_format="%.1f")
    )

    # ── Generate individual student feedback sections ───────────────
    def reliability_score_to_grade(score, total_time_minutes, num_reviews, severity_bias_zscore):
        """Convert reliability score (0-100) to grade out of 10, adjusted for time and review count."""
        base_grade = 7.0  # Default grade
        
        if pd.isna(score) or score == 'N/A':
            base_grade = 7.0
        elif score >= 85:
            base_grade = 10.0
        elif score >= 75:
            base_grade = 9.0
        elif score >= 65:
            base_grade = 8.0
        elif score >= 55:
            base_grade = 7.0
        elif score >= 45:
            base_grade = 6.0
        elif score >= 35:
            base_grade = 5.0
        else:
            base_grade = 4.0
        
        # Apply penalty for incomplete reviews (less than 2 required)
        if num_reviews < 2:
            base_grade = base_grade * 0.5  # Half marks for incomplete reviews
        
        # Apply time penalty if total time for 2 reviews is under 2 minutes
        elif total_time_minutes < 2.0:
            base_grade = max(3.0, base_grade - 3.0)  # Reduce by 3 points, minimum 3/10
        
        # Apply penalty only for extreme bias (>2 standard deviations)
        if not pd.isna(severity_bias_zscore) and abs(severity_bias_zscore) > 2.0:
            base_grade = max(4.0, base_grade - 2.0)  # Reduce by 2 points for extreme bias
        
        return base_grade

    def get_feedback_message(score, grade, total_time_minutes, num_reviews, severity_bias, severity_bias_zscore):
        """Generate personalized feedback based on reliability score and review completion."""
        
        # Review completion issues
        completion_warning = ""
        if num_reviews < 2:
            completion_warning = f" **Important:** You completed only {num_reviews} review(s) instead of the required 2. This has resulted in half marks being awarded."
        elif total_time_minutes < 2.0:
            completion_warning = f" **Note:** Your total time for both reviews was {total_time_minutes:.1f} minutes, which suggests insufficient consideration. This has reduced your grade."
        
        # Bias explanation (informational, not necessarily penalized)
        bias_explanation = ""
        if not pd.isna(severity_bias) and abs(severity_bias) > 0.5:
            if severity_bias > 0.5:
                if not pd.isna(severity_bias_zscore) and severity_bias_zscore > 2.0:
                    bias_explanation = f" Your reviews rate significantly higher than peers (z-score: {severity_bias_zscore:.1f}), which may indicate leniency bias. This extreme bias has affected your grade."
                else:
                    bias_explanation = f" Your reviews tend to rate higher than peers (avg difference: +{severity_bias:.1f}), which may indicate leniency bias toward groups you know well. This is noted but not penalized."
            else:
                if not pd.isna(severity_bias_zscore) and severity_bias_zscore < -2.0:
                    bias_explanation = f" Your reviews rate significantly lower than peers (z-score: {severity_bias_zscore:.1f}), which may indicate very high standards or severity bias. This extreme bias has affected your grade."
                else:
                    bias_explanation = f" Your reviews tend to rate lower than peers (avg difference: {severity_bias:.1f}), which may indicate high standards or honest assessment. This is noted but not penalized."
        
        # Base reliability message
        base_message = ""
        if pd.isna(score) or score == 'N/A':
            base_message = "Your peer review completion has been recorded. Insufficient data for detailed reliability analysis."
        elif score >= 85:
            base_message = "Excellent work! Your reviews show high reliability and consistency with peer consensus. You demonstrate strong critical evaluation skills."
        elif score >= 75:
            base_message = "Good work! Your reviews are generally reliable and consistent. You show good understanding of the evaluation criteria."
        elif score >= 65:
            base_message = "Satisfactory work. Your reviews show reasonable consistency, with some room for improvement in alignment with peer consensus."
        elif score >= 55:
            base_message = "Your reviews show moderate consistency. Consider reviewing the evaluation criteria more carefully to improve alignment with peers."
        elif score >= 45:
            base_message = "Your reviews show some inconsistency with peer consensus." + bias_explanation + " This may reflect genuine differences in assessment standards."
        else:
            base_message = "Your reviews show significant inconsistency with peer consensus." + bias_explanation + " This may indicate very different assessment standards or misunderstanding of the criteria."
        
        # Appeal process
        appeal_message = ""
        if score < 65 or total_time_minutes < 2.0 or num_reviews < 2 or (not pd.isna(severity_bias_zscore) and abs(severity_bias_zscore) > 2.0):
            appeal_message = " If you believe this assessment does not reflect the quality of your reviews or if there are extenuating circumstances, please contact the instructor to discuss your evaluation."
        
        return base_message + completion_warning + appeal_message

    student_feedback_sections = []
    if not reviewer_scores.empty:
        # Calculate total time per reviewer and count reviews
        reviewer_stats = merged.groupby('email_clean').agg({
            'DurationMin': 'sum',
            'Group#_reviewing': 'count'
        }).reset_index()
        reviewer_stats.columns = ['email_clean', 'total_time_minutes', 'num_reviews']
        
        # Calculate z-scores for severity bias to identify extreme outliers
        if not reviewer_scores.empty and 'severity_bias' in reviewer_scores.columns:
            severity_mean = reviewer_scores['severity_bias'].mean()
            severity_std = reviewer_scores['severity_bias'].std()
            if severity_std > 0:
                reviewer_scores['severity_bias_zscore'] = (reviewer_scores['severity_bias'] - severity_mean) / severity_std
            else:
                reviewer_scores['severity_bias_zscore'] = 0
        
        # Merge with roster to get student IDs and names
        feedback_data = reviewer_scores.merge(
            roster[['email_clean', 'student_id_clean']],
            left_on='reviewer_email',
            right_on='email_clean',
            how='left'
        )
        
        # Add reviewer statistics
        feedback_data = feedback_data.merge(
            reviewer_stats,
            left_on='reviewer_email',
            right_on='email_clean',
            how='left'
        )
        
        # Get student names if available in roster
        student_names = {}
        name_columns = [col for col in roster.columns if 'name' in col.lower()]
        if len(name_columns) >= 2:  # FirstName and LastName
            first_name_col = next((col for col in name_columns if 'first' in col.lower()), name_columns[0])
            last_name_col = next((col for col in name_columns if 'last' in col.lower()), name_columns[1])
            for _, row in roster.iterrows():
                full_name = f"{row[first_name_col]} {row[last_name_col]}"
                student_names[row['email_clean']] = full_name
        elif len(name_columns) == 1:  # Single Name column
            for _, row in roster.iterrows():
                student_names[row['email_clean']] = row[name_columns[0]]
        
        for _, row in feedback_data.iterrows():
            score = row['reliability_score']
            total_time = row.get('total_time_minutes', 0)
            num_reviews = row.get('num_reviews', 0)
            severity_bias = row.get('severity_bias', 0)
            severity_bias_zscore = row.get('severity_bias_zscore', 0)
            
            grade = reliability_score_to_grade(score, total_time, num_reviews, severity_bias_zscore)
            feedback = get_feedback_message(score, grade, total_time, num_reviews, severity_bias, severity_bias_zscore)
            student_id = row.get('student_id_clean', row['reviewer_email'])
            student_name = student_names.get(row['reviewer_email'], 'Name not available')
            
            # Determine special circumstances
            special_notes = []
            if num_reviews < 2:
                special_notes.append(f"<strong>Incomplete:</strong> Only {num_reviews} review(s) completed (required: 2)")
            elif total_time < 2.0:
                special_notes.append(f"<strong>Time Warning:</strong> Total time for 2 reviews: {total_time:.1f} minutes")
            
            if not pd.isna(severity_bias_zscore) and abs(severity_bias_zscore) > 2.0:
                special_notes.append(f"<strong>Extreme Bias:</strong> Rating pattern {severity_bias_zscore:.1f} standard deviations from peer average")
            elif not pd.isna(severity_bias) and abs(severity_bias) > 0.5:
                bias_type = "higher" if severity_bias > 0 else "lower"
                special_notes.append(f"<strong>Note:</strong> Tends to rate {bias_type} than peers (not penalized)")
            
            special_notes_html = ""
            if special_notes:
                special_notes_html = "<p>" + "<br>".join(special_notes) + "</p>"
            
            feedback_section = f"""
    <!-- STUDENT FEEDBACK: {student_id} -->
    <div class="student-feedback" style="border: 2px dashed #007acc; margin: 20px 0; padding: 15px; background-color: #f8f9fa;">
        <h3>Student: {student_name} ({student_id})</h3>
        <p><strong>Peer Review Quality Grade: {grade}/10</strong></p>
        <p><strong>Reliability Score: {score:.1f}/100</strong></p>
        <p><strong>Reviews Completed: {num_reviews}/2</strong></p>
        <p><strong>Total Time for Both Reviews: {total_time:.1f} minutes</strong></p>
        {special_notes_html}
        <p><strong>Feedback:</strong> {feedback}</p>
        <hr style="border: 1px solid #007acc;">
        <p><em>Copy the section above (including grade and feedback) to paste into D2L for this student.</em></p>
    </div>
    """
            student_feedback_sections.append(feedback_section)
    
    html_student_feedback = '\n'.join(student_feedback_sections)

    # ── Reviewer quality summary table ──────────────────────────────
    html_reviewer_quality = ""
    if not reviewer_scores.empty:
        quality_summary = reviewer_scores[[
            'reviewer_email', 'reliability_score', 'consistency_score',
            'severity_bias', 'cronbach_alpha', 'icc_value'
        ]].copy()
        
        # Add student IDs
        quality_summary = quality_summary.merge(
            roster[['email_clean', 'student_id_clean']],
            left_on='reviewer_email',
            right_on='email_clean',
            how='left'
        )
        
        quality_display = quality_summary[[
            'student_id_clean', 'reliability_score', 'consistency_score',
            'severity_bias', 'cronbach_alpha', 'icc_value'
        ]].round(3)
        
        quality_display = quality_display.rename(columns={
            'student_id_clean': 'Student ID',
            'reliability_score': 'Overall Score',
            'consistency_score': 'Consistency',
            'severity_bias': 'Severity Bias',
            'cronbach_alpha': 'Cronbach α',
            'icc_value': 'ICC'
        })
        
        html_reviewer_quality = quality_display.to_html(
            index=False, classes="table", border=0,
            justify="left", float_format="%.3f"
        )

    # ── Enhanced Jinja2 template ────────────────────────────────────
    tmpl = Template(
        """
        <html><head><title>Enhanced Peer‑Evaluation Report</title>
        <style>
          body{font-family:sans-serif;max-width:1000px;margin:0 auto;padding:20px}
          .table{border-collapse:collapse;width:100%;margin-bottom:20px}
          .table th,.table td{border:1px solid #ddd;padding:8px;text-align:left}
          .table th{background:#f2f2f2;font-weight:bold}
          .metric-description{background:#f9f9f9;padding:15px;margin:10px 0;border-left:4px solid #007acc}
          .section{margin-bottom:30px}
          .reliability-high{background-color:#d4edda}
          .reliability-medium{background-color:#fff3cd}
          .reliability-low{background-color:#f8d7da}
          .student-feedback{border:2px dashed #007acc;margin:20px 0;padding:15px;background-color:#f8f9fa}
          h1{color:#333;border-bottom:2px solid #007acc;padding-bottom:10px}
          h2{color:#555;margin-top:30px}
          .feedback-instructions{background:#e7f3ff;padding:15px;margin:20px 0;border:1px solid #007acc;border-radius:5px}
        </style></head><body>
        
        <h1>Enhanced Instructor Peer-Evaluation Report</h1>
        
        <div class="metric-description">
            <strong>Reliability Metrics Explanation:</strong><br>
            • <strong>Cronbach's α</strong>: Internal consistency (>0.7 good, >0.8 excellent)<br>
            • <strong>ICC</strong>: Inter-rater reliability (>0.75 good, >0.9 excellent)<br>
            • <strong>Reliability Score</strong>: Composite score (0-100, higher = more reliable reviewer)<br>
            • <strong>Consistency</strong>: Agreement with group consensus (higher = better)<br>
            • <strong>Severity Bias</strong>: Tendency to rate higher/lower than peers (closer to 0 = better)
        </div>

        <div class="section">
            <h2>Group Results with Reliability Metrics</h2>
            {{ group_html | safe }}
        </div>

        {% if reviewer_quality_html %}
        <div class="section">
            <h2>Reviewer Quality Assessment</h2>
            <p><em>Ranked by overall reliability score. Higher scores indicate more reliable reviewers.</em></p>
            {{ reviewer_quality_html | safe }}
        </div>
        {% endif %}

        {% if one_only %}
        <div class="section">
            <h2>Exactly One Submission ({{ one_only|length }})</h2>
            <p>{{ one_only|join(", ") }}</p>
        </div>
        {% endif %}

        {% if three_plus %}
        <div class="section">
            <h2>Three+ Submissions ({{ three_plus|length }})</h2>
            <p>{{ three_plus|join(", ") }}</p>
        </div>
        {% endif %}

        {% if no_submit %}
        <div class="section">
            <h2>No Submission Received ({{ no_submit|length }})</h2>
            <p>{{ no_submit|join(", ") }}</p>
        </div>
        {% endif %}

        <div class="section">
            <h2>Time on Task and Reliability Scores</h2>
            <p><em>Each review duration in minutes with corresponding reliability assessment.</em></p>
            {{ time_html | safe }}
        </div>

        {% if student_feedback_html %}
        <div class="section">
            <h2>Individual Student Feedback (Copy to D2L)</h2>
            <div class="feedback-instructions">
                <strong>Instructions:</strong> Each dashed box below contains feedback for one student. 
                Copy the content (including grade and feedback text) and paste directly into the student's 
                feedback section in D2L. The grades are out of 10 based on peer review quality.
            </div>
            {{ student_feedback_html | safe }}
        </div>
        {% endif %}

        <div class="metric-description">
            <strong>Interpretation Guidelines:</strong><br>
            • Groups with Cronbach's α < 0.6 may have inconsistent rating criteria<br>
            • ICC < 0.5 suggests poor inter-rater agreement<br>
            • High severity bias may indicate need for calibration training<br>
            • Very low variance (halo score) may suggest insufficient differentiation<br><br>
            <strong>Grading Scale (out of 10):</strong><br>
            • 10: Reliability Score 85+ (Excellent)<br>
            • 9: Reliability Score 75-84 (Very Good)<br>
            • 8: Reliability Score 65-74 (Good)<br>
            • 7: Reliability Score 55-64 (Satisfactory)<br>
            • 6: Reliability Score 45-54 (Needs Improvement)<br>
            • 5: Reliability Score 35-44 (Poor)<br>
            • 4: Reliability Score <35 (Very Poor)<br><br>
            <strong>Penalties Applied:</strong><br>
            • Incomplete reviews (<2 required): Half marks<br>
            • Insufficient time (<2 minutes total for both reviews): -3 points<br>
            • Extreme bias (>2 standard deviations from peer average): -2 points<br>
            • Minor bias variations are noted but not penalized<br><br>
            <strong>Time Measurement:</strong> Total time refers to combined time for both required peer reviews.
        </div>

        </body></html>
        """
    )

    return tmpl.render(
        group_html=html_group,
        time_html=html_time,
        reviewer_quality_html=html_reviewer_quality,
        student_feedback_html=html_student_feedback,
        one_only=tag(one_only),
        three_plus=tag(three_plus),
        no_submit=tag(no_submit),
    )


# ──────────────────────────────── Main ──────────────────────────────── #

def cli() -> None:
    ap = argparse.ArgumentParser(
        description="Generate enhanced peer‑evaluation instructor report with reliability metrics."
    )
    ap.add_argument("reviews", type=Path, help="Excel file with peer reviews")
    ap.add_argument("roster", type=Path, help="Excel class‑list file")
    ap.add_argument("-o", "--out", type=Path, default="enhanced_peer_report.html",
                    help="HTML output path")
    args = ap.parse_args()

    LOG.info("Loading data …")
    review_df = read_review(args.reviews)
    roster_df = read_roster(args.roster)

    LOG.info("Building enhanced report with reliability metrics …")
    html = build_enhanced_report(review_df, roster_df)

    args.out.write_text(html, encoding="utf‑8")
    LOG.info("Enhanced report saved to %s", args.out)


if __name__ == "__main__":
    cli()
