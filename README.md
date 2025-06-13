## Enhanced Peer Review Analysis System

Overview
This comprehensive peer review analysis system provides advanced statistical reliability assessment for student peer evaluations, including inter-rater reliability metrics, individual reviewer quality scoring, and automated feedback generation.

### How I Use This System

### The Assignment
Students work in groups to create YouTube videos reviewing products that make scientific claims. They investigate whether the science actually supports these claims. Each student then peer reviews 2 other groups' videos using a Microsoft Form.
My Workflow

- Download class list from D2L with group assignments
- Use Excel to randomly assign students to review 2 groups (not their own)
- Send mail-merge emails with review assignments and Form link
- Download completed reviews from Microsoft Forms
- Run this analysis system to generate instructor reports and student feedback

## Why This Works

- Teaches Scientific Literacy: Students learn to critically evaluate real-world scientific claims
- Develops Peer Review Skills: Statistical feedback helps students become better, less biased reviewers
- Integrates Course Content: Connects peer review experience to scientific process concepts we cover in class
- Ensures Fair Grading: Uses reliability measures (Cronbach's alpha, ICC) to weight peer evaluations appropriately
## System Architecture

### Core Components

#### Primary Analysis Tools

- **`enhanced_instructor_report.py`** - Main analysis engine with reliability calculations
- **`enhanced_gui_launcher.py`** - User-friendly GUI interface for instructor reports (wraps enhanced_instructor_report in GUI for selecting files and folders)
- **`enhanced_group_reports.py`** - Individual group report generator with reliability metrics
- **`reliability_analyzer.py`** - Standalone comprehensive statistical analysis tool

#### Supporting Files

- **`instructor_report.py`** - Original clean instructor report (legacy support)
- **`gui_launcher.py`** - Original GUI launcher (legacy support)
- **`Peer_Review_Quality_Rubric.md`** - Assessment rubric aligned with reliability scoring
- **`Quick_Reference.md`** - Quick start guide for daily use

---

## Required File Structure

### Peer Review Data File (Excel .xlsx)

**Required Columns:**

```
Email                    - Student email address (matching roster)
StudentID               - Student identification number
Group#_reviewing        - Group number being reviewed
Start time              - Review start timestamp
Completion time         - Review completion timestamp
Video_Quality          - Rating 1-10
Presenters             - Rating 1-10  
Explanation            - Rating 1-10
Mechanism              - Rating 1-10
Side_Effects           - Rating 1-10
Bias                   - Rating 1-10
Critical_review        - Rating 1-10
Study_Quality          - Rating 1-10
Study_participants     - Rating 1-10
```

**Optional Feedback Columns:**

```
Comments_ProductionQuality  - Text feedback on presentation quality
Comment_Information        - Text feedback on content quality  
Comment_Research          - Text feedback on research quality
```

### Student Roster File (Excel .xlsx)

**Required Columns:**

```
Email                    - Student email (must match peer review file)
OrgDefinedId            - Student ID from LMS
Product Review Groups   - Group assignment (format: "Group 01", "Group 02", etc.)
```

**Optional Name Columns:**

```
FirstName               - Student first name
LastName                - Student last name
Name                    - Full name (alternative to FirstName/LastName)
```

---

## Customization Guide

### Modifying Rating Columns

**Location:** Multiple files need updating for rating column changes

#### 1. Enhanced Instructor Report (`enhanced_instructor_report.py`)

**Line ~252:** Update the `rating_cols` list

```python
rating_cols = [
    "Video_Quality", "Presenters", "Explanation",
    "Mechanism", "Side_Effects", "Bias", 
    "Critical_review", "Study_Quality", "Study_participants",
]
```

#### 2. Enhanced Group Reports (`enhanced_group_reports.py`)

**Line ~64:** Update the `rating_cols` list

```python
rating_cols = [
    'Video_Quality', 'Presenters', 'Explanation', 'Mechanism', 
    'Side_Effects', 'Bias', 'Critical_review', 'Study_Quality', 
    'Study_participants'
]
```

#### 3. Reliability Analyzer (`reliability_analyzer.py`)

**Line ~320:** Update the `rating_cols` list in the `analyze_peer_review_file` function

```python
rating_cols = [
    "Video_Quality", "Presenters", "Explanation",
    "Mechanism", "Side_Effects", "Bias",
    "Critical_review", "Study_Quality", "Study_participants",
]
```

### Modifying Grading Scale

**Location:** `enhanced_instructor_report.py`
**Function:** `reliability_score_to_grade()` (Line ~338)

**Current Scale:**

```python
elif score >= 85: base_grade = 10.0
elif score >= 75: base_grade = 9.0
elif score >= 65: base_grade = 8.0
elif score >= 55: base_grade = 7.0
elif score >= 45: base_grade = 6.0
elif score >= 35: base_grade = 5.0
else: base_grade = 4.0
```

### Modifying Time Thresholds

#### Total Time Penalty (for both reviews combined)

**Location:** `enhanced_instructor_report.py`
**Line ~358:** Change the 2.0 minute threshold

```python
elif total_time_minutes < 2.0:  # Change this value
    base_grade = max(3.0, base_grade - 3.0)
```

#### Individual Review Time Penalty

**Location:** `enhanced_instructor_report.py`
**Line ~189:** Change the 1.0 minute threshold

```python
if review_time < 1.0:  # Change this value
    consistency_score *= 0.7
```

### Modifying Bias Detection Thresholds

#### Statistical Outlier Detection

**Location:** `enhanced_instructor_report.py`
**Line ~361:** Change the 2.0 standard deviation threshold

```python
if not pd.isna(severity_bias_zscore) and abs(severity_bias_zscore) > 2.0:  # Change this value
    base_grade = max(4.0, base_grade - 2.0)
```

#### Minor Bias Reporting Threshold

**Location:** `enhanced_instructor_report.py`
**Line ~392:** Change the 0.5 threshold for bias reporting

```python
if not pd.isna(severity_bias) and abs(severity_bias) > 0.5:  # Change this value
```

### Modifying Column Names

#### Peer Review File Column Mapping

**Location:** `enhanced_instructor_report.py`
**Function:** `read_review()` (Line ~213)

```python
def read_review(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["email_clean"] = clean(df["Email"])        # Change "Email" column name here
    df["student_id_clean"] = clean(df["StudentID"])  # Change "StudentID" column name here
    return df
```

#### Student Roster File Column Mapping

**Location:** `enhanced_instructor_report.py` 
**Function:** `read_roster()` (Line ~220)

```python
def read_roster(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["email_clean"] = clean(df["Email"])        # Change "Email" column name here
    df["student_id_clean"] = clean(df["OrgDefinedId"])  # Change "OrgDefinedId" column name here
    df["Group"] = df["Product Review Groups"].astype(str).str[-2:]  # Change group column name here
    return df
```

### Modifying Feedback Column Names

**Location:** `enhanced_group_reports.py`
**Lines 144-157:** Update feedback column names

```python
if 'Comments_ProductionQuality' in data.columns:  # Change column name here
    feedback_data['Comments_ProductionQuality'] = [
        str(comment) for comment in data['Comments_ProductionQuality'].dropna() 
        if str(comment) != 'nan' and str(comment).strip()
    ]
if 'Comment_Information' in data.columns:  # Change column name here
    # ... similar pattern
if 'Comment_Research' in data.columns:  # Change column name here
    # ... similar pattern
```

### Modifying Group Assignment Format

**Current Format:** Groups extracted from "Product Review Groups" column using last 2 characters
**Location:** `enhanced_instructor_report.py`, Line ~225

```python
df["Group"] = df["Product Review Groups"].astype(str).str[-2:]
```

**Alternative Formats:**

```python
# For "Group-01", "Group-02" format:
df["Group"] = df["Product Review Groups"].str.split('-').str[1]

# For direct group numbers:
df["Group"] = df["Product Review Groups"]

# For "Team 1", "Team 2" format:
df["Group"] = df["Product Review Groups"].str.split().str[1]
```

---

## Configuration Templates

### Standard Configuration

```python
# Rating columns (1-10 scale)
RATING_COLUMNS = [
    "Video_Quality", "Presenters", "Explanation",
    "Mechanism", "Side_Effects", "Bias",
    "Critical_review", "Study_Quality", "Study_participants"
]

# Time thresholds (minutes)
TOTAL_TIME_THRESHOLD = 2.0      # Combined time for both reviews
INDIVIDUAL_TIME_THRESHOLD = 1.0  # Per individual review

# Bias detection
EXTREME_BIAS_THRESHOLD = 2.0     # Standard deviations for penalty
MINOR_BIAS_THRESHOLD = 0.5       # Threshold for reporting (no penalty)

# Grading scale (reliability score -> grade out of 10)
GRADE_THRESHOLDS = {
    85: 10.0,  # Excellent
    75: 9.0,   # Very Good
    65: 8.0,   # Good
    55: 7.0,   # Satisfactory
    45: 6.0,   # Needs Improvement
    35: 5.0,   # Poor
}
DEFAULT_GRADE = 4.0  # Very Poor
```

### Alternative Subject Configuration (Example)

```python
# For different subject areas, modify rating columns:
RATING_COLUMNS_LITERATURE = [
    "Thesis_Clarity", "Evidence_Quality", "Analysis_Depth",
    "Writing_Style", "Citations", "Originality"
]

RATING_COLUMNS_SCIENCE = [
    "Hypothesis", "Methodology", "Data_Analysis", 
    "Results_Presentation", "Discussion", "Conclusions"
]
```

---

## Advanced Customization

### Adding New Reliability Metrics

**Location:** `enhanced_instructor_report.py`
**Function:** `calculate_reviewer_scores()` (Line ~118)

Add new metrics to the reviewer_scores dictionary:

```python
reviewer_scores.append({
    'reviewer_email': reviewer_id,
    'group_reviewed': group_reviewed,
    'cronbach_alpha': alpha,
    'icc_value': icc_value,
    # Add new metrics here:
    'new_metric': your_calculation,
})
```

### Modifying HTML Templates

#### Instructor Report Template

**Location:** `enhanced_instructor_report.py`
**Lines ~483-580:** HTML template definition

#### Group Report Template

**Location:** `enhanced_group_reports.py`
**Lines ~165-315:** HTML content generation

### Adding New File Format Support

**Current:** Excel (.xlsx) only
**To Add CSV Support:** Modify `read_review()` and `read_roster()` functions:

```python
def read_review(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    # ... rest of function
```

---

## Testing Configurations

### Test Data Requirements

- Minimum 2 reviewers per group for reliability calculations
- At least 3 groups for meaningful statistics
- Complete timestamp data for time analysis
- Varied rating patterns to test bias detection

### Validation Checklist

- [ ] All rating columns present and numeric
- [ ] Email addresses match between files
- [ ] Group assignments properly formatted
- [ ] Timestamp columns parseable
- [ ] No completely missing reviews
- [ ] Student IDs consistent

---

## Troubleshooting Common Customizations

### Column Not Found Errors

1. Check exact column name spelling (case-sensitive)
2. Verify column exists in both sample and actual data
3. Update all references across multiple files

### Grade Scale Issues

1. Ensure thresholds are in descending order
2. Check that reliability scores are 0-100 scale
3. Verify grade output is appropriate scale (e.g., 0-10)

### Time Calculation Problems

1. Verify timestamp format consistency
2. Check timezone handling if applicable
3. Ensure Start time < Completion time

### Group Assignment Failures

1. Check group column format matches extraction method
2. Verify group numbers are consistent
3. Test with sample data first

---

## Support and Maintenance

For issues or enhancements:

1. Check configuration matches your data format exactly
2. Test with small sample data set first
3. Verify all required columns are present
4. Check that dependencies are properly installed

The system is designed to be flexible while maintaining statistical rigor. Most customizations require changes in 2-3 locations to maintain consistency across all tools.
