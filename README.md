## Enhanced Peer Review Analysis System

Overview
This comprehensive peer review analysis system provides advanced statistical reliability assessment for student peer evaluations, including inter-rater reliability metrics, individual reviewer quality scoring, and automated feedback generation.

## Start Here

Run the main instructor workflow with:

```bash
python src/gui_launcher.py
```

This is the primary entry point for the project. It asks you to point to the specific input files and folders, then generates the instructor HTML report plus optional consolidated grade exports for mail merge and D2L import.

### How I Use This System

### The Assignment
Students work in groups to create YouTube videos reviewing products that make scientific claims. They investigate whether the science actually supports these claims. Each student then peer reviews 2 other groups' videos using a Microsoft Form.

### My Workflow

- Download class list from D2L with group assignments
- Run the automated assigner (`python src/assign_reviews.py`) to randomly assign every student two non-self groups to review
- Send mail-merge emails with review assignments and Form link
- Download completed reviews from Microsoft Forms
- Run `python src/gui_launcher.py` to generate the instructor report, student feedback, and consolidated grade exports

---

## Step 1: Assign Peer Reviews with `assign_reviews.py`

This tool takes a simple 3-column file (Email, Group, URL) and assigns each student two review groups.

### Input: Prepare Your CSV File

You need **exactly three columns**:

| Column Keyword | Purpose | Acceptable Names | ❌ Won't Work |
|---|---|---|---|
| **email** | Student's email | `Email`, `E-mail`, `EmailAddress` | `Student ID`, `Name` |
| **group** | Student's group | `Group`, `Product Review Groups`, `GroupName` | `Team`, `Section` |
| **url** | Link to submission | `URL`, `SubmissionURL`, `VideoLink` | (any 3rd column) |

**Key Rule**: Column headers must **contain** these keywords (case-insensitive).
- ✅ `Product Review Groups` works (contains "group")
- ❌ `Team Name` fails (doesn't contain "group")

### How to Create the File

**Step 1: Get Student Emails from D2L**
1. In Brightspace: `Course Admin` → `Classlist`
2. Click `Export` and download the CSV
3. Open in Excel, keep only the email column
4. Rename header to **`Email`** or anything containing "email"

**Step 2: Get Group Assignments from D2L**
1. In Brightspace: `Course Admin` → `Groups` → `Manage Groups`
2. Export your submission category's groups
3. In Excel, add a new column with header **`Group`** or anything containing "group"
4. Fill in group names for each student (e.g., `Group 01`, `Group 02`)
   - All students in the same group must have the **same group name**

**Step 3: Add Submission URLs**
1. Add a third column with header **`URL`** or anything containing "url"
2. Add the submission link for each group:
   - **D2L Dropbox**: Right-click folder → copy link
   - **Google Drive/OneDrive**: Copy share link
   - **YouTube**: Paste video URL
3. **All students in the same group must have the same URL**

**Step 4: Save and Run**
1. Save as `.csv` file
2. Run: `python src/assign_reviews.py`
3. Select your file in the file picker
4. Output: New file with `-with-reviews.csv` suffix containing:
   - `ReviewGroup1` & `ReviewURL1` (first assigned review)
   - `ReviewGroup2` & `ReviewURL2` (second assigned review)

### Example Files

**Best Practice (Recommended):**
```csv
Email,Group,URL
alice@ucalgary.ca,Group 01,https://www.youtube.com/watch?v=abc123
bob@ucalgary.ca,Group 01,https://www.youtube.com/watch?v=abc123
charlie@ucalgary.ca,Group 02,https://youtu.be/def456
diana@ucalgary.ca,Group 02,https://youtu.be/def456
```

**Also Works (Alternative Names):**
```csv
E-mail,Product Review Groups,SubmissionURL
alice@ucalgary.ca,Group 01,https://www.youtube.com/watch?v=abc123
bob@ucalgary.ca,Group 01,https://www.youtube.com/watch?v=abc123
charlie@ucalgary.ca,Group 02,https://youtu.be/def456
diana@ucalgary.ca,Group 02,https://youtu.be/def456
```

**Does NOT Work:**
```csv
StudentEmail,Team,Link
alice@ucalgary.ca,Group 01,https://www.youtube.com/watch?v=abc123
```
❌ "Team" doesn't contain "group"

### Troubleshooting `assign_reviews.py`

| Error | Fix |
|---|---|
| "No email-like column found" | Rename column to `Email`, `E-mail`, or `EmailAddress` |
| "No group-like column found" | Rename column to `Group`, `Product Review Groups`, or `GroupName` |
| "Could not uniquely identify URL column" | Delete extra columns—you need exactly 3: Email, Group, URL |
| Different random assignments each time | You changed `RANDOM_SEED`. Keep the same value to repeat results |

> **Tip**: If you want different randomized assignments, change `RANDOM_SEED` at the top of `src/assign_reviews.py` and re-run.

---

## Step 2: Prepare Data for Instructor Analysis with `gui_launcher.py`

**NOTE**: This step is **optional**. Skip this section if you only need to assign reviews.

If you also want to generate the instructor analysis report, you'll need a roster file with additional information.

### Input: Prepare Your Roster for Analysis

You need a CSV with at least these fields (header names flexible):

| Data | Purpose |
|---|---|
| Email | Student's email (must match Microsoft Forms export) |
| OrgDefinedID | Student's institutional ID from D2L |
| Product Review Groups | The group the student belongs to |

### How to Create the Roster File

**Step 1: Export Classlist from D2L**
1. In Brightspace: `Course Admin` → `Classlist`
2. Click `Export`
3. Keep these columns (you can keep others, but at minimum need these):
   - `Email`
   - `Org Defined ID`
   - `First Name` (optional but helpful)
   - `Last Name` (optional but helpful)

**Step 2: Export Groups from D2L**
1. In Brightspace: `Course Admin` → `Groups` → `Manage Groups`
2. Export your submission category
3. Note which group each student belongs to

**Step 3: Merge Files**
1. In the Classlist file, add a new column: `Product Review Groups`
2. For each student, fill in their group name from the Groups export
3. Save as `.xlsx` or `.csv`

### Expected Roster Format

```csv
Email,OrgDefinedId,FirstName,LastName,Product Review Groups
alice@ucalgary.ca,30012345,Alice,Smith,Group 01
bob@ucalgary.ca,30012346,Bob,Jones,Group 01
charlie@ucalgary.ca,30012347,Charlie,Brown,Group 02
diana@ucalgary.ca,30012348,Diana,Davis,Group 02
```

### Important Notes

- Email values must match the Microsoft Forms review export exactly
- OrgDefinedId should be the D2L student ID, not a username
- Group names can be `Group 01`, `Group 1`, `Team A`, etc.—any consistent format works
- If Brightspace doesn't show `Export`, that's usually a permissions issue, not a project problem

## Why This Works

- Teaches Scientific Literacy: Students learn to critically evaluate real-world scientific claims
- Develops Peer Review Skills: Statistical feedback helps students become better, less biased reviewers
- Integrates Course Content: Connects peer review experience to scientific process concepts we cover in class
- Ensures Fair Grading: Uses reliability measures (Cronbach's alpha, ICC) to weight peer evaluations appropriately
## System Architecture

### Core Components

#### Primary Analysis Tools

- **`instructor_report.py`** - Main analysis engine with reliability calculations
- **`gui_launcher.py`** - Main GUI entry point for the instructor workflow; generates the HTML instructor report and optional grade exports
- **`group_peer_reports.py`** - Individual group report generator with reliability metrics
- **`reliability_analyzer.py`** - Standalone comprehensive statistical analysis tool
- **`assign_reviews.py`** - CSV-based reviewer assignment generator that balances two non-self review targets per student

#### Supporting Files

- **`instructor_report.py`** - Original clean instructor report (legacy support)
- **`gui_launcher.py`** - Original GUI launcher (legacy support)
- **`Peer_Review_Quality_Rubric.md`** - Assessment rubric aligned with reliability scoring
- **`Quick_Reference.md`** - Quick start guide for daily use

---

## Recommended Instructor Workflow

Use this order if you want both the instructor report and a final CSV/XLSX for mail merge or D2L.

1. Generate or collect your group-level outputs as usual:
   `Group_#Peer_Review_Report.html`, `feedback_group_#.html`, and the peer review export from Microsoft Forms.
2. Run:

```bash
python src/gui_launcher.py
```

3. In the GUI, select:
   - the peer-review file from Microsoft Forms
   - the D2L grading export / student roster file
   - the output path for `Instructor_Report.html`
4. If you want a consolidated grade export, also select either:
   - the folder containing `feedback_Group-#.html` files
   - the folder containing `Group_#Peer_Review_Report.html` files
   - or a spreadsheet containing written report grades if you are not using the HTML feedback files
5. Choose where to save the consolidated workbook.
6. The launcher can now create:
   - `Instructor_Report.html`
   - `Consolidated_Grades.xlsx`
   - `Consolidated_Grades.csv`
   - `Consolidated_Grades_D2L_Import.csv`

### Convert HTML Reports to PDF

If you want to upload the group report files back to D2L as feedback attachments, converting the HTML reports to PDF is usually the safest option.

#### Prerequisites

You need both of these installed on Windows:

- `pandoc`
- `wkhtmltopdf`

Check that both are available in PowerShell:

```powershell
pandoc --version
wkhtmltopdf --version
```

If either command is not recognized, install that tool first and reopen PowerShell before continuing.

#### Convert all group peer review reports to PDF

Open PowerShell in the folder that contains your `Group_#Peer_Review_Report.html` files and run:

```powershell
Get-ChildItem -Filter "Group_*Peer_Review_Report.html" | ForEach-Object {
    pandoc $_.FullName -o ($_.DirectoryName + "\" + $_.BaseName + ".pdf") --pdf-engine=wkhtmltopdf
}
```

This creates a PDF beside each HTML file, for example:

```text
Group_1Peer_Review_Report.html
Group_1Peer_Review_Report.pdf
```

#### Convert all written report feedback files to PDF

If you also want to convert the `feedback_Group-#.html` files, run:

```powershell
Get-ChildItem -Filter "feedback_Group-*.html" | ForEach-Object {
    pandoc $_.FullName -o ($_.DirectoryName + "\" + $_.BaseName + ".pdf") --pdf-engine=wkhtmltopdf
}
```

#### Convert both types at once

If both file types are in the same folder, you can convert them in one pass:

```powershell
Get-ChildItem -Path ".\*" -Include "Group_*Peer_Review_Report.html","feedback_Group-*.html" -File | ForEach-Object {
    pandoc $_.FullName -o ($_.DirectoryName + "\" + $_.BaseName + ".pdf") --pdf-engine=wkhtmltopdf
}
```

#### Practical Notes

- Run these commands from the folder that contains the HTML reports.
- The PDF files will be created in the same folder as the original HTML files.
- PDF is generally more reliable than HTML when uploading feedback files to D2L.
- If you plan to use D2L bulk feedback upload, convert first, then place the PDFs into the D2L feedback zip structure.

### Consolidated Grade Calculation

The consolidated export uses this breakdown:

| Component | Weight | Source |
| --- | ---: | --- |
| Peer review completion | 10% | Individual reviewer-quality grade from the instructor report |
| Peer scores and ranking | 50% | Group peer-review score based on how other students rated the video |
| Written report | 40% | Group written-report grade from your spreadsheet or `feedback_group_*.html` files |

### What Each Export Is For

- `Instructor_Report.html`
  Instructor-facing report with group reliability metrics and individual peer review completion grades.
- `Consolidated_Grades.xlsx`
  Best file for review, filtering, checking totals, and mail merge preparation.
- `Consolidated_Grades.csv`
  Same detailed data in CSV form for spreadsheet tools or mail merge workflows.
- `Consolidated_Grades_D2L_Import.csv`
  Minimal CSV for D2L grade import.

### Mail Merge Fields

The consolidated workbook and CSV now include student-facing text fields that can be inserted into a mail merge template:

- `PeerReviewSummary`
  Short sentence summarizing the student's peer review completion grade, review count, and time spent.
- `PeerReviewFeedback`
  Expanded plain-text feedback based on the same logic used in the instructor report.
- `PeerReviewNotes`
  Plain-text flags for incomplete reviews, extreme bias, or unusually short completion times.
- `EnhancedGroupPeerReviewPercent`
  Group peer-review percentage extracted from `Group_#Peer_Review_Report.html` when available.
- `EnhancedGroupWeighted50`
  Weighted group peer-review score extracted from `Group_#Peer_Review_Report.html` when available.
- `EnhancedGroupTotalMark60`
  Group total mark out of 60 extracted from `Group_#Peer_Review_Report.html` when available.

### D2L Import Note

The D2L CSV uses this grade column header by default:

```text
Final Assignment Grade Points Grade
```

If your actual D2L grade item has a different exact name, rename that column header in the CSV before importing.

The generated D2L CSV also includes:

```text
Org Defined ID
End-of-Line Indicator
```

which matches the usual Brightspace grade import structure.

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

### Student Roster File (Excel .xlsx or .csv)

**Required Columns:**

```
Email                    - Student email (must match peer review file)
OrgDefinedId            - Student ID from LMS
Product Review Groups   - Group assignment (examples: "Group 01", "Group 1", "Team 01", "01")
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

#### 1. Instructor Report (`instructor_report.py`)

**Line ~252:** Update the `rating_cols` list

```python
rating_cols = [
    "Video_Quality", "Presenters", "Explanation",
    "Mechanism", "Side_Effects", "Bias", 
    "Critical_review", "Study_Quality", "Study_participants",
]
```

#### 2. Group Peer Reports (`group_peer_reports.py`)

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

**Location:** `instructor_report.py`
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

**Location:** `instructor_report.py`
**Line ~358:** Change the 2.0 minute threshold

```python
elif total_time_minutes < 2.0:  # Change this value
    base_grade = max(3.0, base_grade - 3.0)
```

#### Individual Review Time Penalty

**Location:** `instructor_report.py`
**Line ~189:** Change the 1.0 minute threshold

```python
if review_time < 1.0:  # Change this value
    consistency_score *= 0.7
```

### Modifying Bias Detection Thresholds

#### Statistical Outlier Detection

**Location:** `instructor_report.py`
**Line ~361:** Change the 2.0 standard deviation threshold

```python
if not pd.isna(severity_bias_zscore) and abs(severity_bias_zscore) > 2.0:  # Change this value
    base_grade = max(4.0, base_grade - 2.0)
```

#### Minor Bias Reporting Threshold

**Location:** `instructor_report.py`
**Line ~392:** Change the 0.5 threshold for bias reporting

```python
if not pd.isna(severity_bias) and abs(severity_bias) > 0.5:  # Change this value
```

### Modifying Column Names

#### Peer Review File Column Mapping

**Location:** `instructor_report.py`
**Function:** `read_review()`

```python
def read_review(path: Path) -> pd.DataFrame:
    df = _load_table(path)
    email_col = _find_column(df, "review email", ["Email", "E-mail", "Email Address"])
    student_id_col = _find_column(df, "review student ID", ["StudentID", "Student ID", "OrgDefinedId"])
    df["email_clean"] = clean(df[email_col])
    df["student_id_clean"] = clean(df[student_id_col])
    return df
```

#### Student Roster File Column Mapping

**Location:** `instructor_report.py` 
**Function:** `read_roster()`

```python
def read_roster(path: Path) -> pd.DataFrame:
    df = _load_table(path)
    email_col = _find_column(df, "roster email", ["Email", "E-mail", "Email Address"])
    student_id_col = _find_column(df, "roster student ID", ["OrgDefinedId", "Org Defined ID", "StudentID", "Student ID"])
    group_col = _find_column(df, "roster group", ["Product Review Groups", "Product Review Group", "Group", "Groups", "Group Name"])
    df["email_clean"] = clean(df[email_col])
    df["student_id_clean"] = clean(df[student_id_col])
    group_text = df[group_col].astype(str).str.strip()
    extracted_groups = group_text.str.extract(r"(\d{1,3})", expand=False)
    df["Group"] = extracted_groups.fillna(group_text).str.zfill(2)
    return df
```

### Modifying Feedback Column Names

**Location:** `group_peer_reports.py`
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

**Current Format:** The loader extracts the numeric group value from the group column and zero-pads it when needed.
**Location:** `instructor_report.py`

```python
group_text = df[group_col].astype(str).str.strip()
extracted_groups = group_text.str.extract(r"(\d{1,3})", expand=False)
df["Group"] = extracted_groups.fillna(group_text).str.zfill(2)
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

**Location:** `instructor_report.py`
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

**Location:** `instructor_report.py`
**Lines ~483-580:** HTML template definition

#### Group Report Template

**Location:** `group_peer_reports.py`
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

