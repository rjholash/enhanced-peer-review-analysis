# Quick Reference: Enhanced Peer Review Tools

## Getting Started (5 Minutes)

### Step 1: Run the Enhanced Tool

```bash
python enhanced_gui_launcher.py
```

### Step 2: Select Files

1. **Peer Review File**: Your downloaded Excel file from the online form
2. **Student Roster**: Your class list Excel file
3. **Output Location**: Where to save the enhanced report

### Step 3: Interpret Results

Open the HTML report and look for:

## Key Metrics to Watch

| Metric                | Good Values | Action if Low               |
| --------------------- | ----------- | --------------------------- |
| **Cronbach's Alpha**  | > 0.7       | Clarify rubric criteria     |
| **ICC (Inter-rater)** | > 0.75      | Provide reviewer training   |
| **Reliability Score** | > 60        | Individual student training |

## Quick Interpretation Guide

### Group-Level Issues

- **Low Cronbach's Alpha (< 0.7)**: Students interpreting criteria differently
- **Low ICC (< 0.5)**: High disagreement between reviewers
- **High Standard Deviation**: Inconsistent rating standards

### Individual Reviewer Issues

- **Reliability Score < 40**: Needs immediate attention
- **High Severity Bias**: Consistently rates too high/low
- **Low Consistency**: Disagrees with peer consensus

## Action Items by Reliability Score

### Score 80-100: Excellent Reviewers

- Use as peer mentors
- Consider for grading assistance
- Exemplars for training others

### Score 60-79: Good Reviewers

- Standard grading weight
- Minor calibration may help
- Generally reliable

### Score 40-59: Needs Improvement

- Reduce grading weight
- Provide additional examples
- One-on-one discussion

### Score < 40: Immediate Intervention

- Exclude from peer grading
- Mandatory re-training
- Consider alternative assessment

## File Locations

All new files are in your existing directory:

```
C:\Users\holas\OneDrive - University of Calgary\Programming\course_tools\Peer-Review\
├── enhanced_instructor_report.py     # Main enhanced analysis
├── enhanced_gui_launcher.py          # Easy-to-use interface  
├── reliability_analyzer.py           # Deep statistical analysis
├── README_Enhanced_Analysis.md       # Full documentation
└── Quick_Reference.md                # This file
```

## Common Workflows

### Regular Course Management

1. Run `enhanced_gui_launcher.py` after each peer review
2. Check group reliability metrics
3. Identify students with scores < 60
4. Provide targeted feedback

### Research/Publication

1. Use `reliability_analyzer.py` for comprehensive statistics
2. Generate publication-quality plots
3. Extract detailed statistical measures
4. Document assessment validation

### Grade Adjustment

1. Note reliability scores for each student
2. Weight peer evaluation grades accordingly
3. Consider excluding unreliable reviewers
4. Document decisions for transparency

## Need Help?

- **Full Documentation**: See `README_Enhanced_Analysis.md`
- **Statistical Details**: Comments in source code files
- **Method References**: Standard psychometric literature on ICC and Cronbach's alpha
