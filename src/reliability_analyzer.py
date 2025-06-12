            icc_data = pd.DataFrame(rating_matrix.T)
            icc_value, icc_p = calculate_icc(icc_data)
            results['icc_value'] = icc_value
            results['icc_p_value'] = icc_p
            results['icc_interpretation'] = self._interpret_icc(icc_value)
        
        # Fleiss' kappa for categorical agreement
        results['fleiss_kappa'] = self.calculate_fleiss_kappa(group_data)
        
        # Variance analysis
        results['between_rater_variance'] = np.var(np.mean(rating_matrix, axis=1))
        results['within_rater_variance'] = np.mean([np.var(rating_matrix[i, :]) for i in range(len(rating_matrix))])
        
        # Bias analysis
        overall_mean = np.mean(rating_matrix)
        rater_means = np.mean(rating_matrix, axis=1)
        results['severity_bias_scores'] = (rater_means - overall_mean).tolist()
        results['range_restriction'] = np.max(rating_matrix) - np.min(rating_matrix)
        
        # Agreement statistics
        pairwise_correlations = []
        for i in range(len(rating_matrix)):
            for j in range(i+1, len(rating_matrix)):
                corr = np.corrcoef(rating_matrix[i, :], rating_matrix[j, :])[0, 1]
                if not np.isnan(corr):
                    pairwise_correlations.append(corr)
        
        results['mean_pairwise_correlation'] = np.mean(pairwise_correlations) if pairwise_correlations else np.nan
        results['min_pairwise_correlation'] = np.min(pairwise_correlations) if pairwise_correlations else np.nan
        results['max_pairwise_correlation'] = np.max(pairwise_correlations) if pairwise_correlations else np.nan
        
        return results
    
    def _interpret_icc(self, icc_value: float) -> str:
        """Provide interpretation of ICC values."""
        if np.isnan(icc_value):
            return "Cannot calculate"
        elif icc_value < 0.5:
            return "Poor reliability"
        elif icc_value < 0.75:
            return "Moderate reliability"
        elif icc_value < 0.9:
            return "Good reliability"
        else:
            return "Excellent reliability"
    
    def full_analysis(self) -> Dict:
        """
        Perform comprehensive reliability analysis across all groups.
        """
        all_results = {
            'overall_stats': {},
            'group_analyses': {},
            'reviewer_profiles': {}
        }
        
        # Overall dataset statistics
        overall_matrix = self.data[self.rating_cols].values
        from enhanced_instructor_report import calculate_cronbach_alpha
        all_results['overall_stats'] = {
            'total_reviews': len(self.data),
            'total_reviewers': self.data['email_clean'].nunique(),
            'total_groups': self.data['Group#_reviewing'].nunique(),
            'overall_cronbach_alpha': calculate_cronbach_alpha(pd.DataFrame(overall_matrix)),
            'overall_mean_rating': np.mean(overall_matrix),
            'overall_std_rating': np.std(overall_matrix)
        }
        
        # Group-by-group analysis
        for group_id, group_data in self.data.groupby('Group#_reviewing'):
            all_results['group_analyses'][group_id] = self.analyze_group_reliability(group_id, group_data)
        
        # Individual reviewer profiles
        for reviewer_id, reviewer_data in self.data.groupby('email_clean'):
            if len(reviewer_data) >= 2:  # Need multiple reviews to assess consistency
                reviewer_ratings = reviewer_data[self.rating_cols].values
                
                profile = {
                    'reviewer_id': reviewer_id,
                    'n_reviews': len(reviewer_data),
                    'mean_severity': np.mean(reviewer_ratings),
                    'consistency_score': 1.0 / (1.0 + np.std(np.mean(reviewer_ratings, axis=1))),
                    'rating_variance': np.var(reviewer_ratings.flatten()),
                    'groups_reviewed': reviewer_data['Group#_reviewing'].tolist()
                }
                
                all_results['reviewer_profiles'][reviewer_id] = profile
        
        return all_results
    
    def generate_reliability_report(self, output_path: Path = None) -> str:
        """
        Generate a comprehensive reliability analysis report.
        """
        results = self.full_analysis()
        
        report = f"""
        PEER REVIEW RELIABILITY ANALYSIS REPORT
        =====================================
        
        OVERALL DATASET STATISTICS
        -------------------------
        Total Reviews: {results['overall_stats']['total_reviews']}
        Total Reviewers: {results['overall_stats']['total_reviewers']}
        Total Groups Reviewed: {results['overall_stats']['total_groups']}
        Overall Cronbach's Alpha: {results['overall_stats']['overall_cronbach_alpha']:.3f}
        Mean Rating: {results['overall_stats']['overall_mean_rating']:.2f}
        Standard Deviation: {results['overall_stats']['overall_std_rating']:.2f}
        
        GROUP-LEVEL RELIABILITY ANALYSIS
        ===============================
        """
        
        for group_id, group_analysis in results['group_analyses'].items():
            if 'error' not in group_analysis:
                report += f"""
        Group {group_id}:
        - Reviewers: {group_analysis['n_reviewers']}
        - Cronbach's Alpha: {group_analysis['cronbach_alpha']:.3f}
        - ICC: {group_analysis.get('icc_value', 'N/A'):.3f} ({group_analysis.get('icc_interpretation', 'N/A')})
        - Fleiss' Kappa: {group_analysis['fleiss_kappa']:.3f}
        - Mean Pairwise Correlation: {group_analysis['mean_pairwise_correlation']:.3f}
        - Range Restriction: {group_analysis['range_restriction']:.2f}
        """
        
        report += f"""
        
        REVIEWER CONSISTENCY PROFILES
        ============================
        """
        
        # Sort reviewers by consistency score
        sorted_reviewers = sorted(
            results['reviewer_profiles'].items(),
            key=lambda x: x[1]['consistency_score'],
            reverse=True
        )
        
        for reviewer_id, profile in sorted_reviewers[:10]:  # Top 10 most consistent
            report += f"""
        {reviewer_id}:
        - Reviews Completed: {profile['n_reviews']}
        - Consistency Score: {profile['consistency_score']:.3f}
        - Mean Severity: {profile['mean_severity']:.2f}
        - Rating Variance: {profile['rating_variance']:.3f}
        """
        
        if output_path:
            output_path.write_text(report, encoding='utf-8')
        
        return report
    
    def create_reliability_plots(self, output_dir: Path = None):
        """
        Create visualization plots for reliability analysis.
        """
        if output_dir:
            output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: ICC values by group
        group_iccs = []
        group_names = []
        for group_id, group_data in self.data.groupby('Group#_reviewing'):
            if len(group_data) >= 2:
                rating_matrix = group_data[self.rating_cols].values
                if len(self.rating_cols) >= 2:
                    from enhanced_instructor_report import calculate_icc
                    icc_data = pd.DataFrame(rating_matrix.T)
                    icc_value, _ = calculate_icc(icc_data)
                    if not np.isnan(icc_value):
                        group_iccs.append(icc_value)
                        group_names.append(f"Group {group_id}")
        
        axes[0, 0].bar(group_names, group_iccs)
        axes[0, 0].set_title('Inter-Rater Reliability (ICC) by Group')
        axes[0, 0].set_ylabel('ICC Value')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].axhline(y=0.75, color='r', linestyle='--', label='Good Reliability Threshold')
        axes[0, 0].legend()
        
        # Plot 2: Distribution of reviewer severity
        reviewer_means = []
        for reviewer_id, reviewer_data in self.data.groupby('email_clean'):
            reviewer_means.append(np.mean(reviewer_data[self.rating_cols].values))
        
        axes[0, 1].hist(reviewer_means, bins=15, alpha=0.7)
        axes[0, 1].set_title('Distribution of Reviewer Severity')
        axes[0, 1].set_xlabel('Mean Rating Given')
        axes[0, 1].set_ylabel('Number of Reviewers')
        axes[0, 1].axvline(x=np.mean(reviewer_means), color='r', linestyle='--', label='Overall Mean')
        axes[0, 1].legend()
        
        # Plot 3: Cronbach's Alpha by group
        group_alphas = []
        alpha_group_names = []
        for group_id, group_data in self.data.groupby('Group#_reviewing'):
            if len(group_data) >= 2:
                rating_matrix = group_data[self.rating_cols].values
                from enhanced_instructor_report import calculate_cronbach_alpha
                alpha = calculate_cronbach_alpha(pd.DataFrame(rating_matrix))
                if not np.isnan(alpha):
                    group_alphas.append(alpha)
                    alpha_group_names.append(f"Group {group_id}")
        
        axes[1, 0].bar(alpha_group_names, group_alphas)
        axes[1, 0].set_title('Internal Consistency (Cronbach\'s α) by Group')
        axes[1, 0].set_ylabel('Cronbach\'s Alpha')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0.7, color='r', linestyle='--', label='Acceptable Threshold')
        axes[1, 0].legend()
        
        # Plot 4: Rating variance by reviewer
        reviewer_variances = []
        for reviewer_id, reviewer_data in self.data.groupby('email_clean'):
            if len(reviewer_data) >= 2:
                reviewer_ratings = reviewer_data[self.rating_cols].values
                reviewer_variances.append(np.var(reviewer_ratings.flatten()))
        
        axes[1, 1].hist(reviewer_variances, bins=15, alpha=0.7)
        axes[1, 1].set_title('Distribution of Reviewer Rating Variance')
        axes[1, 1].set_xlabel('Rating Variance')
        axes[1, 1].set_ylabel('Number of Reviewers')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / 'reliability_analysis_plots.png', dpi=300, bbox_inches='tight')
        
        plt.show()


def analyze_peer_review_file(review_file: Path, roster_file: Path, output_dir: Path = None):
    """
    Convenience function to run complete reliability analysis on peer review files.
    """
    from enhanced_instructor_report import read_review, read_roster
    
    # Load data
    review_df = read_review(review_file)
    roster_df = read_roster(roster_file)
    
    # Merge data
    merged = review_df.merge(
        roster_df[["email_clean", "student_id_clean", "Group"]],
        on="email_clean",
        how="left"
    )
    
    # Define rating columns
    rating_cols = [
        "Video_Quality", "Presenters", "Explanation",
        "Mechanism", "Side_Effects", "Bias",
        "Critical_review", "Study_Quality", "Study_participants",
    ]
    
    # Create analyzer
    analyzer = ReliabilityAnalyzer(merged, rating_cols)
    
    # Generate report
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        report_path = output_dir / 'reliability_analysis_report.txt'
    else:
        report_path = None
    
    report = analyzer.generate_reliability_report(report_path)
    
    # Create plots
    analyzer.create_reliability_plots(output_dir)
    
    return analyzer, report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive reliability analysis for peer reviews")
    parser.add_argument("review_file", type=Path, help="Excel file with peer reviews")
    parser.add_argument("roster_file", type=Path, help="Excel file with student roster")
    parser.add_argument("-o", "--output", type=Path, help="Output directory for reports and plots")
    
    args = parser.parse_args()
    
    analyzer, report = analyze_peer_review_file(args.review_file, args.roster_file, args.output)
    print(report)
