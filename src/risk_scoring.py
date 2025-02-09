import numpy as np
import pandas as pd
from scipy import stats
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class DrugRiskScorer:
    def __init__(self):
        # Define weights for different variants based on clinical significance
        self.variant_weights = {
            'DPYD': {
                'rs3918290': 0.4,  # Most severe - complete DPD deficiency
                'rs67376798': 0.35,  # Severe - partial DPD deficiency
                'rs55886062': 0.25   # Moderate - reduced DPD activity
            },
            'TPMT': {
                'rs1800462': 0.4,  # Most severe - complete TPMT deficiency
                'rs1800460': 0.35,  # Severe - partial TPMT deficiency
                'rs1142345': 0.25   # Moderate - reduced TPMT activity
            },
            'UGT1A1': {
                'rs8175347': 0.6,   # Most severe - Gilbert syndrome
                'rs4148323': 0.4    # Moderate - reduced UGT1A1 activity
            }
        }
        
        # Define drug-specific risk thresholds
        self.risk_thresholds = {
            '5-FU': {
                'high': 0.7,
                'moderate': 0.4,
                'low': 0.2
            },
            'Capecitabine': {
                'high': 0.65,
                'moderate': 0.35,
                'low': 0.15
            },
            'Irinotecan': {
                'high': 0.6,
                'moderate': 0.3,
                'low': 0.1
            },
            'Thiopurines': {
                'high': 0.75,
                'moderate': 0.45,
                'low': 0.25
            }
        }

    def calculate_gene_risk_score(self, frequencies, gene):
        """Calculate risk score for a specific gene"""
        if gene not in self.variant_weights:
            return 0.0
        
        score = 0.0
        weights = self.variant_weights[gene]
        
        for variant, weight in weights.items():
            if variant in frequencies:
                score += frequencies[variant]['risk_allele_frequency'] * weight
        
        return score

    def calculate_population_risk_scores(self, allele_frequencies):
        """Calculate risk scores for each population"""
        population_scores = {}
        
        for pop_data in allele_frequencies:
            for gene in ['DPYD', 'TPMT', 'UGT1A1']:
                if gene in pop_data:
                    pop = pop_data[gene][next(iter(pop_data[gene]))]['population']
                    if pop not in population_scores:
                        population_scores[pop] = {}
                    
                    score = self.calculate_gene_risk_score(pop_data[gene], gene)
                    population_scores[pop][gene] = score
        
        return population_scores

    def classify_drug_risk(self, gene_scores):
        """Classify drug risks based on gene scores"""
        drug_risks = {}
        
        # 5-FU and Capecitabine (DPYD)
        if 'DPYD' in gene_scores:
            dpyd_score = gene_scores['DPYD']
            for drug in ['5-FU', 'Capecitabine']:
                thresholds = self.risk_thresholds[drug]
                if dpyd_score >= thresholds['high']:
                    risk_level = 'High'
                elif dpyd_score >= thresholds['moderate']:
                    risk_level = 'Moderate'
                else:
                    risk_level = 'Low'
                drug_risks[drug] = {'risk_level': risk_level, 'score': dpyd_score}
        
        # Thiopurines (TPMT)
        if 'TPMT' in gene_scores:
            tpmt_score = gene_scores['TPMT']
            thresholds = self.risk_thresholds['Thiopurines']
            if tpmt_score >= thresholds['high']:
                risk_level = 'High'
            elif tpmt_score >= thresholds['moderate']:
                risk_level = 'Moderate'
            else:
                risk_level = 'Low'
            drug_risks['Thiopurines'] = {'risk_level': risk_level, 'score': tpmt_score}
        
        # Irinotecan (UGT1A1)
        if 'UGT1A1' in gene_scores:
            ugt1a1_score = gene_scores['UGT1A1']
            thresholds = self.risk_thresholds['Irinotecan']
            if ugt1a1_score >= thresholds['high']:
                risk_level = 'High'
            elif ugt1a1_score >= thresholds['moderate']:
                risk_level = 'Moderate'
            else:
                risk_level = 'Low'
            drug_risks['Irinotecan'] = {'risk_level': risk_level, 'score': ugt1a1_score}
        
        return drug_risks

    def generate_population_risk_report(self, population_scores):
        """Generate a detailed risk report for each population"""
        reports = {}
        
        for population, gene_scores in population_scores.items():
            drug_risks = self.classify_drug_risk(gene_scores)
            
            reports[population] = {
                'gene_scores': gene_scores,
                'drug_risks': drug_risks,
                'recommendations': self.generate_recommendations(drug_risks)
            }
        
        return reports

    def generate_recommendations(self, drug_risks):
        """Generate clinical recommendations based on drug risks"""
        recommendations = []
        
        for drug, risk_info in drug_risks.items():
            if risk_info['risk_level'] == 'High':
                recommendations.append(f"{drug}: Consider alternative therapy. If essential, reduce dose by 50-75% and monitor closely.")
            elif risk_info['risk_level'] == 'Moderate':
                recommendations.append(f"{drug}: Start with 25-50% reduced dose. Monitor closely and adjust based on response.")
            else:
                recommendations.append(f"{drug}: Standard dosing appropriate. Regular monitoring recommended.")
        
        return recommendations

    def visualize_population_risks(self, population_scores, output_dir):
        """Create visualizations of population risk scores"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for plotting
        plot_data = []
        for pop, gene_scores in population_scores.items():
            for gene, score in gene_scores.items():
                plot_data.append({
                    'Population': pop,
                    'Gene': gene,
                    'Risk Score': score
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        heatmap_data = df.pivot(index='Population', columns='Gene', values='Risk Score')
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Population-Specific Genetic Risk Scores')
        plt.tight_layout()
        plt.savefig(output_dir / 'risk_heatmap.png')
        plt.close()
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Population', y='Risk Score', hue='Gene')
        plt.title('Population Risk Scores by Gene')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'risk_barplot.png')
        plt.close()

def main():
    try:
        # Load allele frequencies
        with open('gwas_results/allele_frequencies.json', 'r') as f:
            allele_frequencies = json.load(f)
        
        # Initialize risk scorer
        scorer = DrugRiskScorer()
        
        # Calculate population risk scores
        population_scores = scorer.calculate_population_risk_scores(allele_frequencies)
        
        # Generate risk reports
        risk_reports = scorer.generate_population_risk_report(population_scores)
        
        # Create output directory
        output_dir = Path('risk_analysis_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save risk reports
        with open(output_dir / 'population_risk_reports.json', 'w') as f:
            json.dump(risk_reports, f, indent=2)
        
        # Generate visualizations
        scorer.visualize_population_risks(population_scores, output_dir)
        
        print("Risk analysis completed. Results saved in 'risk_analysis_results' directory.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
