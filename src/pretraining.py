import pandas as pd
import numpy as np
from scipy import stats
import logging
import os
import sys
import json
from tqdm import tqdm
import time

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

print_flush("Script started")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

print_flush("Logging configured")

class DNAAnalyzer:
    def __init__(self):
        # Initialize VIP gene patterns
        self.vip_genes = {
            'CYP2D6': ['TGAAGCC', 'GCAAGGA'],  # Common CYP2D6 motifs
            'CYP2C19': ['GAAGAGT', 'CTGCCAT'],  # CYP2C19 regulatory elements
            'CYP3A4': ['ATGAACT', 'AGGTCA'],    # CYP3A4 binding sites
            'SLCO1B1': ['TGACCT', 'GGTTCA'],    # SLCO1B1 transporter motifs
            'VKORC1': ['GGCACG', 'CCAAT'],      # VKORC1 regulatory elements
            'TPMT': ['GGTCCT', 'AGGCCA'],       # TPMT enzyme motifs
            'UGT1A1': ['GTGAGT', 'CCTGCT'],     # UGT1A1 promoter elements
            'DPYD': ['CCGCCC', 'TATAAA']        # DPYD regulatory regions
        }
        
        # Initialize cancer-related VIP gene patterns and their clinical implications
        self.cancer_vip_genes = {
            'DPYD': {
                'motifs': ['CCGCCC', 'TATAAA', 'GCAAGT'],  # Key DPYD regulatory regions
                'drugs': ['5-Fluorouracil', 'Capecitabine'],
                'adverse_reactions': {
                    'high_risk': [
                        'Severe myelosuppression',
                        'Gastrointestinal toxicity',
                        'Hand-foot syndrome',
                        'Potentially fatal toxicity'
                    ],
                    'variants': {
                        'CCGCCC': 'Reduced enzyme activity (~50%)',
                        'TATAAA': 'Severely reduced activity (~25%)',
                        'GCAAGT': 'Normal activity'
                    }
                }
            },
            'TPMT': {
                'motifs': ['GGTCCT', 'AGGCCA', 'TTTGGT'],  # TPMT enzyme motifs
                'drugs': ['6-Mercaptopurine', 'Azathioprine', 'Thioguanine'],
                'adverse_reactions': {
                    'high_risk': [
                        'Severe myelosuppression',
                        'Increased risk of secondary cancers',
                        'Hepatotoxicity'
                    ],
                    'variants': {
                        'GGTCCT': 'Reduced activity (~50%)',
                        'AGGCCA': 'Severely reduced activity (~10%)',
                        'TTTGGT': 'Normal activity'
                    }
                }
            },
            'UGT1A1': {
                'motifs': ['GTGAGT', 'CCTGCT', 'TATATAA'],  # UGT1A1 promoter elements
                'drugs': ['Irinotecan', 'Nilotinib', 'Pazopanib'],
                'adverse_reactions': {
                    'high_risk': [
                        'Severe neutropenia',
                        'Severe diarrhea',
                        'Increased bilirubin levels'
                    ],
                    'variants': {
                        'GTGAGT': 'Reduced activity (~60%)',
                        'CCTGCT': 'Severely reduced activity (~30%)',
                        'TATATAA': 'Normal activity'
                    }
                }
            }
        }
        
    def calculate_gc_content(self, sequence):
        """Calculate GC content of a DNA sequence"""
        gc_count = sequence.count('G') + sequence.count('C')
        total_bases = len(sequence)
        return gc_count / total_bases if total_bases > 0 else 0
    
    def find_motifs(self, sequence):
        """Find common DNA motifs in the sequence"""
        common_motifs = [
            'TATA',  # TATA box
            'CAAT',  # CAAT box
            'GATA',  # GATA binding
            'GGAA',  # ETS binding
            'CCAAT'  # CAT box
        ]
        
        found_motifs = []
        for motif in common_motifs:
            count = sequence.count(motif)
            if count > 0:
                found_motifs.append(f"{motif} (x{count})")
        
        return found_motifs
    
    def find_vip_genes(self, sequence):
        """Find VIP gene motifs in the sequence"""
        found_vip_genes = {}
        
        for gene, motifs in self.vip_genes.items():
            matches = []
            for motif in motifs:
                count = sequence.count(motif)
                if count > 0:
                    matches.append((motif, count))
            if matches:
                found_vip_genes[gene] = matches
                
        return found_vip_genes
    
    def analyze_cancer_vip_genes(self, sequence):
        """Analyze cancer-related VIP genes and predict adverse reactions"""
        results = {}
        
        for gene, data in self.cancer_vip_genes.items():
            gene_results = {
                'found_motifs': [],
                'predicted_activity': 'Normal',
                'risk_level': 'Low',
                'affected_drugs': [],
                'potential_reactions': []
            }
            
            # Find motifs and assess risk
            reduced_activity_count = 0
            severe_reduction_count = 0
            
            for motif in data['motifs']:
                count = sequence.count(motif)
                if count > 0:
                    variant_effect = data['adverse_reactions']['variants'].get(motif, 'Unknown effect')
                    gene_results['found_motifs'].append({
                        'motif': motif,
                        'count': count,
                        'effect': variant_effect
                    })
                    
                    if 'reduced activity (~50%)' in variant_effect.lower():
                        reduced_activity_count += count
                    elif 'severely reduced' in variant_effect.lower():
                        severe_reduction_count += count
            
            # Assess overall risk and predict reactions
            if severe_reduction_count > 0:
                gene_results['predicted_activity'] = 'Severely Reduced'
                gene_results['risk_level'] = 'High'
                gene_results['affected_drugs'] = data['drugs']
                gene_results['potential_reactions'] = data['adverse_reactions']['high_risk']
            elif reduced_activity_count > 0:
                gene_results['predicted_activity'] = 'Reduced'
                gene_results['risk_level'] = 'Moderate'
                gene_results['affected_drugs'] = data['drugs']
                gene_results['potential_reactions'] = [r for r in data['adverse_reactions']['high_risk'] 
                                                     if 'fatal' not in r.lower()]
            
            if gene_results['found_motifs']:
                results[gene] = gene_results
        
        return results
    
    def analyze_sequence(self, sequence_data):
        """Analyze a DNA sequence"""
        sequence = sequence_data['sequence']
        population = sequence_data['population']
        known_gc = sequence_data['gc_content']
        
        print_flush(f"\nAnalyzing sequence from {population}...")
        print_flush(f"Sequence length: {len(sequence)} bases")
        print_flush(f"Known GC content: {known_gc:.2%}")
        
        # Calculate GC content
        calculated_gc = self.calculate_gc_content(sequence)
        
        # Find motifs
        motifs = self.find_motifs(sequence)
        
        # Analyze cancer VIP genes
        cancer_vip_analysis = self.analyze_cancer_vip_genes(sequence)
        
        analysis = {
            'population': population,
            'sequence_length': len(sequence),
            'calculated_gc_content': calculated_gc,
            'known_gc_content': known_gc,
            'gc_content_difference': abs(calculated_gc - known_gc),
            'found_motifs': motifs,
            'cancer_vip_analysis': cancer_vip_analysis,
            'sequence_start': sequence[:50] + "..." if len(sequence) > 50 else sequence
        }
        
        print_flush("\nAnalysis results:")
        print_flush(f"- Population: {analysis['population']}")
        print_flush(f"- Sequence length: {analysis['sequence_length']} bases")
        print_flush(f"- Calculated GC content: {analysis['calculated_gc_content']:.2%}")
        
        if cancer_vip_analysis:
            print_flush("\n- Cancer VIP genes found:")
            for gene, results in cancer_vip_analysis.items():
                print_flush(f"  * {gene}:")
                print_flush(f"    Activity: {results['predicted_activity']}")
                print_flush(f"    Risk Level: {results['risk_level']}")
                if results['affected_drugs']:
                    print_flush(f"    Affected Drugs: {', '.join(results['affected_drugs'])}")
                if results['potential_reactions']:
                    print_flush("    Potential Reactions:")
                    for reaction in results['potential_reactions']:
                        print_flush(f"      - {reaction}")
        
        print_flush("-" * 80)
        return analysis
    
    def process_sequences(self, sequences, output_dir):
        """Process a list of DNA sequences"""
        os.makedirs(output_dir, exist_ok=True)
        
        analyses = []
        population_stats = {}
        
        for i, seq in enumerate(tqdm(sequences, desc="Analyzing sequences")):
            try:
                print_flush(f"\nProcessing sequence {i+1}/{len(sequences)}")
                analysis = self.analyze_sequence(seq)
                analyses.append(analysis)
                
                # Update population statistics
                pop = analysis['population']
                if pop not in population_stats:
                    population_stats[pop] = {
                        'count': 0,
                        'total_gc': 0,
                        'total_length': 0,
                        'motif_counts': {}
                    }
                
                stats = population_stats[pop]
                stats['count'] += 1
                stats['total_gc'] += analysis['calculated_gc_content']
                stats['total_length'] += analysis['sequence_length']
                
                for motif in analysis['found_motifs']:
                    motif_name = motif.split(' ')[0]
                    if motif_name not in stats['motif_counts']:
                        stats['motif_counts'][motif_name] = 0
                    stats['motif_counts'][motif_name] += 1
                
                # Save progress periodically
                if (i + 1) % 10 == 0:
                    self._save_results(analyses, population_stats, output_dir)
                    
            except Exception as e:
                print_flush(f"Error analyzing sequence: {str(e)}")
                continue
        
        # Save final results
        self._save_results(analyses, population_stats, output_dir)
    
    def _save_results(self, analyses, population_stats, output_dir):
        """Save analysis results to files"""
        # Save individual analyses
        analyses_file = os.path.join(output_dir, "sequence_analyses.jsonl")
        with open(analyses_file, "w") as f:
            for analysis in analyses:
                f.write(json.dumps(analysis) + "\n")
        
        # Generate and save summary
        summary_file = os.path.join(output_dir, "analysis_summary.txt")
        with open(summary_file, "w") as f:
            f.write("Cancer VIP Gene Analysis Summary\n")
            f.write("==============================\n\n")
            f.write(f"Total sequences analyzed: {len(analyses)}\n\n")
            
            # Analyze cancer VIP genes by population
            cancer_vip_stats = {
                'DPYD': {'high_risk': {}, 'moderate_risk': {}},
                'TPMT': {'high_risk': {}, 'moderate_risk': {}},
                'UGT1A1': {'high_risk': {}, 'moderate_risk': {}}
            }
            
            for analysis in analyses:
                pop = analysis['population']
                vip_analysis = analysis.get('cancer_vip_analysis', {})
                
                for gene, results in vip_analysis.items():
                    if results['risk_level'] == 'High':
                        if pop not in cancer_vip_stats[gene]['high_risk']:
                            cancer_vip_stats[gene]['high_risk'][pop] = 0
                        cancer_vip_stats[gene]['high_risk'][pop] += 1
                    elif results['risk_level'] == 'Moderate':
                        if pop not in cancer_vip_stats[gene]['moderate_risk']:
                            cancer_vip_stats[gene]['moderate_risk'][pop] = 0
                        cancer_vip_stats[gene]['moderate_risk'][pop] += 1
            
            # Write cancer VIP gene statistics
            f.write("Cancer VIP Gene Risk Distribution:\n")
            f.write("--------------------------------\n")
            for gene in cancer_vip_stats:
                f.write(f"\n{gene}:\n")
                
                f.write("  High Risk Cases:\n")
                for pop, count in cancer_vip_stats[gene]['high_risk'].items():
                    percentage = (count / population_stats[pop]['count']) * 100
                    f.write(f"    - {pop}: {count} sequences ({percentage:.1f}%)\n")
                
                f.write("\n  Moderate Risk Cases:\n")
                for pop, count in cancer_vip_stats[gene]['moderate_risk'].items():
                    percentage = (count / population_stats[pop]['count']) * 100
                    f.write(f"    - {pop}: {count} sequences ({percentage:.1f}%)\n")
                
                f.write("\n  Clinical Implications:\n")
                for drug in self.cancer_vip_genes[gene]['drugs']:
                    f.write(f"    - {drug}\n")
                f.write("\n  Potential Adverse Reactions:\n")
                for reaction in self.cancer_vip_genes[gene]['adverse_reactions']['high_risk']:
                    f.write(f"    - {reaction}\n")
                f.write("\n")
            
            f.write("\nPopulation Statistics:\n")
            f.write("--------------------\n")
            for pop, stats in population_stats.items():
                f.write(f"\n{pop}:\n")
                f.write(f"  Total Sequences: {stats['count']}\n")
        
        print_flush(f"\nResults saved to {output_dir}")

class GWASAnalyzer:
    def __init__(self):
        # Define cancer-related SNPs and their associations
        self.cancer_snps = {
            'DPYD': {
                'rs3918290': {'alleles': ['G', 'A'], 'risk_allele': 'A', 'phenotype': '5-FU toxicity'},
                'rs67376798': {'alleles': ['T', 'A'], 'risk_allele': 'A', 'phenotype': 'Fluoropyrimidine toxicity'},
                'rs55886062': {'alleles': ['A', 'T'], 'risk_allele': 'T', 'phenotype': 'DPD deficiency'}
            },
            'TPMT': {
                'rs1800462': {'alleles': ['G', 'C'], 'risk_allele': 'C', 'phenotype': 'TPMT deficiency'},
                'rs1800460': {'alleles': ['A', 'G'], 'risk_allele': 'G', 'phenotype': 'Thiopurine toxicity'},
                'rs1142345': {'alleles': ['T', 'C'], 'risk_allele': 'C', 'phenotype': 'Reduced TPMT activity'}
            },
            'UGT1A1': {
                'rs8175347': {'alleles': ['6', '7'], 'risk_allele': '7', 'phenotype': 'Gilbert syndrome'},
                'rs4148323': {'alleles': ['G', 'A'], 'risk_allele': 'A', 'phenotype': 'Irinotecan toxicity'}
            }
        }
        
        # Define phenotype associations
        self.phenotype_associations = {
            'drug_response': [
                'Treatment efficacy',
                'Adverse reactions',
                'Drug metabolism rate'
            ],
            'cancer_risk': [
                'Disease susceptibility',
                'Progression rate',
                'Treatment outcome'
            ],
            'population_effects': [
                'Ethnic-specific variations',
                'Geographic distribution',
                'Ancestral patterns'
            ]
        }
    
    def calculate_allele_frequencies(self, sequences, population):
        """Calculate allele frequencies for known cancer-related SNPs"""
        frequencies = {}
        
        for gene, snps in self.cancer_snps.items():
            frequencies[gene] = {}
            for snp_id, info in snps.items():
                allele_counts = {allele: 0 for allele in info['alleles']}
                total_count = 0
                
                for seq in sequences:
                    # Simulate SNP detection in sequence
                    # In real GWAS, you would have actual SNP data
                    for allele in info['alleles']:
                        if allele in seq['sequence']:
                            allele_counts[allele] += seq['sequence'].count(allele)
                            total_count += seq['sequence'].count(allele)
                
                if total_count > 0:
                    frequencies[gene][snp_id] = {
                        'allele_frequencies': {
                            allele: count/total_count for allele, count in allele_counts.items()
                        },
                        'risk_allele_frequency': allele_counts[info['risk_allele']]/total_count if total_count > 0 else 0,
                        'population': population,
                        'phenotype': info['phenotype']
                    }
        
        return frequencies
    
    def perform_association_analysis(self, all_frequencies):
        """Perform basic association analysis between populations"""
        associations = {}
        
        for gene in self.cancer_snps.keys():
            associations[gene] = {}
            for snp_id in self.cancer_snps[gene].keys():
                populations = []
                risk_freqs = []
                
                for pop_freqs in all_frequencies:
                    if gene in pop_freqs and snp_id in pop_freqs[gene]:
                        populations.append(pop_freqs[gene][snp_id]['population'])
                        risk_freqs.append(float(pop_freqs[gene][snp_id]['risk_allele_frequency']))  # Convert to float
                
                if len(populations) > 1:
                    # Perform one-way ANOVA to test for significant differences
                    f_stat, p_value = stats.f_oneway(*[
                        [freq] for freq in risk_freqs
                    ])
                    
                    associations[gene][snp_id] = {
                        'populations': populations,
                        'risk_frequencies': [float(f) for f in risk_freqs],  # Convert to float
                        'p_value': float(p_value),  # Convert to float
                        'significant': bool(p_value < 0.05)  # Convert to bool
                    }
        
        return associations
    
    def suggest_gwas_studies(self, associations):
        """Generate GWAS study suggestions based on analysis results"""
        suggestions = []
        
        for gene, snps in associations.items():
            for snp_id, data in snps.items():
                if data['significant']:
                    # Find populations with highest difference
                    max_freq = max(data['risk_frequencies'])
                    min_freq = min(data['risk_frequencies'])
                    max_pop = data['populations'][data['risk_frequencies'].index(max_freq)]
                    min_pop = data['populations'][data['risk_frequencies'].index(min_freq)]
                    
                    phenotype = self.cancer_snps[gene][snp_id]['phenotype']
                    
                    suggestion = {
                        'title': f"GWAS Study: {gene} {snp_id} in {max_pop} vs {min_pop}",
                        'rationale': f"Significant variation in {phenotype} risk allele frequency " +
                                   f"({max_freq:.2%} vs {min_freq:.2%})",
                        'focus_areas': [
                            f"Population-specific {phenotype} mechanisms",
                            "Drug response variation",
                            "Genetic modifiers"
                        ],
                        'priority': 'High' if (max_freq - min_freq) > 0.1 else 'Medium'
                    }
                    suggestions.append(suggestion)
        
        return suggestions

def main():
    print_flush("Entering main function")
    try:
        logger.info("Starting DNA sequence analysis process...")
        print_flush("Starting DNA sequence analysis process...")
        
        # Load H3Africa reference sequences
        csv_path = os.path.join(os.getcwd(), "data", "h3africa", "processed", "h3africa_reference_sequences.csv")
        print_flush(f"Loading sequences from {csv_path}")
        
        if not os.path.exists(csv_path):
            error_msg = f"Could not find sequence file at {csv_path}"
            print_flush(error_msg)
            raise FileNotFoundError(error_msg)
        
        print_flush("Reading CSV file...")
        df = pd.read_csv(csv_path)
        print_flush(f"Loaded {len(df)} sequences")
        
        # Convert DataFrame rows to list of dictionaries
        sequences = [
            {
                "sequence": row["sequence"],
                "population": row["population"],
                "gc_content": row["gc_content"]
            }
            for _, row in df.iterrows()
        ]
        
        # Initialize analyzer
        print_flush("Initializing DNA Sequence Analyzer...")
        analyzer = DNAAnalyzer()
        
        # Process sequences
        output_dir = os.path.join(os.getcwd(), "analysis_results")
        print_flush(f"Starting analysis process. Results will be saved to {output_dir}")
        analyzer.process_sequences(sequences, output_dir)
        
        # Group sequences by population
        population_sequences = {}
        for _, row in df.iterrows():
            pop = row['population']
            if pop not in population_sequences:
                population_sequences[pop] = []
            population_sequences[pop].append({
                'sequence': row['sequence'],
                'gc_content': row['gc_content']
            })
        
        # Initialize GWAS analyzer
        print_flush("Initializing GWAS Analyzer...")
        gwas_analyzer = GWASAnalyzer()
        
        # Calculate frequencies for each population
        print_flush("Calculating allele frequencies...")
        all_frequencies = []
        for pop, sequences in population_sequences.items():
            frequencies = gwas_analyzer.calculate_allele_frequencies(sequences, pop)
            all_frequencies.append(frequencies)
        
        # Perform association analysis
        print_flush("Performing association analysis...")
        associations = gwas_analyzer.perform_association_analysis(all_frequencies)
        
        # Generate GWAS study suggestions
        print_flush("Generating GWAS study suggestions...")
        suggestions = gwas_analyzer.suggest_gwas_studies(associations)
        
        # Save results
        gwas_output_dir = os.path.join(os.getcwd(), "gwas_results")
        os.makedirs(gwas_output_dir, exist_ok=True)
        
        # Save frequencies
        frequencies_file = os.path.join(gwas_output_dir, "allele_frequencies.json")
        with open(frequencies_file, "w") as f:
            json.dump(all_frequencies, f, indent=2)
        
        # Save associations
        associations_file = os.path.join(gwas_output_dir, "population_associations.json")
        with open(associations_file, "w") as f:
            json.dump(associations, f, indent=2)
        
        # Save suggestions
        suggestions_file = os.path.join(gwas_output_dir, "gwas_suggestions.txt")
        with open(suggestions_file, "w") as f:
            f.write("Suggested GWAS Studies\n")
            f.write("=====================\n\n")
            
            for i, suggestion in enumerate(suggestions, 1):
                f.write(f"{i}. {suggestion['title']}\n")
                f.write(f"   Priority: {suggestion['priority']}\n")
                f.write(f"   Rationale: {suggestion['rationale']}\n")
                f.write("   Focus Areas:\n")
                for area in suggestion['focus_areas']:
                    f.write(f"    - {area}\n")
                f.write("\n")
        
        print_flush(f"\nResults saved to {gwas_output_dir}")
        
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print_flush(error_msg)
        logger.error(error_msg, exc_info=True)
        raise

if __name__ == "__main__":
    print_flush("Starting script execution")
    main()
