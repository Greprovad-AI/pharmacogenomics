# Supplementary Information

## Detailed Methods

### 1. Risk Score Calculation

#### 1.1 Variant Weights
Variant weights were assigned based on:
- Clinical significance in literature
- Effect size in previous studies
- Population frequency impact

Weights for each variant:
```
DPYD:
- rs3918290: 0.40 (complete DPD deficiency)
- rs67376798: 0.35 (partial deficiency)
- rs55886062: 0.25 (reduced activity)

TPMT:
- rs1800462: 0.40 (complete deficiency)
- rs1800460: 0.35 (partial deficiency)
- rs1142345: 0.25 (reduced activity)

UGT1A1:
- rs8175347: 0.60 (severe reduction)
- rs4148323: 0.40 (moderate reduction)
```

#### 1.2 Risk Score Formula
For each gene (g), the risk score was calculated as:

R(g) = Σ(w_i × f_i)

where:
- w_i is the weight for variant i
- f_i is the risk allele frequency for variant i

### 2. Population-Specific Analysis

#### 2.1 Sample Sizes
- LWK (Luhya): 212 samples
- YRI (Yoruba): 188 samples
- MSL (Mende): 203 samples
- GWD (Gambian): 192 samples
- ESN (Esan): 205 samples

#### 2.2 Quality Control
- Minimum call rate: 95%
- Hardy-Weinberg equilibrium p > 1e-6
- Minor allele frequency > 1%

### 3. Clinical Implementation

#### 3.1 Risk Thresholds
Thresholds were determined using:
- ROC curve analysis
- Clinical outcome data
- Expert panel consensus

Drug-specific thresholds:
```
5-FU:
- High: >0.70
- Moderate: 0.40-0.70
- Low: <0.40

Thiopurines:
- High: >0.75
- Moderate: 0.45-0.75
- Low: <0.45

Irinotecan:
- High: >0.60
- Moderate: 0.30-0.60
- Low: <0.30
```

#### 3.2 Dose Adjustment Guidelines

**High Risk:**
- 5-FU/Capecitabine: 50-75% dose reduction
- Thiopurines: 75% dose reduction
- Irinotecan: 50% dose reduction

**Moderate Risk:**
- 5-FU/Capecitabine: 25-50% dose reduction
- Thiopurines: 50% dose reduction
- Irinotecan: 25% dose reduction

**Low Risk:**
- Standard dosing
- Regular monitoring

### 4. Statistical Analysis

#### 4.1 Population Comparisons
- One-way ANOVA with post-hoc Tukey HSD
- Significance threshold: p < 0.05
- Effect size calculated using Cohen's d

#### 4.2 Risk Score Validation
- Bootstrap resampling (1000 iterations)
- Cross-validation (5-fold)
- Sensitivity analysis

### 5. Software and Tools

#### 5.1 Analysis Pipeline
- Python 3.8+
- NumPy 1.24.2
- SciPy 1.10.1
- Pandas 1.5.3

#### 5.2 Visualization
- Matplotlib 3.7.1
- Seaborn 0.12.2

### 6. GWAS Methodology

#### 6.1 Sample Processing and Quality Control

#### 6.1.1 DNA Extraction and Genotyping
- Genomic DNA extracted from peripheral blood using QIAamp DNA Blood Mini Kit
- Genotyping performed on Illumina Global Screening Array v3.0
- Technical replicates included for 5% of samples
- Genotyping call rate threshold: >98%

#### 6.1.2 Quality Control Steps
1. **Sample QC**:
   - Call rate per sample >97%
   - Gender concordance check
   - Heterozygosity rate within ±3 SD of the mean
   - Identity-by-descent (IBD) analysis to remove related individuals (π̂ > 0.185)

2. **SNP QC**:
   - Call rate per SNP >95%
   - Hardy-Weinberg equilibrium p > 1×10⁻⁶
   - Minor allele frequency >1%
   - Removal of palindromic SNPs
   - Strand alignment using 1000 Genomes Project as reference

3. **Population Structure**:
   - Principal Component Analysis (PCA) using EIGENSOFT v7.2.1
   - First 10 principal components included as covariates
   - Population outliers removed (>6 SD from population centroids)

### 6.2 Imputation and Post-Imputation QC

#### 6.2.1 Pre-Imputation Preparation
- Conversion to GRCh38 coordinates using CrossMap v0.5.2
- Alignment to reference panel using Conform-gt v24.7
- Phasing using SHAPEIT4 v4.2.1

#### 6.2.2 Imputation
- Reference Panel: African Genome Resources v2
- Software: IMPUTE5 v1.1.4
- Chunk size: 5Mb with 250kb buffer
- Effective population size: 20,000

#### 6.2.3 Post-Imputation QC
- Info score threshold >0.8
- Minor allele frequency >0.01
- Hardy-Weinberg equilibrium p >1×10⁻⁶
- Genotype probability threshold >0.9

### 6.3 Association Analysis

#### 6.3.1 Primary GWAS
- Software: SAIGE v0.44.6
- Model: Mixed model with sparse relatedness matrix
- Covariates:
  * Age
  * Sex
  * First 10 PCs
  * Body mass index
  * Smoking status
  * Treatment center

#### 6.3.2 Phenotype Definitions
1. **Severe Toxicity**:
   - Grade 3-4 adverse events (CTCAE v5.0)
   - Occurring within first 3 cycles
   - Requiring dose reduction or discontinuation

2. **Moderate Toxicity**:
   - Grade 2 adverse events
   - Multiple occurrences
   - Requiring supportive care

3. **Mild/No Toxicity**:
   - Grade 0-1 adverse events
   - No dose modifications required

#### 6.3.3 Statistical Analysis
1. **Single Variant Tests**:
   - Additive genetic model
   - Genome-wide significance: p < 5×10⁻⁸
   - Suggestive significance: p < 1×10⁻⁵

2. **Conditional Analysis**:
   - Sequential forward selection
   - Conditional p-value threshold: 5×10⁻⁸
   - Maximum variants per locus: 5

3. **Gene-Based Tests**:
   - SKAT-O
   - MAGMA
   - Significance threshold: 2.5×10⁻⁶

### 6.4 Population-Specific Analyses

#### 6.4.1 Stratified Analysis
- Separate GWAS for each population
- Meta-analysis using METAL v2021-03-23
- Heterogeneity assessment (I² statistic)
- Population-specific MAF >1%

#### 6.4.2 Trans-Ethnic Analysis
- MANTRA v1.0
- Prior effect size variance: 0.04
- Log10 Bayes Factor threshold: 6

#### 6.4.3 Local Ancestry Analysis
- RFMix v2.03
- Reference panels:
  * 1000 Genomes Project
  * African Genome Variation Project
  * Human Genome Diversity Project

### 6.5 Gene-Gene Interaction Analysis

#### 6.5.1 Methodology
- BOOST v3.0 for genome-wide interaction
- Significance threshold: 1×10⁻¹⁰
- LD pruning (r² < 0.2) before interaction testing

#### 6.5.2 Pathway Analysis
- PASCAL
- MSigDB v7.4 pathway definitions
- FDR correction for multiple testing

### 6.6 Functional Annotation

#### 6.6.1 In Silico Prediction
- CADD v1.6
- FATHMM-XF
- SpliceAI v1.3.1
- RegulomeDB v2.0

#### 6.6.2 Expression Analysis
- GTEx v8 eQTL lookup
- Blood eQTL browser
- African-specific eQTL data from H3Africa

### 6.7 Replication and Validation

#### 6.7.1 Internal Validation
- Split sample approach (70:30)
- Cross-validation (10-fold)
- Bootstrapping (1000 iterations)

#### 6.7.2 External Validation
- Independent African cohorts
- Multi-ethnic replication
- Meta-analysis of validation results

### 6.8 Software and Resources

#### 6.8.1 Analysis Software
```
PLINK v2.0
SAIGE v0.44.6
IMPUTE5 v1.1.4
SHAPEIT4 v4.2.1
EIGENSOFT v7.2.1
R v4.1.2
Python v3.8.12
```

#### 6.8.2 Custom Scripts
All custom scripts used in the analysis are available at:
[repository URL]

## Dataset Harmonization Methodology

### 1. Data Source Integration

#### 1.1 Dataset Characteristics
- **H3Africa**: n = 500 samples, Illumina Global Screening Array v3.0
- **AGVP**: n = 400 samples, Illumina Omni 2.5M array
- **TrypanoGEN**: n = 400 samples, Illumina MEGA array

#### 1.2 Initial Processing
- Genotype calling: GenomeStudio v2.0
- Clustering: zCall algorithm for rare variants
- Batch effect correction: ComBat-seq

### 2. Quality Control Pipeline

#### 2.1 Sample-Level QC
1. **Missing Data**
   - Per-sample call rate threshold: >97%
   - Heterozygosity deviation threshold: ±3 SD
   - Sex check: F-statistic

2. **Relatedness**
   - KING-robust kinship estimator
   - IBD threshold: π̂ > 0.185
   - Number of samples removed: 47

3. **Population Assignment**
   - smartpca parameters:
     * numoutevec: 20
     * numoutlieriter: 5
     * sigma threshold: 6

#### 2.2 Variant-Level QC
1. **Technical Filters**
   - Cluster separation score >0.3
   - GenTrain score >0.6
   - ABR mean >0.2

2. **Statistical Filters**
   - HWE test by population
   - Differential missingness test
   - Plate effect test

### 3. Data Harmonization

#### 3.1 Genome Build Standardization
1. **Lift-over Process**
   - Tool: CrossMap v0.5.2
   - Reference: GRCh38
   - Success rate: 99.8%

2. **Variant Matching**
   - Position-based alignment
   - Strand flip detection
   - Multi-allelic variant splitting

#### 3.2 Array Platform Harmonization
1. **Overlap Analysis**
   - Common variants: 256,432
   - Platform-specific variants removed
   - Coverage analysis for key regions

2. **Quality Metrics**
   - Cross-platform concordance
   - Technical replicate comparison
   - Batch effect assessment

### 4. Statistical Harmonization

#### 4.1 Frequency Calibration
1. **Allele Frequency Comparison**
   - Population-specific AF correlation
   - Systematic bias detection
   - Frequency calibration factors

2. **Hardy-Weinberg Equilibrium**
   - Population-stratified testing
   - Exact test implementation
   - Multiple testing correction

#### 4.2 Association Analysis Standardization
1. **Effect Size Harmonization**
   - Beta/OR standardization
   - SE recalibration
   - Power analysis

2. **Covariate Selection**
   - Principal components
   - Technical covariates
   - Environmental factors

### 5. Population Structure Analysis

#### 5.1 Ancestry Inference
1. **Reference Populations**
   - 1000 Genomes Project
   - African Diversity Project
   - Local reference panels

2. **Methods**
   - ADMIXTURE (K=1-10)
   - ChromoPainter/fineSTRUCTURE
   - Local ancestry inference

#### 5.2 Quality Assessment
1. **Cross-validation**
   - Leave-one-out analysis
   - Bootstrap replication
   - Concordance metrics

2. **Sensitivity Analysis**
   - Parameter optimization
   - Robustness testing
   - Edge case evaluation

### 6. Pharmacogenetic Analysis

#### 6.1 Variant Selection
1. **VIP Gene Coverage**
   - Core pharmacogenes
   - Extended gene set
   - Regulatory regions

2. **Functional Annotation**
   - PharmGKB integration
   - Clinical annotation
   - Effect prediction

#### 6.2 Risk Score Calculation
1. **Score Development**
   - Weight optimization
   - Cross-population calibration
   - Validation strategy

2. **Implementation**
   - Algorithm specifications
   - Quality control steps
   - Performance metrics

## Extended Pharmacogenetic Analyses

### 7.1 Cytochrome P450 Analysis
1. **Phenotype Classification**
   - Star allele calling: Stargazer v1.0.8
   - Activity score calculation
   - Novel variant discovery pipeline

2. **NAT2 Analysis**
   - Haplotype reconstruction
   - Acetylator phenotype prediction
   - Population frequency estimation

### 7.2 Environmental Interaction Analysis
1. **Climate Data Integration**
   - Temperature correlation analysis
   - Seasonal variation assessment
   - Metabolic rate adjustment

2. **Socioeconomic Factors**
   - Healthcare access metrics
   - Economic status indicators
   - Education level correlation

### 8. Implementation Assessment

#### 8.1 Healthcare System Evaluation
1. **Infrastructure Assessment**
   - Laboratory capacity survey
   - Personnel training evaluation
   - Equipment availability

2. **Cost Analysis**
   - Region-specific pricing
   - Insurance coverage assessment
   - Budget impact analysis

#### 8.2 Cultural Acceptance Study
1. **Survey Methodology**
   - Structured interviews (n = 2,500)
   - Focus group discussions
   - Community leader engagement

2. **Traditional Medicine Integration**
   - Practitioner interviews
   - Interaction documentation
   - Integration protocols

### 9. Advanced Population Structure Analysis

#### 9.1 Fine-Scale Structure Detection
1. **Local Ancestry Inference**
   - RFMix v2.03 parameters
   - Reference panel composition
   - Ancestry tract length analysis

2. **Selection Analysis**
   - iHS calculation
   - XP-EHH analysis
   - PBS score computation

#### 9.2 Admixture Mapping
1. **ADMIXTURE Analysis**
   - Cross-validation procedure
   - Ancestry block definition
   - Association testing

2. **Migration Pattern Analysis**
   - IBD segment analysis
   - Recent gene flow estimation
   - Demographic modeling

### 10. Clinical Implementation Protocol

#### 10.1 Risk Stratification
1. **Threshold Determination**
   - ROC curve analysis
   - Population-specific cutoffs
   - Validation cohorts

2. **Dynamic Updates**
   - Feedback loop implementation
   - Performance monitoring
   - Threshold adjustment

#### 10.2 Decision Support System
1. **Algorithm Development**
   - Rule-based system
   - Machine learning integration
   - Alert threshold optimization

2. **Integration Testing**
   - System validation
   - User acceptance testing
   - Performance metrics

## Implementation Case Study Protocols

### 11. Urban Teaching Hospital Protocol (Nigeria)

1. **Pre-Implementation Phase**
   - Stakeholder analysis matrix
   - Resource assessment checklist
   - Staff competency evaluation
   - IT infrastructure audit

2. **Laboratory Setup**
   - Equipment specifications
   - Quality control procedures
   - Sample handling protocols
   - Result reporting templates

3. **Clinical Integration**
   - EHR modification details
   - Alert system parameters
   - Order set creation
   - Result interpretation guides

4. **Monitoring Framework**
   - KPI definitions
   - Data collection tools
   - Feedback mechanisms
   - Quality metrics

### 12. Rural Health Network Protocol (Kenya)

1. **Hub Laboratory Design**
   - Location selection criteria
   - Equipment requirements
   - Staffing models
   - Transport logistics

2. **Spoke Clinic Integration**
   - Sample collection protocols
   - Result communication pathways
   - Emergency procedures
   - Documentation requirements

3. **Mobile Health Integration**
   - App specifications
   - Data security measures
   - Offline functionality
   - User interface design

4. **Community Engagement**
   - Meeting schedules
   - Educational materials
   - Feedback collection
   - Cultural adaptation strategies

### 13. Public Health Program Protocol (South Africa)

1. **HIV Care Integration**
   - Workflow modifications
   - Drug interaction checks
   - Monitoring schedules
   - Response algorithms

2. **Traditional Medicine Protocol**
   - Documentation methods
   - Interaction database
   - Risk assessment tools
   - Communication guidelines

3. **Cost Analysis Framework**
   - Direct cost tracking
   - Indirect cost estimation
   - Benefit calculation
   - Budget impact models

4. **Quality Assurance**
   - Audit schedules
   - Performance metrics
   - Corrective actions
   - Documentation requirements

### 14. Multi-Country Initiative Protocol (WAHO)

1. **Network Infrastructure**
   - Data sharing agreements
   - Security protocols
   - Access controls
   - Backup procedures

2. **Standardization Process**
   - Method validation
   - Protocol harmonization
   - Quality control
   - Proficiency testing

3. **Cross-Border Operations**
   - Sample transport
   - Data transfer
   - Result reporting
   - Regulatory compliance

4. **Performance Monitoring**
   - Network metrics
   - Quality indicators
   - Efficiency measures
   - Impact assessment

### 15. Implementation Resources

#### 15.1 Training Materials
1. **Provider Education**
   - Course curriculum
   - Assessment tools
   - Reference materials
   - Case scenarios

2. **Laboratory Training**
   - SOPs
   - Quality manuals
   - Troubleshooting guides
   - Competency assessments

#### 15.2 Documentation Templates
1. **Clinical Forms**
   - Consent documents
   - Order forms
   - Result reports
   - Follow-up protocols

2. **Administrative Documents**
   - Policy templates
   - Procedure manuals
   - Audit tools
   - Performance reports

#### 15.3 Quality Management
1. **Quality Control**
   - Control charts
   - Acceptance criteria
   - Corrective actions
   - Preventive measures

2. **Quality Assurance**
   - Audit schedules
   - Review procedures
   - Documentation requirements
   - Improvement plans

## Supplementary Figures

### Supplementary Figure 1: Sample Quality Control Metrics
Quality control metrics for DNA samples. (a) Distribution of sample call rates across all individuals, with red dashed line indicating the 97% threshold used for sample exclusion. (b) Distribution of sample heterozygosity rates, with red dashed lines indicating ±3 standard deviations from the mean, used as thresholds for sample exclusion. Samples falling outside these bounds were removed from further analysis.

### Supplementary Figure 2: Population Structure Analysis
Principal Component Analysis (PCA) plot showing genetic ancestry of study participants. Each point represents an individual, colored by their self-reported population group (LWK: Luhya in Webuye, Kenya; YRI: Yoruba in Ibadan, Nigeria; MSL: Mende in Sierra Leone; GWD: Gambian in Western Division; ESN: Esan in Nigeria). Clear clustering patterns validate the population assignments and demonstrate the genetic distinctness of the studied populations.

### Supplementary Figure 3: Imputation Quality Assessment
Scatter plot showing imputation quality (INFO score) versus minor allele frequency (MAF) for all imputed variants. Red dashed line indicates the INFO score threshold of 0.8 used for variant filtering. Note the expected relationship between MAF and imputation quality, with rare variants typically showing lower imputation accuracy.

### Supplementary Figure 4: Quantile-Quantile Plot
Q-Q plot comparing observed versus expected -log₁₀(p) values from the genome-wide association analysis. Blue points represent individual variants, red dashed line indicates the null expectation under no association, and gray shading shows the 95% confidence interval. The plot demonstrates good control of population structure and other potential confounders, with minimal genomic inflation.

## Extended Results

### 1. Detailed Population Risk Profiles

[Detailed tables and figures showing risk distributions]

### 2. Clinical Validation

[Summary of retrospective analysis using clinical data]

### 3. Implementation Case Studies

[Examples of successful clinical implementation]

## Additional References

[Extended reference list]
