# Feature Engineering

## Overview

Our feature engineering pipeline processes both genomic and clinical data to create robust predictive features for pharmacogenomic analysis.

## Genomic Features

### 1. Variant Encoding

#### SNP Encoding
```python
def encode_variants(variants):
    """
    0: Homozygous reference
    1: Heterozygous
    2: Homozygous alternative
    """
    return np.array([
        0 if v == 'ref/ref' else
        1 if v in ['ref/alt', 'alt/ref'] else
        2 if v == 'alt/alt' else
        np.nan
        for v in variants
    ])
```

#### Structural Variant Processing
- Copy number variations (CNVs)
- Insertions/Deletions
- Complex rearrangements

### 2. Population Structure

#### Principal Components
- First 10 PCs from global ancestry
- Local ancestry inference
- Admixture proportions

#### Genetic Background
- Reference population matching
- Ancestry-specific variants
- Population frequency weighting

### 3. Pharmacogenetic Annotations

#### VIP Gene Processing
- Core pharmacogenes (DPYD, TPMT, UGT1A1)
- Extended pharmacogene set
- Regulatory regions

#### Functional Impact
- CADD scores
- PolyPhen predictions
- SIFT scores
- Conservation metrics

## Clinical Features

### 1. Demographics
- Age
- Sex
- Ethnicity
- Geographic location

### 2. Medical History
- Comorbidities
- Previous ADRs
- Family history
- Current medications

### 3. Laboratory Values
- Liver function tests
- Kidney function markers
- Complete blood count
- Drug levels

## Feature Processing

### 1. Missing Data Handling

```python
IMPUTATION_STRATEGY = {
    'genetic_variants': 'mode',
    'continuous_features': 'knn',
    'categorical_features': 'most_frequent'
}

def impute_missing_data(data, strategy=IMPUTATION_STRATEGY):
    for feature_type, method in strategy.items():
        if method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy=method)
        data[feature_type] = imputer.fit_transform(data[feature_type])
    return data
```

### 2. Feature Scaling

```python
def scale_features(data):
    """
    Standardize continuous features and encode categorical ones
    """
    continuous_scaler = StandardScaler()
    categorical_encoder = OneHotEncoder(sparse=False)
    
    data['continuous'] = continuous_scaler.fit_transform(data['continuous'])
    data['categorical'] = categorical_encoder.fit_transform(data['categorical'])
    
    return data
```

### 3. Feature Selection

#### Genetic Features
- Linkage disequilibrium pruning
- MAF filtering
- Functional relevance
- Known pharmacogenetic associations

#### Clinical Features
- Correlation analysis
- Domain knowledge filtering
- Feature importance ranking
- Multicollinearity reduction

## Feature Importance

### 1. Global Importance
- SHAP values
- Permutation importance
- Feature coefficient analysis
- Information gain

### 2. Population-Specific Importance
- Ancestry-stratified analysis
- Population-specific effects
- Interaction terms

## Usage Example

```python
from src.features import FeatureProcessor

# Initialize processor
processor = FeatureProcessor(
    genetic_features=True,
    clinical_features=True,
    population_structure=True
)

# Process features
X_processed = processor.fit_transform(
    genetic_data=genetic_df,
    clinical_data=clinical_df,
    population_data=population_df
)

# Get feature importance
importance_scores = processor.get_feature_importance(X_processed, y)
```

## Quality Control

### 1. Data Quality Checks
- Missing data assessment
- Outlier detection
- Distribution analysis
- Batch effect evaluation

### 2. Feature Validation
- Cross-validation stability
- Population stratification
- Technical replication
- Biological validation

## Performance Impact

Feature engineering improvements:
- AUROC: +0.05
- Precision: +0.07
- Recall: +0.04
- F1 Score: +0.06
