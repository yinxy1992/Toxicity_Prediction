# Toxicity_Prediction
Acute Oral Toxicity Prediction of Chlorobenzene Compounds
## 1. virtual environment
```
conda create -n tox_pred python=3.10
conda activate tox_pred
conda install -c conda-forge rdkit pandas numpy scikit-learn matplotlib seaborn catboost lightgbm shap openpyxl matplotlib-venn
```

## 2. classify by SMATRS
```
python build_global_split.py
```
## 3. gridsearch for RF, LGBM, CatBoost, 9 scripts
```
python gridsearch_F1-C.py	
python gridsearch_F1-G.py
......
python gridsearch_F3-H.py
```
## 4. vote
```
python vote-F1.py
python vote-F2.py
python vote-F3.py
```
## 5. baseline models test
```
python baseline_models.py
```
## 6. chemical diversity
```
python chemical_diversity.py
```
## 7. calibration
```
python calibration_curve-F1.py
python calibration_curve-F2.py
python calibration_curve-F3.py
```
## 8. shap analysis, 9 scripts
```
python shap_stability-F1-C.py
python shap_stability-F1-G.py
......
python shap_stability-F3-H.py
```
## 9. applicability domain,  9 scripts
```
python applicability_domain-F1-C.py
python applicability_domain-F1-G.py
......
python applicability_domain-F3-H.py
```
## 10. P-R curves, 6 scripts
```
python P-R-curves_F1-C.py
python P-R-curves_F1-H.py
......
python P-R-curves_F3-H.py
```
## 11. summarize plot
```
python plot1.py
```
## 12. scatter plot
```
python scatter.py
```
## 13. learning curves
```
python learning_curve.py
```
## 14. class balance
```
python class balance.py
```
## 15. better illustration
```
python batch_shap_analysis.py
python scatter11.py
python heatmap11.py
python chemical_diversity_boxplot.py
```
