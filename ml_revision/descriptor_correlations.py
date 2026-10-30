"""Document isotope-descriptor redundancy across the unique elemental table."""
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from workflow.data import ROOT

def main():
    properties=pd.read_csv(ROOT/'inputs/reference/elemental_properties.csv').set_index('element')
    assert properties.index.is_unique
    rows=[]
    for retained,excluded in [('MH_bond_kJ','MD_bond_kJ'),('Frequency_H','Frequency_D'),('Frequency_H','delta_ZPE_H')]:
        pair=properties[[retained,excluded]].dropna()
        rows.append(dict(retained=retained,excluded=excluded,n_elements=len(pair),
            pearson=float(pearsonr(pair[retained],pair[excluded]).statistic),
            spearman=float(spearmanr(pair[retained],pair[excluded]).statistic),
            cohort='unique elements; pairwise complete; no imputation'))
    out=ROOT/'tables';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'isotope_descriptor_correlations.csv',index=False)

if __name__=='__main__':main()
