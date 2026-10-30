"""Fold-fitted descriptors for each prespecified representation."""
import numpy as np
from workflow.data import FEATURES, MaterialDescriptors

COVARIATES = ['I_raw', 'Rs_from_EIS', 'E_IR_corrected_final']
SCHEMES = ['icp', 'precursor', 'fraction', 'unshifted_measured', 'shifted', 'shifted_measured']


class DescriptorTransform:
    def __init__(self, scheme):
        if scheme not in SCHEMES:
            raise ValueError(scheme)
        self.scheme = scheme
        self.material = MaterialDescriptors('precursor' if scheme == 'precursor' else 'icp')
        self.features = list(FEATURES)
        if scheme == 'precursor':
            self.features = [c.replace('icp_weighted_', 'precursor_weighted_') for c in self.features]
        if scheme == 'fraction':
            self.features += ['actual_frac_element1']
        if scheme.endswith('_measured'):
            self.features += COVARIATES

    def fit(self, frame):
        self.material.fit(frame)
        self.medians = self.material.medians
        return self

    def transform(self, frame):
        out = self.material.transform(frame)
        if self.scheme == 'precursor':
            out = out.rename(columns={c: c.replace('icp_weighted_', 'precursor_weighted_') for c in out})
        if self.scheme in ('shifted', 'shifted_measured'):
            offset = frame.E_IR_corrected_final.to_numpy() + .098
            for base, sign in [('work_function', 1), ('pzc', -1), ('E_std', -1)]:
                for c in [base+'_element1', base+'_element2', 'icp_weighted_'+base]:
                    out[c] += sign * offset
        if self.scheme.endswith('_measured'):
            for c in COVARIATES:
                out[c] = frame[c].to_numpy()
        if self.scheme == 'fraction':
            out['actual_frac_element1'] = frame.actual_frac_element1.to_numpy()
        out = out[self.features]
        assert np.isfinite(out.to_numpy()).all()
        return out
