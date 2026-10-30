"""Shared composition weighting and joint-panel plotting helpers."""
import sys
from workflow.data import ROOT

HELPER=ROOT.parent/'0-Figures/Figure4/descriptor_joint_panel_helpers.py'
sys.path.insert(0,str(HELPER.parent))
import descriptor_joint_panel_helpers as joint


def weighted(data,base,mode):
    f=data['actual_frac_element1' if mode=='icp' else 'nominal_frac_element1']
    return (f*data[base+'_element1']).where(f.ne(0),0)+((1-f)*data[base+'_element2']).where(f.ne(1),0)
