"""Plot precursor and ICP composition for initial and independent BO samples."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from workflow.data import ROOT, load_data
import plot_style


BO_STYLES = {
    'Co-Fe': ('#d62728', 'o'),
    'Co-Mo': ('#d95f02', 's'),
    'Co-Ni': ('#1f77b4', '^'),
    'Co-V': ('#14866d', 'D'),
    'Fe-Ni': ('#9467bd', 'P'),
}


def main():
    out=ROOT/'figures/precursor_vs_icp_composition'
    out.mkdir(parents=True,exist_ok=True)
    data,_=load_data()
    initial=data[(data.cohort=='initial')&(data.element1!=data.element2)]
    bo=data[(data.cohort=='BO')&(data.element1!=data.element2)].copy()
    bo['system']=bo.element1+'-'+bo.element2
    assert len(initial)==531 and len(bo)==111
    assert bo.system.value_counts().to_dict()=={
        'Co-Fe':27,'Co-Mo':21,'Co-Ni':21,'Co-V':21,'Fe-Ni':21}
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    axes[0].scatter(initial.nominal_frac_element1,initial.actual_frac_element1,
                    s=18,alpha=.40,color='#777777',edgecolors='none',zorder=2)
    for system,(color,marker) in BO_STYLES.items():
        group=bo[bo.system==system]
        axes[1].scatter(group.nominal_frac_element1,group.actual_frac_element1,
                        s=33,alpha=.95,color=color,marker=marker,linewidths=.4,
                        label=f'{system} (n = {len(group)})',zorder=3)
    axes[0].set_title(f'Initial bimetallic samples | n = {len(initial)}',fontsize=12)
    axes[1].set_title(f'Independent BO samples | n = {len(bo)}',fontsize=12)
    axes[1].legend(loc='upper left',frameon=False,fontsize=9,
                   handletextpad=.6,labelspacing=.45)
    for ax in axes:
        ax.plot([0,1],[0,1],'--',color='black',lw=.9,zorder=1)
        ax.set(xlim=(-.04,1.04),ylim=(-.04,1.04))
        ax.set_xlabel('Nominal precursor M1 volume fraction',fontsize=11)
        ax.set_ylabel('ICP M1 molar fraction',fontsize=11)
        ax.tick_params(labelsize=10,direction='out')
        ax.spines[['top','right']].set_visible(False)
    for ext in ['png','svg','pdf']:
        fig.savefig(out/f'precursor_vs_icp_composition.{ext}',dpi=300)
    plt.close(fig)
    print(f'Composition figure: {len(initial)} initial bimetallic samples; '
          f'{len(bo)} independent BO samples. No model fits.')


if __name__=='__main__': main()
