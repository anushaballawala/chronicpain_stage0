import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np


def plot_pca_loadings(
    pca_model,
    var_labels,
    title='ptID',
    save_path=None
):

    variance_explained = pca_model.explained_variance_ratio_ * 100

    cumulative_variance = np.cumsum(variance_explained)

    loadings = pca_model.components_

    pc_labels = [
        f'PC{i+1} ({variance_explained[i]:.1f}%)'
        for i in range(len(variance_explained))
    ]

    fig = plt.figure(figsize=(12,4))

    gs = gridspec.GridSpec(
        1,
        2,
        width_ratios=[1,2],
        wspace=0.4
    )

    # =================================
    # Scree plot
    # =================================

    ax1 = fig.add_subplot(gs[0])

    x = np.arange(1, len(variance_explained)+1)

    ax1.bar(
        x,
        variance_explained,
        width=0.5
    )

    ax1.plot(
        x,
        cumulative_variance,
        marker='o',
        linewidth=2
    )

    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained (%)')

    ax1.set_title(f'{title} PCA')

    ax1.set_xticks(x)

    ax1.set_ylim([0,110])

    ax1.spines[['top','right']].set_visible(False)

    ax1.grid(
        axis='y',
        linestyle='--',
        alpha=0.4
    )

    # =================================
    # Loadings heatmap
    # =================================

    ax2 = fig.add_subplot(gs[1])

    sns.heatmap(
        loadings,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        xticklabels=var_labels,
        yticklabels=pc_labels,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={
            'label':'Loading',
            'shrink':0.8
        },
        ax=ax2
    )

    ax2.set_title(f'{title} PCA Loadings',fontsize=13, pad=10)

    ax2.tick_params(
        axis='x',
        rotation=45,
        labelsize=10
    )

    ax2.tick_params(
        axis='y',
        rotation=0,
        labelsize=10
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()