import matplotlib.pyplot as plt

parameters = [1, 8, 35, 150, 650, 300000]   # M
relation = {'gvp': [0.6875, 0.7183, 0.7820, 0.8267, 0.8544, 0.8703],
            'egnn': [0.7336, 0.7532, 0.7968, 0.8205, 0.8456, 0.8637],
            'molformer': [0.3921, 0.4228, 0.4773, 0.5546, 0.6210, 0.6683]}


def ablation_curve():
    colors = ['darkorange', 'dodgerblue', 'firebrick']
    plt.figure(figsize=(10, 8), dpi=80)
    plt.rcParams.update({'font.size': 24})
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"

    for k, (i, j) in enumerate(relation.items()):
        plt.plot(parameters, j, linewidth=10, color=f'{colors[k]}', label=f'{i.upper()}')

    # plt.title('Training and Validation Losses')
    plt.xlabel('Number of Parameters ($10^6$)', fontsize=24)
    plt.ylabel('Pearson Correlation', fontsize=24)
    plt.xscale('log')
    plt.legend()
    # plt.show()
    plt.savefig('../../../ablation.pdf')

# ablation_curve()


def auroc():
    plt.figure(figsize=(10, 8), dpi=80)
    plt.rcParams.update({'font.size': 24})
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"

    import numpy as np
    X = ['GVP-GNN', 'EGNN', 'Molformer']
    no_plm = [0.8855, 0.8294, 0.8736]
    plm = [0.9215, 0.8940, 0.9403]

    X_axis = np.arange(len(X))
    plt.bar(X_axis - 0.2, no_plm, 0.4, label='w.o. PLM')   # edgecolor='black'
    plt.bar(X_axis + 0.2, plm, 0.4, label='w. PLM')

    plt.xticks(X_axis, X)
    plt.ylabel("AUROC", fontsize=24)
    plt.title("Protein-protein Interface Prediction", fontsize=24)
    plt.ylim(0.8, 0.98)
    plt.legend()
    # plt.show()
    plt.savefig('../../../auroc.pdf')


auroc()


