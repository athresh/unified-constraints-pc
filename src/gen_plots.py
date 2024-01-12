import numpy as np
import pickle
import matplotlib.pyplot as plt
import os 
import seaborn as sns 

sns.set_style("whitegrid")
# sns.set_style('ticks')
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 1.50
    
dataset = "set-mnist-odd"
models  = ("EinsumNet/leaf=CategoricalArray","RatSPN/leaf=Categorical")
plot_data = {}
for model in models:
    exp_dir = f"../experiments/{dataset}/{model}"
    plot_data[model] = {}
    for constrained in [True, False]:
        dir = f"{exp_dir}/constrained={constrained}"
        plot_data[model][f"constrained={constrained}"] = {}
        for trial in os.listdir(dir):
            if(not os.path.isdir(os.path.join(dir, trial))):
                continue
            if(not os.path.exists(os.path.join(dir, trial, "results", "losses.pkl"))):
                continue
            with open(os.path.join(dir, trial, "results", "losses.pkl"), 'rb') as f:
                history = pickle.load(f)
                for key in history:
                    plot_data[model][f"constrained={constrained}"][key] = [np.array(history[key])] if key not in plot_data[model][f"constrained={constrained}"] else [np.array(history[key])] + plot_data[model][f"constrained={constrained}"][key]
                print(f"{model:35s}", f"constrained={str(constrained):6s} ", trial, f"{history['tst_loss'][-1]:.3f} {len(history['tst_loss'])}")
        for key in history:
            plot_data[model][f"constrained={constrained}"][key] = np.vstack(plot_data[model][f"constrained={constrained}"][key])

print("\n-->Test Results")
for model in models:
    exp_dir = f"../experiments/{dataset}/{model}"
    plt.figure(figsize=(5,4))
    for constrained in [True, False]:
        dir = f"{exp_dir}/constrained={constrained}"
        x_axis   = plot_data[model][f"constrained={constrained}"]['epochs'][0]
        train_ll = plot_data[model][f"constrained={constrained}"]['trn_loss']
        val_ll   = plot_data[model][f"constrained={constrained}"]['val_loss']
        test_ll  = plot_data[model][f"constrained={constrained}"]['tst_loss']
        color = 'r' if constrained else 'b'
        label = model.split('/')[0] 
        label = f"{label}" if not constrained else f"{label} + GC"
        mean, std = train_ll.mean(axis=0), train_ll.std(axis=0)
        plt.plot(x_axis, mean, label=f"Train: {label}", ls='--', color=color, lw=2)
        plt.fill_between(x_axis, mean-std,mean+std,color=color, alpha=0.15)
        
        mean, std = test_ll.mean(axis=0), test_ll.std(axis=0)/np.sqrt(test_ll.shape[0])
        plt.plot(x_axis, mean, label=f"Valid: {label}", ls='-', color=color, lw=2)
        plt.fill_between(x_axis, mean-std, mean+std,color=color, alpha=0.15)
        
        print(f"{dataset:20s} {model:32s}", f"constrained={str(constrained):6s}: {test_ll.mean(axis=0)[-1]:.3f} +- {test_ll.std(axis=0)[-1]/np.sqrt(test_ll.shape[0]):.3f}", )
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Epochs", fontweight='bold', fontsize=15)
    plt.ylabel("Log Likelihood", fontweight='bold', fontsize=15)
    plt.savefig(f"{exp_dir}/{dataset}_{model.split('/')[0]}_learning_curve.png", bbox_inches='tight')
    