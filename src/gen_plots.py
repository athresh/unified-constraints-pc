import numpy as np
import pickle
import matplotlib.pyplot as plt
import os 
import seaborn as sns 

sns.set_style("whitegrid")
# sns.set_style('ticks')
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 1.50
    
dataset = "set-mnist-even"
models  = ("EinsumNet/leaf=CategoricalArray","RatSPN/leaf=Categorical")
plot_data = {}
for model in models:
    exp_dir = f"../experiments/{dataset}/{model}"
    plot_data[model] = {}
    for constrained in [False]:
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
                    print(f"{model:50s}", f"constrained={constrained} ", trial,plot_data[model][f"constrained={constrained}"][key][0].shape)
        for key in history:
            plot_data[model][f"constrained={constrained}"][key] = np.vstack(plot_data[model][f"constrained={constrained}"][key])

for model in models:
    exp_dir = f"../experiments/{dataset}/{model}"
    for constrained in [False]:
        dir = f"{exp_dir}/constrained={constrained}"
        plt.figure(figsize=(5,5))
        x_axis   = plot_data[model][f"constrained={constrained}"]['epochs'][0]
        train_ll = plot_data[model][f"constrained={constrained}"]['trn_loss']
        val_ll   = plot_data[model][f"constrained={constrained}"]['val_loss']
        test_ll  = plot_data[model][f"constrained={constrained}"]['tst_loss']
        print(f"x-axis: {x_axis.shape} \t train_ll: {train_ll.shape} \t val_ll: {val_ll.shape} \t test_ll: {test_ll.shape}")
        color = 'r' if constrained else 'b'
        plt.plot(x_axis, train_ll.mean(axis=0), label="Train", ls='--', color=color, lw=2)
        plt.fill_between(x_axis, train_ll.mean(axis=0)-train_ll.std(axis=0),train_ll.mean(axis=0)+train_ll.std(axis=0),color=color, alpha=0.2)
        
        plt.plot(x_axis, val_ll.mean(axis=0), label="Valid", ls='-', color=color, lw=2)
        plt.fill_between(x_axis, val_ll.mean(axis=0)-val_ll.std(axis=0),val_ll.mean(axis=0)+val_ll.std(axis=0),color=color, alpha=0.2)
        
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("Epochs", fontweight='bold', fontsize=14)
        plt.ylabel("Log Likelihood", fontweight='bold', fontsize=14)
        plt.savefig(f"{dir}/learning_curve.png", bbox_inches='tight')