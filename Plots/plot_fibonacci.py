import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set scientific style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

def plot_generalization_gap():
    # Data from Table 1
    relations = ['Fibonacci', 'Linear', 'Tribonacci', 'Geometric', 'Fibo+1']
    train_acc = [100, 100, 100, 100, 100]
    # Using the Sequential Test Accuracy (the "Real" test)
    test_acc = [31, 0, 0, 40, 37.5]  # Averaged 18-44 to 31 for plotting

    x = np.arange(len(relations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, train_acc, width, label='Training (Seen Seeds)', color='#2ecc71')
    rects2 = ax.bar(x + width/2, test_acc, width, label='Testing (Novel Seeds)', color='#e74c3c')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('The Generalization Gap: Rule Learning Fails')
    ax.set_xticks(x)
    ax.set_xticklabels(relations)
    ax.legend()
    ax.set_ylim(0, 110)

    # Add value labels
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.tight_layout()
    plt.savefig('generalization_gap.png', dpi=300)
    print("Saved generalization_gap.png")

def plot_probing_results():
    # Data from your probing_results.json
    layers = [0, 1, 2]
    acc_n1 = [62.5, 62.5, 62.5]  # F(n-1)
    acc_n2 = [58.3, 58.3, 58.3]  # F(n-2)
    
    plt.figure(figsize=(8, 5))
    
    # Plot lines
    plt.plot(layers, acc_n1, marker='o', linewidth=3, label='Probe F(n-1) [Ratio Term]', color='#3498db')
    plt.plot(layers, acc_n2, marker='s', linewidth=3, label='Probe F(n-2) [Rule Term]', color='#e67e22', linestyle='--')
    
    # Add reference lines
    plt.axhline(y=100, color='green', linestyle=':', alpha=0.5, label='Perfect Rule Encoding')
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Random Chance')
    
    plt.title('Mechanistic Probing: Flat, Weak Representations')
    plt.xlabel('Transformer Layer')
    plt.ylabel('Probe Accuracy (%)')
    plt.ylim(40, 105)
    plt.xticks(layers)
    plt.legend(loc='center right')
    
    # Annotate the gap
    plt.text(1, 52, "Weak 'Fuzzy' Memory", ha='center', color='gray', style='italic')
    
    plt.tight_layout()
    plt.savefig('probing_results.png', dpi=300)
    print("Saved probing_results.png")

if __name__ == "__main__":
    plot_generalization_gap()
    plot_probing_results()