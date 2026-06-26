"""
Create comprehensive accuracy comparison showing the full story:
1. Single-step training accuracy (100%)
2. Sequential evaluation on training seeds (44%)
3. Sequential evaluation on novel seeds (18%)
"""
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style("whitegrid")

# Load data
with open('results/evaluation_summary.json', 'r') as f:
    eval_results = json.load(f)

with open('results/training_history.json', 'r') as f:
    train_history = json.load(f)

# Get final training accuracy (single-step)
final_train_acc = train_history['train_accuracies'][-1]

# Get sequential evaluation accuracies
seq_train_acc = eval_results['training_seeds']['average_accuracy']
seq_novel_acc = eval_results['novel_seeds']['average_accuracy']

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

categories = [
    'Single-Step\nTraining',
    'Sequential\nTraining Seeds',
    'Sequential\nNovel Seeds'
]
accuracies = [final_train_acc, seq_train_acc, seq_novel_acc]
colors = ['#3498db', '#2ecc71', '#e74c3c']  # Blue, Green, Red

bars = ax.bar(categories, accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=16, fontweight='bold')

# Add horizontal line at 100%
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax.text(0.02, 102, '100% (Perfect)', fontsize=10, color='gray', style='italic')

# Add horizontal line at 50% (random baseline)
ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
ax.text(0.02, 52, 'Random baseline', fontsize=10, color='gray', style='italic')

# Labels and title
ax.set_ylabel('Exact Match Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Performance: The Complete Picture', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 110)

# Add annotations
ax.annotate('', xy=(0, 97), xytext=(1, 97),
            arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
ax.text(0.5, 92, '56 pp drop', ha='center', fontsize=11,
        color='red', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=1.5))

ax.annotate('', xy=(1, 41), xytext=(2, 41),
            arrowprops=dict(arrowstyle='<->', lw=2, color='darkred'))
ax.text(1.5, 36, '26 pp drop', ha='center', fontsize=11,
        color='darkred', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='darkred', linewidth=1.5))

# Add text box explaining
textstr = ('Single-Step: Next-token prediction with correct context\n'
           'Sequential: Autoregressive generation (positions 10-30)')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('visualizations/comprehensive_accuracy.png', dpi=150, bbox_inches='tight')
print("✓ Created comprehensive_accuracy.png")
print(f"\nShowing:")
print(f"  Single-step training: {final_train_acc:.1f}%")
print(f"  Sequential training:  {seq_train_acc:.1f}%")
print(f"  Sequential novel:     {seq_novel_acc:.1f}%")
