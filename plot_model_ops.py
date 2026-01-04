import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Data provided by user
ops_counts = {
    'SQUARED_DIFFERENCE': 8, 
    'ADD': 32, 
    'SUB': 8, 
    'RSQRT': 8, 
    'MUL': 12, 
    'FULLY_CONNECTED': 19, 
    'RESHAPE': 24, 
    'TRANSPOSE': 20, 
    'BATCH_MATMUL': 8, 
    'SOFTMAX': 5, 
    'EXPAND_DIMS': 8, 
    'CONV_2D': 8, 
    'MEAN': 1
}

# Define Categories based on acceleration potential
categories = {
    'Compute Core (Tier 1)': ['BATCH_MATMUL', 'CONV_2D', 'FULLY_CONNECTED'],
    'Element-wise (Tier 2)': ['ADD', 'SUB', 'MUL', 'SQUARED_DIFFERENCE'],
    'Complex Math (Tier 3)': ['RSQRT', 'SOFTMAX', 'MEAN'],
    'Data Layout (Tier 4)': ['RESHAPE', 'TRANSPOSE', 'EXPAND_DIMS']
}

# Colors for each category
# Red for Compute (Hot/Critical), Blue for Element-wise, Green for Complex, Gray for Layout
category_colors = {
    'Compute Core (Tier 1)': '#d62728',      # Red
    'Element-wise (Tier 2)': '#1f77b4',      # Blue
    'Complex Math (Tier 3)': '#2ca02c',      # Green
    'Data Layout (Tier 4)': '#7f7f7f'        # Gray
}

# Map ops to colors
op_colors = {}
for op in ops_counts.keys():
    found = False
    for cat, ops in categories.items():
        if op in ops:
            op_colors[op] = category_colors[cat]
            found = True
            break
    if not found:
        op_colors[op] = '#7f7f7f' # Default to Gray if unknown

# Sort data for better visualization (descending order)
sorted_ops = dict(sorted(ops_counts.items(), key=lambda item: item[1], reverse=True))
labels = list(sorted_ops.keys())
values = list(sorted_ops.values())
colors = [op_colors[op] for op in labels]

# Create output directory
output_dir = './results'
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.figure(figsize=(14, 9))

# Create Horizontal Bar Chart
bars = plt.barh(labels, values, color=colors, alpha=0.85)

# Add labels and title
plt.xlabel('Count', fontsize=12, fontweight='bold')
plt.ylabel('Operator Type', fontsize=12, fontweight='bold')
plt.title('TFLite Model Operator Distribution by Acceleration Category', fontsize=16, fontweight='bold', pad=20)

# Invert y-axis to have the highest count at the top
plt.gca().invert_yaxis()

# Add value labels to the end of each bar
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{int(width)}', 
             ha='left', va='center', fontsize=11, fontweight='bold')

# Create Legend
legend_patches = [mpatches.Patch(color=color, label=label) for label, color in category_colors.items()]
plt.legend(handles=legend_patches, title="Operator Categories", loc='lower right', fontsize=11, title_fontsize=12)

plt.tight_layout()
output_path = os.path.join(output_dir, 'model_ops_distribution_colored.png')
plt.savefig(output_path)
print(f"Chart saved to {output_path}")
