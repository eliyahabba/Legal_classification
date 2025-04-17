import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
from bidi.algorithm import get_display
import matplotlib as mpl
import textwrap

from src.utils.constants import DATA_DIR

# File paths
INPUT_FILE = DATA_DIR / "classification_results.csv"
RESULTS_FILE = DATA_DIR / "new_classification_results_with_keywords.csv"
OUTPUT_DIR = DATA_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

def wrap_label(label_text, max_width=18):
    """
    עוטף את הטקסט לשורות, לפי רווחים, בלי לחתוך מילים באמצע.
    max_width קובע בערך כמה תווים בשורה.
    """
    words = label_text.split()
    lines = []
    current_line = []
    current_length = 0

    for w in words:
        if current_length + len(w) + len(current_line) > max_width:
            lines.append(" ".join(current_line))
            current_line = [w]
            current_length = len(w)
        else:
            current_line.append(w)
            current_length += len(w)

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)

def main():
    # Set up matplotlib for Hebrew text
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Load the original dataset
    print("Loading original dataset...")
    original_df = pd.read_csv(INPUT_FILE)
    
    # Print statistics about the original dataset
    print(f"Original dataset contains {len(original_df)} sentences")
    
    # Remove duplicates
    unique_df = original_df.drop_duplicates(subset=['origin_sentence'], keep='first')
    print(f"After removing duplicates: {len(unique_df)} unique sentences")
    
    # Print category distribution in original dataset
    print("\nOriginal category distribution:")
    orig_category_counts = unique_df['category'].value_counts()
    for category, count in orig_category_counts.items():
        percentage = 100 * count / len(unique_df)
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Filter for the specific category we're working with
    target_category = "אמינה כי בהירה, פשוטה, עקבית, אותנטית, ללא הגזמה"
    filtered_df = unique_df[unique_df['category'] == target_category]
    print(f"\nAfter filtering for '{target_category}':")
    print(f"  {len(filtered_df)} sentences to classify")
    
    # Load the new classification results
    if not Path(RESULTS_FILE).exists():
        print("\nNo classification results file found. Run the classifier first.")
        return
    
    results_df = pd.read_csv(RESULTS_FILE)
    print(f"\nLoaded {len(results_df)} classified sentences")
    
    # Calculate new category distribution
    new_category_counts = results_df['new_category'].value_counts()
    new_category_percentages = 100 * new_category_counts / len(results_df)
    
    # Print new category statistics
    print("\nNew category distribution:")
    for category, count in new_category_counts.items():
        percentage = 100 * count / len(results_df)
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Create the visualization
    plt.figure(figsize=(14, 8))
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Count': new_category_counts,
        'Percentage': new_category_percentages
    }).sort_values('Count', ascending=False)
    
    # Process Hebrew text for proper RTL display
    wrapped_and_rtl_categories = [get_display(wrap_label(str(cat))) for cat in plot_data.index]
    
    # Create the bar plot
    ax = sns.barplot(x=np.arange(len(wrapped_and_rtl_categories)), 
                     y='Count', data=plot_data, palette='viridis')
    
    # Set x-tick labels with wrapped-and-RTL text
    ax.set_xticks(np.arange(len(wrapped_and_rtl_categories)))
    ax.set_xticklabels(wrapped_and_rtl_categories, rotation=0, ha='center', fontsize=14)
    
    # Add count labels on top of bars
    for i, (count, percentage) in enumerate(zip(plot_data['Count'], plot_data['Percentage'])):
        ax.text(i, count + 0.5, f"{count}\n({percentage:.1f}%)", 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Set labels and title with RTL text
    plt.title(get_display('התפלגות הקטגוריות החדשות'), fontsize=16)
    plt.xlabel(get_display('קטגוריה'), fontsize=14)
    plt.ylabel(get_display('מספר המשפטים'), fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, ha='center', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_file = OUTPUT_DIR / "category_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_file}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main() 