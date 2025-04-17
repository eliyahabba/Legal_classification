import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
from bidi.algorithm import get_display
import argparse

from src.utils.constants import DATA_DIR

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
        # אם המילה מוסיפה אורך גדול מדי, סוגרים שורה ומתחילים חדשה
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


def plot_old_method(counts, percentages, total_len, output_dir, with_train_val=False):
    """
    מצייר איור אחד עבור השיטה הישנה - תלוי אם כוללים גם train/val או לא.
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    # הכנת דאטה
    plot_data = pd.DataFrame({
        'Count': counts,
        'Percentage': percentages
    })
    plot_data = plot_data.sort_values('Count', ascending=False)

    # הכנת תוויות עטופות לשורות
    wrapped_cats = [get_display(wrap_label(str(cat))) for cat in plot_data.index]

    sns.barplot(x=np.arange(len(wrapped_cats)), y='Count', data=plot_data, palette='viridis', ax=ax)
    
    ax.set_xticks(np.arange(len(wrapped_cats)))
    ax.set_xticklabels(wrapped_cats, rotation=0, ha='center', fontsize=16)

    for i, (count, pct) in enumerate(zip(plot_data['Count'], plot_data['Percentage'])):
        ax.text(i, count + 0.5, f"{count}\n({pct:.1f}%)",
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    # כותרת בהתאם לפרמטר with_train_val
    if with_train_val:
        ax.set_title(get_display(f'שיטה ישנה (כולל train/val)\n(סה"כ: {total_len})'), fontsize=20)
    else:
        ax.set_title(get_display(f'שיטה ישנה (רק results)\n(סה"כ: {total_len})'), fontsize=20)

    ax.set_xlabel(get_display('קטגוריה'), fontsize=14)
    ax.set_ylabel(get_display('מספר המשפטים'), fontsize=14)

    fig.suptitle(get_display('התפלגות קטגוריות בשיטה הישנה'), fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    filename = "old_method_with_train_val.png" if with_train_val else "old_method_only_results.png"
    output_file = output_dir / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved old method figure to {output_file}")

    plt.show()
    plt.close(fig)


def plot_new_method(new_counts, new_percentages, new_len, output_dir):
    """
    מצייר איור אחד עבור השיטה החדשה.
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    plot_data = pd.DataFrame({'Count': new_counts, 'Percentage': new_percentages})
    plot_data = plot_data.sort_values('Count', ascending=False)

    wrapped_cats = [get_display(wrap_label(str(cat))) for cat in plot_data.index]
    
    sns.barplot(x=np.arange(len(wrapped_cats)), y='Count', data=plot_data, palette='viridis', ax=ax)
    ax.set_xticks(np.arange(len(wrapped_cats)))
    ax.set_xticklabels(wrapped_cats, rotation=0, ha='center', fontsize=13.5)
    
    for i, (count, pct) in enumerate(zip(plot_data['Count'], plot_data['Percentage'])):
        ax.text(i, count + 0.5, f"{count}\n({pct:.1f}%)",
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_title(get_display(f'שיטה חדשה\n(סה"כ: {new_len})'), fontsize=20)
    ax.set_xlabel(get_display('קטגוריה'), fontsize=14)
    ax.set_ylabel(get_display('מספר המשפטים'), fontsize=14)

    fig.suptitle(get_display('התפלגות קטגוריות בשיטה החדשה'), fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    output_file = output_dir / "new_method.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved new method figure to {output_file}")

    plt.show()
    plt.close(fig)


def main(old_results_file=None, new_results_file=None, include_train_val=False):
    # Default file paths if not provided as arguments
    if old_results_file is None:
        old_results_file = DATA_DIR / "results.csv"
    if new_results_file is None:
        new_results_file = DATA_DIR / "classification_results.csv"
    
    # Output directory
    output_dir = DATA_DIR / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Set up matplotlib for Hebrew text
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Load the old classification results
    print(f"Loading old classification results from {old_results_file}...")
    old_results_df = pd.read_csv(old_results_file)
    print(f"Total rows in old results file: {len(old_results_df)}")
    
    if include_train_val:
        # אם ביקשנו להשתמש גם בקבצי train/val
        train_file = DATA_DIR / "train_sentence_ids.csv"
        val_file = DATA_DIR / "val_sentence_ids.csv"
        
        if train_file.exists():
            train_df = pd.read_csv(train_file)
            old_results_df = pd.concat([old_results_df, train_df], ignore_index=True)
            print(f"Combined train file: {len(train_df)} rows")
        else:
            print(f"Warning: train file not found: {train_file}")
        
        if val_file.exists():
            val_df = pd.read_csv(val_file)
            old_results_df = pd.concat([old_results_df, val_df], ignore_index=True)
            print(f"Combined val file: {len(val_df)} rows")
        else:
            print(f"Warning: val file not found: {val_file}")

        print(f"After combining train/val, total old rows: {len(old_results_df)}")
    
    # הסרת כפילויות מהשיטה הישנה
    unique_old_results_df = old_results_df.drop_duplicates(subset=['sentence_id','title'], keep='first')
    print(f"Unique in old results: {len(unique_old_results_df)}")
    
    # חישוב התפלגויות
    old_counts = unique_old_results_df['label_0'].value_counts()
    old_percentages = 100 * old_counts / len(unique_old_results_df)
    
    # הדפסה
    print("\nOld method category distribution:")
    for category, count in old_counts.items():
        pct = 100 * count / len(unique_old_results_df)
        print(f"  {category}: {count} ({pct:.1f}%)")
    
    # טוענים את השיטה החדשה
    print(f"\nLoading new classification results from {new_results_file}...")
    new_results_df = pd.read_csv(new_results_file)
    print(f"Total rows in new results file: {len(new_results_df)}")
    
    # מסירים כפילויות
    unique_new_results_df = new_results_df.drop_duplicates(subset=['sentence_text'], keep='first')
    print(f"Unique in new results: {len(unique_new_results_df)}")
    
    # איחוד הקטגוריה שמתחילה ב"עוסק בעמדת הצדדים"
    unified_category = "עוסק בעמדת הצדדים בנוגע להערכת המהימנות ולא עמדת השופט, הלכות כלליות בנוגע להערכת מהימנות, לא נוגע להערכה מהימנות כלל"
    unique_new_results_df = unique_new_results_df.copy()
    mask = unique_new_results_df['category'].astype(str).str.startswith('עוסק בעמדת הצדדים')
    unique_new_results_df.loc[mask, 'category'] = unified_category
    print(f"Unified {mask.sum()} rows in new results for 'עוסק בעמדת הצדדים'")
    
    # חישוב התפלגויות שיטה חדשה
    new_counts = unique_new_results_df['category'].value_counts()
    new_percentages = 100 * new_counts / len(unique_new_results_df)

    print("\nNew method category distribution:")
    for category, count in new_counts.items():
        pct = 100 * count / len(unique_new_results_df)
        print(f"  {category}: {count} ({pct:.1f}%)")
    
    # ציור איור אחד של השיטה הישנה
    plot_old_method(
        counts=old_counts,
        percentages=old_percentages,
        total_len=len(unique_old_results_df),
        output_dir=output_dir,
        with_train_val=include_train_val
    )
    
    # ציור איור אחד של השיטה החדשה
    plot_new_method(
        new_counts,
        new_percentages,
        len(unique_new_results_df),
        output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare category distributions between old and new classification methods')
    parser.add_argument('--old', type=str, help='Path to old classification results CSV file')
    parser.add_argument('--new', type=str, help='Path to new classification results CSV file')
    parser.add_argument('--include-train-val', action='store_true', help='Include train and validation data in old method', default=True)
    
    args = parser.parse_args()
    
    old_results_path = Path(args.old) if args.old else None
    new_results_path = Path(args.new) if args.new else None
    
    main(old_results_path, new_results_path, args.include_train_val) 