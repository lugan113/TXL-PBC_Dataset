import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_bounding_box_area(label_file):
    areas = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            width = float(parts[3])
            height = float(parts[4])
            area = width * height
            areas.append(area)
    return areas

def load_dataset_areas(label_dir, dataset_name):
    areas = []
    for label_file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, label_file)
        areas.extend(calculate_bounding_box_area(file_path))
    return pd.DataFrame({'Dataset': [dataset_name] * len(areas), 'Bounding Box Area': areas})

bccd_label_dir = 'path_to_bccd_labels'
bcd_label_dir = 'path_to_bcd_labels'
txlpbc_label_dir = 'path_to_txlpbc_labels'

bccd_areas = load_dataset_areas(bccd_label_dir, 'BCCD')
bcd_areas = load_dataset_areas(bcd_label_dir, 'BCD')
txlpbc_areas = load_dataset_areas(txlpbc_label_dir, 'TXL-PBC')

df = pd.concat([bccd_areas, bcd_areas, txlpbc_areas])

plt.figure(figsize=(10, 6))
sns.boxplot(x='Dataset', y='Bounding Box Area', data=df)
plt.title('Box Plot of Bounding Box Areas')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Dataset', y='Bounding Box Area', data=df)
plt.title('Violin Plot of Bounding Box Areas')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=range(len(df)), y='Bounding Box Area', hue='Dataset', data=df)
plt.title('Scatter Plot of Bounding Box Areas')
plt.show()

label_counts = {
    'Dataset': ['BCCD', 'BCD', 'TXL-PBC'],
    'RBC Labels': [18000, 17000, 25000],
    'WBC Labels': [4000, 3800, 4500],
    'Platelet Labels': [2000, 2200, 3000]
}

label_df = pd.DataFrame(label_counts)
label_df.set_index('Dataset').plot(kind='bar', figsize=(10, 6))
plt.title('Comparison of Label Counts in BCCD, BCD, and TXL-PBC Datasets')
plt.ylabel('Number of Labels')
plt.show()
