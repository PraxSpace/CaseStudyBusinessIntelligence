import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = [
    {
        'Tool': 'RapidMiner',
        'Cost': 5000,
        'FreeTier': 'Yes',
        'OpenSource': 'No',
        'GUI': 'Yes',
        'Automation': 'Yes',
        'Integration': 5,
        'Usability': 4,
        'BestFor': 'SME'
    },
    {
        'Tool': 'KNIME',
        'Cost': 0,
        'FreeTier': 'Yes',
        'OpenSource': 'Yes',
        'GUI': 'Yes',
        'Automation': 'Yes',
        'Integration': 4,
        'Usability': 4,
        'BestFor': 'SME'
    },
    {
        'Tool': 'IBM SPSS Modeler',
        'Cost': 10000,
        'FreeTier': 'No',
        'OpenSource': 'No',
        'GUI': 'Yes',
        'Automation': 'Yes',
        'Integration': 5,
        'Usability': 3,
        'BestFor': 'Enterprise'
    },
    {
        'Tool': 'SAS',
        'Cost': 15000,
        'FreeTier': 'No',
        'OpenSource': 'No',
        'GUI': 'Yes',
        'Automation': 'Yes',
        'Integration': 5,
        'Usability': 3,
        'BestFor': 'Enterprise'
    },
    {
        'Tool': 'Dataiku',
        'Cost': 9000,
        'FreeTier': 'Yes',
        'OpenSource': 'No',
        'GUI': 'Yes',
        'Automation': 'Yes',
        'Integration': 4,
        'Usability': 5,
        'BestFor': 'Enterprise'
    },
    {
        'Tool': 'Orange',
        'Cost': 0,
        'FreeTier': 'Yes',
        'OpenSource': 'Yes',
        'GUI': 'Yes',
        'Automation': 'No',
        'Integration': 3,
        'Usability': 4,
        'BestFor': 'Academia'
    }
]

df_tools = pd.DataFrame(data)
df_tools.to_csv('DataMining_Tools_Comparison.csv', index=False)
print(df_tools)


df = pd.read_csv('DataMining_Tools_Comparison.csv')

# Sort tools by Integration (descending), then Usability
sorted_df = df.sort_values(['Integration', 'Usability'], ascending=[False, False])

print('Top Data Mining Tools by Integration & Usability:')
print(sorted_df[['Tool', 'Cost', 'Integration', 'Usability', 'FreeTier', 'OpenSource', 'BestFor']].head(3))

# Filter for free/open-source tools
free_open = df[(df['FreeTier'] == 'Yes') & (df['OpenSource'] == 'Yes')]
print('\\nFree/Open Source Solutions:')
print(free_open[['Tool', 'Integration', 'Usability', 'BestFor']])

# Convert Yes/No to 1/0 for binary features
binary_cols = ['FreeTier', 'OpenSource', 'GUI', 'Automation']
for col in binary_cols:
    df[col] = df[col].map({'Yes':1, 'No':0})

plt.figure(figsize=(8, 4))
sns.heatmap(df.set_index('Tool')[binary_cols], annot=True, cmap='Blues', cbar=False)
plt.title('Feature Support by Data Mining Tool')
plt.show()
