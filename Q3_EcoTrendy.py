import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('Q1_EcoTrendy_Churn_Sample.csv')

# Example: Define potential target for upsell/cross-sell
# Criteria for high potential (you can change thresholds)
df['UpsellTarget'] = (
    (df['Loyalty'] == 'Yes') &
    (df['Engagement'] == 'High') &
    (df['NumPurchases'] >= df['NumPurchases'].median()) &
    (df['SpecialOffer'] == 'Yes')
)

# Segment analysis by Engagement and Loyalty
segment_counts = df.groupby(['Engagement', 'Loyalty'])['UpsellTarget'].sum().unstack(fill_value=0)

# Plot segment by Engagement/Loyalty
segment_counts.plot(kind='bar', stacked=True, color=['skyblue', 'coral'])
plt.title('High-potential Upsell/Cross-sell Segments by Engagement and Loyalty')
plt.ylabel('Number of High-potential Customers')
plt.show()

# Analyze feature importance: who accepted special offers in the past?
offer_accept = df[df['SpecialOffer'] == 'Yes']
feature_means = offer_accept.groupby('Engagement').agg({'NumPurchases':'mean','AvgOrderValue':'mean'})

# Visualize offer acceptance by other key features
sns.boxplot(x='Engagement', y='AvgOrderValue', data=offer_accept)
plt.title('Average Order Value among Offer-accepting Customers by Engagement')
plt.show()

# Top recommendations for targeting
top_segments = df[df['UpsellTarget']].groupby(['PurchaseFreq', 'Engagement']).size().sort_values(ascending=False)
print('Top segments for upsell targeting by Purchase Frequency and Engagement:')
print(top_segments)

# Export list for further marketing action
df[df['UpsellTarget']][['CustomerID','Loyalty','Engagement','NumPurchases','PurchaseFreq']].to_csv('Upsell_Target_Customers.csv', index=False)
