import numpy as np
import pandas as pd
import datetime
np.random.seed(42)

channels = ["Email", "Phone", "Chat", "Social Media"]
channelProb = [0.2, 0.1, 0.4, 0.4]
categories = ["Delivery Delay", "Product Quality", "Returns", "Customer Support", "Payment", "Other"]
complaint_texts = [
    "My order hasn't arrived on time.",
    "Received the wrong item.",
    "The product quality is poor.",
    "Couldn't get through to customer support.",
    "I want to return my order but the process is difficult.",
    "Payment did not go through.",
    "Refund has not been processed.",
    "The support agent was rude.",
    "Website is not user-friendly.",
    "Tracking number doesn't work.",
    "Repeated issues with the same problem.",
    "No response to my email.",
    "I was overcharged.",
    "Package arrived damaged.",
    "Long waiting time for assistance."
]
def generateData(n):
    ComplaintID = range(1, n+1)
    ComplaintDate = [];
    for i in ComplaintID:
        time_delta = datetime.date.today() - datetime.date(2023, 1, 1)
        random_days = np.random.randint(0, time_delta.days)
        ComplaintDate.append(datetime.date(2023, 1, 1) + datetime.timedelta(days=random_days))
    data = {
    "ComplaintID": range(1, n+1),
    "CustomerID": np.random.randint(1, 201, n),
    "ComplaintDate": ComplaintDate,
    "Channel": np.random.choice(channels, n, channelProb),
    "ComplaintText": np.random.choice(complaint_texts, n),
    "Category": np.random.choice(categories, n),
    "ResolutionTime": np.random.randint(2, 72, n),  # hours
    "Resolved": np.random.choice(["Yes", "No"], n, p=[0.8,0.2]),
    "SatisfactionScore": np.random.choice([1,2,3], n, p=[0.6,0.35,0.05])
    }
    df = pd.DataFrame(data)
    df.to_csv("Q2_EcoTrendy_Complaints_Sample.csv", index=False)
    print("File Created Successfully!")
generateData(10000) # Number of complaints


import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. Load Data
df = pd.read_csv('Q2_EcoTrendy_Complaints_Sample.csv')
print(df.head())

# 2. Count by category and channel
cat_counts = df['Category'].value_counts()
channel_counts = df['Channel'].value_counts()
print(cat_counts)
print(channel_counts)

# 3. Word Cloud for Complaint Text
all_text = " ".join(df['ComplaintText'])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Frequent Words in Complaint Texts')
plt.show()

# 4. Bar plot of complaints by category
cat_counts.plot(kind='bar', color='coral')
plt.ylabel('Number of Complaints')
plt.title('Complaints by Category')
plt.show()

# 5. Avg Resolution Time by Category
resolution_by_cat = df.groupby('Category')['ResolutionTime'].mean().sort_values(ascending=False)
resolution_by_cat.plot(kind='bar', color='skyblue')
plt.ylabel('Avg Resolution Time (hrs)')
plt.title('Avg Resolution Time by Category')
plt.show()

# 6. Satisfaction by Category
satisfaction_by_cat = df.groupby('Category')['SatisfactionScore'].mean().sort_values()
satisfaction_by_cat.plot(kind='bar', color='green')
plt.ylabel('Avg Satisfaction (1 = worst, 3 = better)')
plt.title('Customer Satisfaction by Complaint Category')
plt.show()
