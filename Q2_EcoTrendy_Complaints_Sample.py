import numpy as np
import pandas as pd
import datetime
np.random.seed(42)

channels = ["Email", "Phone", "Chat", "Social Media"]
channelProb = [0.1, 0.05, 0.45, 0.4]
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
complaint_testsProb = [0.05, 0.01, 0.02, 0.05, 0.02, 0.08, 0.08, 0.07, 0.06, 0.02, 0.05, 0.1, 0.1, 0.1, 0.19]
def generateData(n):
    ComplaintID = range(1, n+1)
    ComplaintDate = [];
    for i in ComplaintID:
        time_delta = datetime.date.today() - datetime.date(2023, 1, 1)
        random_days = np.random.randint(0, time_delta.days)
        ComplaintDate.append(datetime.date(2023, 1, 1) + datetime.timedelta(days=random_days))
    ComplaintText = np.random.choice(complaint_texts, n, complaint_testsProb)
    Category = []
    satisfactionScore = []
    resolutionTime = []
    for i in ComplaintText:
        if "arrived" in i or "wrong item" in i or "Tracking" in i or "damaged" in i:
            Category.append("Delivery Delay")
            satisfactionScore.append(np.random.choice([1, 2, 3], 1, p = [0.5, 0.3, 0.2])[0])
            resolutionTime.append(np.random.randint(2, 72, 1)[0])
        elif "Quality" in i or "quality" in i:
            Category.append("Product Quality")
            satisfactionScore.append(np.random.choice([1, 2, 3], 1, p = [0.2, 0.3, 0.5])[0])
            resolutionTime.append(np.random.randint(2, 48, 1)[0])
        elif "return" in i or "Return" in i:
            Category.append("Returns")
            satisfactionScore.append(np.random.choice([1, 2, 3], 1, p = [0.2, 0.4, 0.4])[0])
            resolutionTime.append(np.random.randint(24, 72, 1)[0])
        elif "Customer support" in i or "customer support" in i or "support agent" in i or "email" in i or "assistance" in i:
            Category.append("Customer Support")
            satisfactionScore.append(np.random.choice([1, 2, 3], 1, p = [0.3, 0.4, 0.3])[0])
            resolutionTime.append(np.random.randint(48, 96, 1)[0])
        elif "Payment" in i or "payment" in i or "refund" in i or "Refund" in i:
            Category.append("Payment")
            satisfactionScore.append(np.random.choice([1, 2, 3], 1, p = [0.4, 0.3, 0.3])[0])
            resolutionTime.append(np.random.randint(2, 24, 1)[0])
        else:
            Category.append(np.random.choice(categories, 1, p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5])[0])
            satisfactionScore.append(np.random.choice([1, 2, 3], 1, p = [0.3333, 0.3333, 0.3334])[0])
            resolutionTime.append(np.random.randint(2, 72, 1)[0])
            
    data = {
    "ComplaintID": range(1, n+1),
    "CustomerID": np.random.randint(1, 201, n),
    "ComplaintDate": ComplaintDate,
    "Channel": np.random.choice(channels, n, p = [0.1, 0.05, 0.45, 0.4]),
    "ComplaintText": ComplaintText,
    "Category": Category,   # np.random.choice(categories, n),
    "ResolutionTime": resolutionTime,  # hours
    "Resolved": np.random.choice(["Yes", "No"], n, p=[0.8,0.2]),
    "SatisfactionScore": satisfactionScore
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
