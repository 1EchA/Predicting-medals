import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

gold_medals = {
    'Gender_Ratio': 7.55,
    'Is_Host': 1.20,
    'Total_Events': 5.32,
    'Total_Participants': 67.36,
    'Historical_Medals': 15.21,
    'Olympic_Events': 3.36
}

total_medals = {
    'Total_Events': 0.0446,
    'Gender_Ratio': 0.0762,
    'Is_Host': 0.0053,
    'Historical_Medals': 0.1278,
    'Total_Participants': 0.7061,
    'Olympic_Events': 0.0400
}

colors = {
    'Gender_Ratio': 'skyblue',
    'Is_Host': 'lightgreen',
    'Total_Events': 'lightcoral',
    'Total_Participants': 'lightyellow',
    'Historical_Medals': 'lightgrey',
    'Olympic_Events': 'lightpink'
}

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
labels = gold_medals.keys()
sizes = gold_medals.values()
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=[colors[label] for label in labels])
plt.title('Gold-Medals Evaluation Results')

plt.subplot(1, 2, 2)
labels = total_medals.keys()
sizes = total_medals.values()
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=[colors[label] for label in labels])
plt.title('Total-Medals Evaluation Results')

plt.tight_layout()
plt.show()