import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = [
    ['E001', 'M', 34, 123, 'Normal', 350],
    ['E002', 'F', 40, 114, 'Overweight', 450],
    ['E003', 'F', 37, 135, 'Obesity', 169],
    ['E004', 'M', 30, 139, 'Underweight', 189],
    ['E005', 'F', 44, 117, 'Underweight', 183],
    ['E006', 'M', 36, 121, 'Normal', 80],
    ['E007', 'M', 32, 133, 'Obesity', 166],
    ['E008', 'F', 26, 140, 'Normal', 120],
    ['E009', 'M', 32, 133, 'Normal', 75],
    ['E010', 'M', 36, 133, 'Underweight', 40],
]

df = pd.DataFrame(data, columns=['EMPID', 'Gender', 'Age', 'Sales', 'BMI', 'Income'])
labels = [f'Emp {i}' for i in range(1, len(df) + 1)]

print("Histogram")
df.hist()
plt.show()

df.plot.bar()
plt.bar(df['Age'], df['Sales'])
plt.xlabel("Age")
plt.ylabel("Sales")
print("Bar chart")
plt.show()

df.plot.box()
plt.boxplot(df['Income'])
print("Box Plot chart")
plt.show()

plt.pie(df['Age'], labels=labels, autopct='%1.1f%%', shadow=True)
plt.title('Age Distribution')
plt.show()

plt.pie(df['Income'], labels=labels, autopct='%1.1f%%', shadow=True)
plt.title('Income Distribution')
plt.show()

plt.pie(df['Sales'], labels=labels, autopct='%1.1f%%', shadow=True)
plt.title('Sales Distribution')
plt.show()

plt.scatter(df['Income'], df['Sales'])
plt.title('Income vs Sales')
plt.show()

plt.scatter(df['Sales'], df['Age'])
print("Scatter plot")
plt.title('Sales vs Age')
plt.show()