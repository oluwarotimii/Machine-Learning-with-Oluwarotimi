import matplotlib.pyplot as plt


Sports =  ['Football', 'Rugby', 'Baseball', 'Basketball', 'Cricket']
Players = [100, 30, 26, 50, 10]

plt.bar(Sports, Players, color='blue', alpha = 0.8)
plt.title('Sports Distribution')
plt.xlabel('Sports')
plt.ylabel('Number of Player')
plt.show()


# Pie chart
plt.pie(Players, labels=Sports, autopct='%1.1f%%')
plt.title('Pie Chart Distribution of Sport Players')
plt.axis('equal')
plt.show()