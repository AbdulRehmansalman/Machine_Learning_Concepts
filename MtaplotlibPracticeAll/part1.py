import matplotlib.pyplot as plt

x=[1,2,3,4,5,6,7]
y=[50,51,52,48,47,49,46]

plt.xlabel('Temperature')
plt.ylabel('Weather')
plt.title('Weather')

# *creates a plot with blue (b) plus markers (+) connected by dashed lines (--). Here’s a breakdown of the format string:
# plt.plot(x,y,'b+--')

# With Only Markers OnlY
# plt.plot(x,y,color='blue',marker='+',linestyle='' ,markersize=20)
# Alpha Will set The Transaparenct Level to 50 %
plt.plot(x, y, 'g<', alpha=0.5)
plt.show()