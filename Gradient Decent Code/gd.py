import numpy as np

def gradent_decent(x,y):
    mcurr = bcurr = 0
    iterations = 1000
    n= len(x)
    # Learning arate Trail and Error So It Wiil be Applied to Reduce the Cost
    learning_rate =0.08
    for i in range(iterations):
        yp = mcurr * x + bcurr
        # /y-yp Delta ak Vector de ga
    #     Aus peList Comprehension ka use kar ke Use kar saktai;Har Aik Delta value ko Utha rhe aue And Squared it and Sum
        cost = (1/n) * sum([val**2 for val in (y-yp)])
    #     to find the Partial Derivative of m
        md = -(2/n) * sum(x*(y-yp))
        bd = -(2/n) * sum(y-yp)
    #     Step Equation
        mcurr = mcurr - learning_rate * md
        bcurr = bcurr - learning_rate * bd
        print("m {} b {} cost {} iteration {}".format(mcurr, bcurr,cost,i))
    pass
x = np.array([1,2,3,4,5,])
y = np.array([5,7,9,11,13])
gradent_decent(x,y)