import numpy as np

def extract_tour_order(x, n):
    #print(x)
    tour = []

    for i in x:
        if i >= n:
            continue
        else:
            tour.append(i)

    return tour