import numpy as np

# Initial bids for each placement
bids = np.array([50, 50, 50])

# Total budget
budget = 9000

# Learning rate
alpha = 0.1

# Number of iterations
num_iterations = 100

# Data for each placement
data = {
    'A': {'impressions': 96034, 'clicks': 2682, 'spend': 9134, 'orders': 369, 'sales': 18944},
    'B': {'impressions': 518530, 'clicks': 245, 'spend': 479, 'orders': 9, 'sales': 361},
    'C': {'impressions': 274591, 'clicks': 1109, 'spend': 2057, 'orders': 126, 'sales': 6223},
}

# Function to calculate the ROAS given the bids and data
def calc_roas(bids, data):
    roas = {}
    for placement, values in data.items():
        spend = values['spend']
        sales = values['sales']
        roas[placement] = sales/spend
    return roas

# Gradient descent loop
for i in range(num_iterations):
    # Calculate the gradient of ROAS with respect to the bids
    gradient = {}
    for placement, values in data.items():
        impressions = values['impressions']
        clicks = values['clicks']
        ctr = clicks / impressions
        bid = bids[placement]
        gradient[placement] = -ctr / bid
    # Update the bids using learning rate
    bids = bids - alpha * gradient
    # Keep bids within budget
    bids = np.clip(bids, 0, budget)
    # Calculate the new ROAS
    roas = calc_roas(bids, data)
    # Print the current bids, ROAS, and gradient for each placement
    print("Iteration: {}, Bids: {}, ROAS: {}".format(
