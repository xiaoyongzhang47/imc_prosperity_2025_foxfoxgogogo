import random

# ----------------------
# Game Parameters
# ----------------------
base_treasure = 10000
cost_paid = 50000

# Container attributes given as (multiplier, inhabitants)
containers = [
    {"multiplier": 10, "inhabitants": 1},   # Container 0
    {"multiplier": 80, "inhabitants": 6},   # Container 1
    {"multiplier": 37, "inhabitants": 3},   # Container 2
    {"multiplier": 17, "inhabitants": 1},   # Container 3
    {"multiplier": 90, "inhabitants": 10},  # Container 4
    {"multiplier": 20, "inhabitants": 2},   # Container 5
    {"multiplier": 31, "inhabitants": 2},   # Container 6
    {"multiplier": 50, "inhabitants": 4},   # Container 7
    {"multiplier": 73, "inhabitants": 4},   # Container 8
    {"multiplier": 89, "inhabitants": 8},   # Container 9
]

num_players = 4096

p = 1

#everyone get a free pick
strategies = []
for _ in range(num_players):
    free_choice = random.randrange(len(containers))
    if random.random() < p:
        # Choose a paid container that is different from the free pick.
        possible_paid_choices = [i for i in range(len(containers)) if i != free_choice]
        paid_choice = random.choice(possible_paid_choices)
    else:
        paid_choice = None
    strategies.append({"free": free_choice, "paid": paid_choice})

# ----------------------
# Count Function
# ----------------------
def compute_container_counts(strategies):

    counts = [0] * len(containers)
    for strat in strategies:
        counts[strat["free"]] += 1
        if strat["paid"] is not None:
            counts[strat["paid"]] += 1
    return counts

# ----------------------
# Payoff Function
# ----------------------
def container_payoff(i, count_i):
    total_possible_choices = num_players + sum(1 for strat in strategies if strat["paid"] is not None)
    m = containers[i]["multiplier"]
    h = containers[i]["inhabitants"]
    return base_treasure * m / (h + (count_i / total_possible_choices) * 100)

# ----------------------
# Best Response Dynamics
# ----------------------
def best_response_update(strategies, counts):


    change_occurred = False
    player_order = list(range(num_players))
    random.shuffle(player_order)
    
    for player in player_order:
        # --- FREE CHOICE UPDATE ---
        current_free = strategies[player]["free"]
        # Remove player's free pick temporarily.
        counts[current_free] -= 1
        
        best_free = current_free
        current_best_payoff_free = container_payoff(current_free, counts[current_free] + 1)
        
        for i in range(len(containers)):
            candidate_payoff = container_payoff(i, counts[i] + 1)
            if candidate_payoff > current_best_payoff_free:
                best_free = i
                current_best_payoff_free = candidate_payoff
        
        if best_free != current_free:
            strategies[player]["free"] = best_free
            change_occurred = True
        
        counts[strategies[player]["free"]] += 1

        # --- PAID CHOICE UPDATE ---
        current_paid = strategies[player]["paid"]

        if current_paid is not None:
            counts[current_paid] -= 1
        
        best_paid = None
        current_best_payoff_paid = 0
        
        # Evaluate each container as candidate if it isn't the free choice.
        for i in range(len(containers)):
            if i == strategies[player]["free"]:
                continue 

            candidate_payoff = container_payoff(i, counts[i] + 1) - cost_paid

            # print(candidate_payoff)

            if candidate_payoff > current_best_payoff_paid:
                best_paid = i
                current_best_payoff_paid = candidate_payoff
        
        if best_paid != current_paid:
            strategies[player]["paid"] = best_paid
            change_occurred = True
        
        # Add back the (possibly updated) paid pick.
        if strategies[player]["paid"] is not None:
            counts[strategies[player]["paid"]] += 1

    return change_occurred

# ----------------------
# Run the Simulation
# ----------------------
max_iterations = 10000
iteration = 0
verbose = False  # Change to True to see player move logs

counts = compute_container_counts(strategies)
while iteration < max_iterations:
    iteration += 1
    changed = best_response_update(strategies, counts)
    if not changed:
        print(f"Convergence reached after {iteration} iterations.")
        break

if iteration == max_iterations:
    print("Maximum iterations reached; equilibrium may be approximate.")

counts = compute_container_counts(strategies)

print("\nFinal container selection counts:")
for i, cnt in enumerate(counts):
    print(f"Container {i}: selected {cnt} times")

total_payoff_free = 0
total_payoff_paid = 0

# Recompute counts for payoff calculation.
final_counts = compute_container_counts(strategies)

for strat in strategies:
    free_c = strat["free"]
    payoff_free = container_payoff(free_c, final_counts[free_c])
    total_payoff_free += payoff_free
    if strat["paid"] is not None:
        payoff_paid = container_payoff(strat["paid"], final_counts[strat["paid"]]) - cost_paid
        total_payoff_paid += payoff_paid
    else:
        payoff_paid = 0  # no paid container yields no additional cost or reward

avg_free = total_payoff_free / num_players
avg_paid = total_payoff_paid / num_players
combined_avg = (total_payoff_free + total_payoff_paid) / num_players

print("\nAverage payoff for FREE pick: {:.2f}".format(avg_free))
print("Average payoff for PAID pick: {:.2f}".format(avg_paid))
print("Combined average payoff: {:.2f}".format(combined_avg))

final_pay_offs = []
for i in range(len(containers)):
    final_pay_offs.append(container_payoff(i, counts[i] + 1))

print(max(final_pay_offs))

print(final_pay_offs)

