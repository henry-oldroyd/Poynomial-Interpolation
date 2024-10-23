# import modules
import random
from math import log10
import numpy as np
import csv
from random import randint


def save_to_csv(data, filename):
    assert data

    # Extract keys from the first dictionary to use as column headers
    keys = data[0].keys()

    # Open the file for writing
    with open(filename, 'w', newline='') as output_file:
        # Create a DictWriter object
        dict_writer = csv.DictWriter(output_file, fieldnames=keys, delimiter=';')

        # Write the column headers
        dict_writer.writeheader()

        # Write the data rows
        dict_writer.writerows(data)

# randomly generate some problem constraints
domain = [-5, 5]
num_coefficients = 6
max_training_iterations = 10**6


# protect_C = True
protect_C = False
learning_rate = 10**-5
# learning_rate = 10**-8


stop_on_threshold = True
# stop_threshold = 10**-4 * learning_rate
stop_threshold = 10**-20

# learning_rate_decrease = 0.99
learning_rate_decrease = 1

assert 0 < learning_rate_decrease <= 1
# 8 increases equivolent to a decrease
learning_rage_increase = (learning_rate_decrease)**(-1/10)

assert 1 <= learning_rage_increase < 2

num_points_total = 1000
minibatch_size = 100
assert num_points_total % minibatch_size == 0, "Minibatch size must to a factor of num_points_total"
test_proportion = 0.2

number_points_train = int((1-test_proportion) * num_points_total)
number_points_test = int(test_proportion * num_points_total)

# generate_coefficients 
polynomial = [random.uniform(-1, 1) for _ in range(num_coefficients)]

print(f"Polynomial of degree {len(polynomial)-1} randomly generated:")
print(" + ".join(f"{round(polynomial[i], 4)}*x^{i}" for i in range(len(polynomial))))
print("This is represented by this array of coefficients:")
print(polynomial)

def evaluate_polynomial(x, C):
    total = 0
    # reverse as first coefficient should correspond to greatest degree not least
    for coefficient in C[::-1]:
        total = x*total + coefficient
    return total

# generate points
x_points_train = [random.uniform(*domain) for _ in range(number_points_train)]
y_points_train = [evaluate_polynomial(x_point, polynomial) for x_point in x_points_train]

# add some noise to y points for training only
y_points_train = [y_point + random.uniform(-5, 5) for y_point in y_points_train]


x_points_test = [random.uniform(*domain) for _ in range(number_points_test)]
y_points_test = [evaluate_polynomial(x_point, polynomial) for x_point in x_points_test]

# construct vectors and matrix
n = number_points_train
m = num_coefficients
a = learning_rate

X = np.array(x_points_train)
Y = np.array(y_points_train)

# C = [1 for _ in range(m)]
C = [random.uniform(-1, 1) for _ in range(m)]

# construct M matrix
M_columns = []
new_column = [1 for _ in range(n)]
M_columns.append(new_column)

for _ in range(1, m):
    new_column = [x*e for e, x in zip(new_column, X)]
    M_columns.append(new_column)

M = []
for row in range(n):
    M.append([column[row] for column in M_columns])


M = np.array(M)

def generate_minibatch_matricies():
    indexes = np.arange(0, number_points_train)
    np.random.shuffle(indexes)

    minibatch_indexes_matricies = np.split(indexes, minibatch_size)
    for minibatch_indexes_matrix in minibatch_indexes_matricies:
        yield {
            # "X": X[minibatch_indexes_matrix],
            "Y": Y[minibatch_indexes_matrix],
            "M": M[minibatch_indexes_matrix]    
        }

# train model
print("Training")
R = M @ C
old_MSE = (1/n) * ((Y-R) @ (Y-R).T)
old_C = C

def log_abs(x):
    if x == 0:
        return None
    else:
        return log10(x if x >= 0 else -x)


iterations_csv_data = []
total_csv_entries = 200
csv_sample_rate = max_training_iterations // total_csv_entries

MSE_changes_to_record = 1000
threshold_checks_per_window = 10
threshold_check_rate = MSE_changes_to_record // threshold_checks_per_window
MSE_changes = [None for _ in range(MSE_changes_to_record)]

epochs = max_training_iterations//(number_points_train//minibatch_size)

try:
    for i in range(epochs):
        
        # take a random batch form m

        # let MM be minibatch Matrix
        for minibach_data in generate_minibatch_matricies():
            MM = minibach_data["M"]
            MY = minibach_data["Y"]
            
            MR = MM @ C
            unscaled_derivative = (2/n) * (MM.T @ (MY - (MM @ C)))
            C = C + a * unscaled_derivative

            new_MSE = (1/n) * ((MY-MR) @ (MY-MR).T)

            # print(f"new_c")
            # print(f"{new_MSE} - {old_MSE}")
            # print(f"{new_MSE - old_MSE}")

            # # ignore on first iteration when mse is None
            # if not old_MSE:
            #     continue
                
            MSE_change = new_MSE - old_MSE
            MSE_changes[i % MSE_changes_to_record] = MSE_change
            if MSE_change > 0:
                # print(f"MSE change was positive: old_MSE={old_MSE:.6f}, new_MSE={new_MSE:.6f}, MSE_change={MSE_change:.6f}")
                # a /= 2
                a *= learning_rate_decrease
                if protect_C:
                    C = old_C
                    new_MSE = old_MSE

                # CHANGE HERE TO SHOW FAIL CASE
                if log_abs(a) < -20:
                    raise Exception(f"a diverging: log_abs(a)={log_abs(a)}  a={a:.10f}")
                # print(f"Old MSE: {old_MSE:.10f}, New MSE: {new_MSE:.10f}")
                # print(f"Iteration {i} saw a: MSE increase")
                # print(f"log_abs(ΔMSE)={log_abs(MSE_change)}; ΔMSE={MSE_change}")
                # print(f"log_abs(a)={log_abs(a)}; a={a:.10f}")
                # print(f"MSE={old_MSE:.10f}; C=[{', '.join(f'{c:.6f}' for c in C)}]")

                if i % (csv_sample_rate) == 0:
                    iterations_csv_data.append({
                        "MSE_change_made": not protect_C,
                        "iteration": i,
                        "MSE": old_MSE,
                        "dMSE/dC": ",".join(str(-e) for e in unscaled_derivative),
                        "MSE_change": MSE_change,
                        "a_multiplier_used": learning_rate_decrease,
                        "a": a,
                        "C": ",".join(str(c) for c in C),
                        "log_iteration": log_abs(i),
                        "log_MSE": log_abs(old_MSE),
                        "log_dMSE/dC": ",".join(str(log_abs(e)) for e in unscaled_derivative),
                        "log_MSE_change": log_abs(MSE_change),
                        "log_a": log_abs(a),
                    })


                
                continue
            else:
                # only change old_MSE from none when MSE_change < 0
                old_MSE = new_MSE
                # a *= 1.000_001
                a *= learning_rage_increase
                # CHANGE HERE TO SHOW FAIL CASE
                # a *= 1.000_01

                if i % (csv_sample_rate) == 0:
                    iterations_csv_data.append({
                        "MSE_change_made": "True",
                        "iteration": i,
                        "MSE": old_MSE,
                        "dMSE/dC": ",".join(str(-e) for e in unscaled_derivative),
                        "MSE_change": MSE_change,
                        "a_multiplier_used": learning_rage_increase,
                        "a": a,
                        "C": ",".join(str(c) for c in C),
                        "log_iteration": log_abs(i),
                        "log_MSE": log_abs(old_MSE),
                        "log_dMSE/dC": ",".join(str(log_abs(e)) for e in unscaled_derivative),
                        "log_MSE_change": log_abs(MSE_change),
                        "log_a": log_abs(a),
                    })
                
        if i % (max_training_iterations / 100) == 0:
            # print(f"Training iteration {i}: log_abs(-ΔMSE)={log_abs(-MSE_change):.5f}; log_abs(a)={log_abs(a):.5f}")
            print(f"Training iteration {i}: log_abs(-ΔMSE)={log_abs(-MSE_change)}; log_abs(a)={log_abs(a)}")
            print(f"Training iteration {i}: MSE={new_MSE:.10f}; C=[{', '.join(f'{c:.6f}' for c in C)}]")

            # print(f"(-MSE_change) < stop_threshold")
            # print(f"{(-MSE_change)} < {stop_threshold}")
            # print(f"{(-MSE_change) < stop_threshold}")

        
        # print(MSE_changes.count(None))

        # if MSE_changes[MSE_changes_to_record - 1] is None:
        if i <= MSE_changes_to_record:
        # if MSE_changes.count(None):
            continue

        # print(MSE_changes.count(None))
        # print(MSE_changes)

        # assert not any(e is None for e in MSE_changes)

        # print(f"(i % threshold_check_rate != 0)")
        # print(f"({i} % {threshold_check_rate} != 0)")
        # print(f"{(i % threshold_check_rate != 0)}")
        if (i % threshold_check_rate != 0):
            continue
        
        mean_MSE_change = sum(MSE_changes) / MSE_changes_to_record
        
        # print(f"(-mean_MSE_change) < stop_threshold")
        # print(f"{(-mean_MSE_change)} < {stop_threshold}")
        # print(f"{(-mean_MSE_change) < stop_threshold}")

        def absolute(x): return x if x >= 0 else -x
        
        if absolute(mean_MSE_change) < stop_threshold and stop_on_threshold:
            print(f"Stop threshold reached: log_abs(-meanΔMSE)={log_abs(-mean_MSE_change):.5f}; ΔMSE={mean_MSE_change:.10f} Ending training.")
            break


except Exception as e:
    raise e
finally:
    save_to_csv(
        data=iterations_csv_data,
        filename="training_data.csv"
    )

# test model
mean_error_squared = (1/number_points_test) * sum(
    (evaluate_polynomial(x_point, C) - y_point)**2 for x_point, y_point in zip(x_points_test, y_points_test)
)
print(f"Polynomial test data predictions MSE:   {mean_error_squared:.4f}")

# print("Data points for training:")
# for x, y in zip(x_points_train, y_points_train):
#     print(f"({x:.4f}, {y:.4f})")

# print("Data points for testing:")
# for x, y in zip(x_points_test, y_points_test):
#     print(f"({x:.4f}, {y:.4f})")

print("Selected coefficients")
for c in polynomial:
    print(f"{c:.6f}")

print("learned polynomial")
for c in C:
    print(f"{c:.6f}")
