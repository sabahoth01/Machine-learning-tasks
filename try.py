# import numpy as np
# import matplotlib.pyplot as plt



# # def y_function(x):
# #     return x**2

# # def der_y_function(x):
# #     return 2*x

# # cur_pos = (90,y_function(90))
# # x= np.arange(-100,100,0.1)
# # y= y_function(x)

# # lr=0.01

# # for _ in range(1000):
# #     new_x = cur_pos[0]  - lr * der_y_function(cur_pos[0])
# #     new_y = y_function(new_x)
# #     cur_pos = (new_x,new_y)
# #     # print(x)
# #     plt.plot(x, y)
# #     plt.scatter(cur_pos[0], cur_pos[1], color='red')
# #     plt.pause(0.001)
# #     plt.clf()

# import numpy as np

# def gradient_descent(X, y, alpha=0.01, lambda_=0.1, num_iterations=1000):
#     """
#     Обучает модель линейной регрессии методом градиентного спуска.

#     Параметры:
#     X (numpy.array): матрица признаков
#     y (numpy.array): вектор целевых переменных
#     alpha (float): шаг обучения (по умолчанию 0.01)
#     lambda_ (float): коэффициент регуляризации (по умолчанию 0.1)
#     num_iterations (int): количество итераций (по умолчанию 1000)

#     Возвращает:
#     w (numpy.array): вектор весов
#     """
#     n, d = X.shape
#     w = np.zeros(d)

#     for _ in range(num_iterations):
#         # Вычисление градиента
#         gradient = (2 / n) * X.T.dot(X.dot(w) - y) + 2 * lambda_ * w

#         # Обновление вектора весов
#         w -= alpha * gradient

#     return w


# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_predicted):
	
	# Calculating the loss or cost
	cost = np.sum((y_true-y_predicted)**2) / len(y_true)
	return cost

# Gradient Descent Function
# Here iterations, learning_rate, stopping_threshold
# are hyperparameters that can be tuned
def gradient_descent(x, y, iterations = 1000, learning_rate = 0.0001, 
					stopping_threshold = 1e-6):
	
	# Initializing weight, bias, learning rate and iterations
	current_weight = 0.1
	current_bias = 0.01
	iterations = iterations
	learning_rate = learning_rate
	n = float(len(x))
	
	costs = []
	weights = []
	previous_cost = None
	
	# Estimation of optimal parameters 
	for i in range(iterations):
		
		# Making predictions
		y_predicted = (current_weight * x) + current_bias
		
		# Calculating the current cost
		current_cost = mean_squared_error(y, y_predicted)

		# If the change in cost is less than or equal to 
		# stopping_threshold we stop the gradient descent
		if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
			break
		
		previous_cost = current_cost

		costs.append(current_cost)
		weights.append(current_weight)
		
		# Calculating the gradients
		weight_derivative = -(2/n) * sum(x * (y-y_predicted))
		bias_derivative = -(2/n) * sum(y-y_predicted)
		
		# Updating weights and bias
		current_weight = current_weight - (learning_rate * weight_derivative)
		current_bias = current_bias - (learning_rate * bias_derivative)
				
		# Printing the parameters for each 1000th iteration
		print(f"Iteration {i+1}: Cost {current_cost}, Weight \
		{current_weight}, Bias {current_bias}")
	
	
	# Visualizing the weights and cost at for all iterations
	plt.figure(figsize = (8,6))
	plt.plot(weights, costs)
	plt.scatter(weights, costs, marker='o', color='red')
	plt.title("Cost vs Weights")
	plt.ylabel("Cost")
	plt.xlabel("Weight")
	plt.show()
	
	return current_weight, current_bias


def main():
	
	# Data
	
	X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
		55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
		45.41973014, 54.35163488, 44.1640495 , 58.16847072, 56.72720806,
		48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
	Y = np.array([31.70700585, 68.77759598, 62.5623823 , 71.54663223, 87.23092513,
		78.21151827, 79.64197305, 59.17148932, 75.3312423 , 71.30087989,
		55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
		60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])

	# Estimating weight and bias using gradient descent
	estimated_weight, estimated_bias = gradient_descent(X, Y, iterations=2000)
	print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {estimated_bias}")

	# Making predictions using estimated parameters
	Y_pred = estimated_weight*X + estimated_bias

	# Plotting the regression line
	plt.figure(figsize = (8,6))
	plt.scatter(X, Y, marker='o', color='red')
	plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue',markerfacecolor='red',
			markersize=10,linestyle='dashed')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.show()

	
if __name__=="__main__":
	main()
