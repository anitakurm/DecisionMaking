import numpy as np

win = 1
loss = 0

outcomes = np.array([win,loss])

#actual probabilities of winning and losing
actual_prob_q1 = [0.1, 0.9]
actual_prob_q2 = [0.9, 0.1]

actual_probabilities = np.array([actual_prob_q1, actual_prob_q2])

optimal_choice= np.argmax([[prob[0]] for prob in actual_probabilities])


def simulate_agent(Q1, Q2, beta, alpha, trials):
    q_values = np.array([Q1, Q2])
    print(q_values)
    for trial in np.arange(trials):
        #make a choice
        exps = [np.exp(q/beta) for q in q_values]
        sum_of_exps = sum(exps)
        prob_of_i = np.array([exp_value/sum_of_exps for exp_value in exps])
        choice_made = prob_of_i.argmax()
        
        #get feedback: win or loss based on probabilities for the chosen option
        outcome = np.random.choice(outcomes, size = 1, p = actual_probabilities[choice_made])

        #update Q values
        prediction_error = outcome - q_values[choice_made]
        q_values[choice_made] = q_values[choice_made] + alpha * prediction_error
        
        #see how it updated
        print('Trial ' + str(trial+1) + ' ' + str(q_values))  
    print('Probability of choosing the optimal value is ' + str(q_values[optimal_choice]))


simulate_agent(0.99, 0.1, beta = 0.005, alpha = 0.2, trials= 50)




# #function for probability of making each choice
# def choose(q_values, beta):
#     exps = [np.exp(q/beta) for q in q_values]
#     sum_of_exps = sum(exps)
#     prob_of_i = np.array([exp_value/sum_of_exps for exp_value in exps])
#     return prob_of_i.argmax()


# #function for updating q values
# def update(q_values, alpha):
#     choice_made = choose(q_values, 2)
#     outcome = np.random.choice(outcomes, size = 1, p =actual_probabilities[choice_made])
#     prediction_error = outcome - q_values[choice_made]
#     q_values[choice_made] = q_values[choice_made] + alpha*prediction_error
#     return q_values

# #function for simulating the agent going through trials
# def simulate_agent(q_values, trials):
#     print(q_values)
#     for trial in np.arange(trials):
#         print(update(q_values, 0.1))
    
#     optimal_choice= np.argmax([[prob[0]] for prob in actual_probabilities])
#     print(q_values[optimal_choice])


# simulate_agent(Q_values, 100)

# # IT WORKS!