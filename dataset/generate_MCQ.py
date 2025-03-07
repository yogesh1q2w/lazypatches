import random

def genQ(T_actions, F_actions):
    Qs = []
    As = []
    Q = 'Which of these action occur in the video? Please respond with the correct option only.'
    num_F = len(F_actions)
    num_FC = 3
    for T in T_actions:
        Cs = []
        Cs.append(T)
        FC = random.sample(range(num_F), num_FC)
        Cs.extend([F_actions[i] for i in FC])
        random.shuffle(Cs)
        tc_index = Cs.index(T)
        Cs_with_numbers = [f"{i + 1}. {option}" for i, option in enumerate(Cs)]
        # print(Cs)
        Qs.append('\n'.join([Q] + Cs_with_numbers))
        As.append(Cs_with_numbers[tc_index])
    return Qs, As
    
T_actions = ['Sitting at a table', 'Tidying up a table', 'Washing a table', 'Working at a table']
F_actions = ['Taking a bag from somewhere', 'Closing a book', 'Taking a towel/s from somewhere', 'Holding a laptop', 'Standing on a chair', 'Taking a blanket from somewhere', 'Smiling in a mirror', 'Taking paper/notebook from somewhere', 'Grasping onto a doorknob']

Qs, As = genQ(T_actions, F_actions)
print(len(Qs), Qs)