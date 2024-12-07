import numpy as np

class IncrementalMCQAcc:
    def __init__(self):
        self.count = 0
        self.map = []

    def add_sample(self, answer, key):
        if answer == key:
            self.map.append(1)
        else:
            self.map.append(0)
        current_acc = np.mean(np.array(self.map))
        self.count += 1

        return current_acc

    def get_all_acc(self):
        return np.mean(np.array(self.map))

    def get_num_test(self):
        return self.count
    
acc = IncrementalMCQAcc()
current = acc.add_sample("2. open the door", "2. open the door")
current = acc.add_sample("2. open the door", "3. open the door")
current = acc.add_sample("2. open the door", "2. open the door")
print(current)