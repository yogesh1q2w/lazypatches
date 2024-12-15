from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class IncrementalBleuScorer:
    def __init__(self):
        self.scores = []
        self.total_score = 0
        self.count = 0
        self.smoothing_function = SmoothingFunction().method1

    def add_sample(self, reference, hypothesis):
        ref_tokens = [reference.split()]
        hyp_tokens = hypothesis.split() 

        score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=self.smoothing_function)

        self.scores.append(score)
        self.total_score += score
        self.count += 1

        return score, self.total_score / self.count

    def get_all_scores(self):
        return self.scores

    def get_average_score(self):
        return self.total_score / self.count if self.count > 0 else 0


test = IncrementalBleuScorer()
test.add_sample('A person ins walking through a door way and is holding a pillow. The person opens a closet door and puts the pillow in there. The person is also tidying up the closet. The person then looks in a mirror and stairs at themselves.', 'The video shows a young boy entering a room and walking towards a closet. He then opens the closet door and places a towel inside. After that, he walks towards a mirror and stands in front of it. The video ends with the boy walking away from the mirror.')
print(test.get_all_scores())