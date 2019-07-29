class Env:
    def __init__(self):
        self.do_jump = 0
        self.do_nothing = 1

        self.n_objects = n_objects
        self.n_colours = n_colours
        self.n_first_states=15
        self.n_sec_states=760
        self.n_states=self.n_first_states*self.n_sec_states
        self.n_actions=2

        self.starting_state=(  1, 390)
        self.ending_state=  (276, 485)


    def feature_vector(self, i, feature_map="ident"):
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)
