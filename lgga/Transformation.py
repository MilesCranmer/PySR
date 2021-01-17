class Transformation:
    def __init__(self, index, name="Identity Transformation"):
        self.name = name
        self.index = index

    def transform(self, X):
        """
        Takes in a data point of shape (n, d) and returns an augmented data point based on the constraint
        """
        return X

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self)

    def get_params(self):
        raise NotImplementedError


class SymTransformation(Transformation):
    def __init__(self, x1=0, x2=1):
        """
        x1, x2 = indices of the variables which are symmetric
        """
        super().__init__(1, name=f"Symmetry Between Variable {x1} and {x2}")
        self.x1 = x1
        self.x2 = x2

    def transform(self, X):
        """
        """
        temp = X.copy()
        temp[:, self.x2] = X[:, self.x1].copy()
        temp[:, self.x1] = X[:, self.x2].copy()
        return temp

    def get_params(self):
        return [self.x1, self.x2]


class ZeroTransformation(Transformation):
    def __init__(self, inds=[0]):
        """
        inds is a list of indices to set to 0
        """
        super().__init__(2, name=f"Zero Constraint for Variables {inds}")
        self.inds = inds

    def transform(self, X):
        temp = X.copy()
        for ind in self.inds:
            temp[:, ind] = 0
        return temp

    def get_params(self):
        return list(self.inds)


class ValueTransformation(Transformation):
    def __init__(self, inds=[0]):
        """
        inds is list of indices to set to the same value as the first element in that list
        """
        super().__init__(3, name=f"Value Constraint for Variables {inds}")
        self.inds = inds

    def transform(self, X):
        temp = X.copy()
        val = temp[:, self.inds[0]]
        for ind in self.inds[1:]:
            temp[:, ind] = val
        return temp

    def get_params(self):
        return list(self.inds)