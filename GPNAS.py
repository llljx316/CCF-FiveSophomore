import numpy as np
import copy

__all__ = ["GPNAS"]

class GPNAS(object):

    _estimator_type = "regressor"

    def __init__(self, cov_w = None, w = None, c_flag=2, m_flag=2, hp_mat = 0.0000001, hp_cov = 0.01, icov = 1):
        self.hp_mat = hp_mat
        self.hp_cov = hp_cov
        self.cov_w = cov_w
        self.w = w 
        self.c_flag = c_flag
        self.m_flag = m_flag
        self.icov = icov        # if we use initial cov as prior

    # def __init__(self, c_flag=2, m_flag=2):

    #     self.hp_mat = 0.0000001
    #     self.hp_cov = 0.01
    #     self.cov_w = None
    #     self.w = None
    #     self.c_flag = c_flag
    #     self.m_flag = m_flag

    def _get_corelation(self, mat1, mat2):
        """
        give two typical kernel function

        Auto kernel hyperparameters estimation to be updated
        """

        mat_diff = abs(mat1 - mat2)

        if self.c_flag == 1:

            return 0.5 * np.exp(-np.dot(mat_diff, mat_diff) / 16)

        elif self.c_flag == 2:

            return 1 * np.exp(-np.sqrt(np.dot(mat_diff, mat_diff)) / 12)

    def _preprocess_X(self, X):
        """
        preprocess of input feature/ tokens of architecture
        more complicated preprocess can be added such as nonlineaer transformation
        """

        X = X.tolist()
        p_X = copy.deepcopy(X)

        for feature in p_X:
            feature.append(1)

        return p_X

    def _get_cor_mat(self, X):
        """
        get kernel matrix
        """
        X = np.array(X)
        l = X.shape[0]
        cor_mat = []

        for c_idx in range(l):
            col = []
            c_mat = X[c_idx].copy()

            for r_idx in range(l):
                r_mat = X[r_idx].copy()
                temp_cor = self._get_corelation(c_mat, r_mat)
                col.append(temp_cor)
            cor_mat.append(col)

        return np.mat(cor_mat)

    def _get_cor_mat_joint(self, X, X_train):
        """
        get kernel matrix
        """
        X = np.array(X)
        X_train = np.array(X_train)
        l_c = X.shape[0]
        l_r = X_train.shape[0]
        cor_mat = []

        for c_idx in range(l_c):
            col = []
            c_mat = X[c_idx].copy()

            for r_idx in range(l_r):
                r_mat = X_train[r_idx].copy()
                temp_cor = self._get_corelation(c_mat, r_mat)
                col.append(temp_cor)
            cor_mat.append(col)

        return np.mat(cor_mat)

    def get_predict(self, X):
        """
        get the prediction of network architecture X
        """
        X = self._preprocess_X(X)
        X = np.mat(X)

        return X * self.w

    def get_predict_jiont(self, X, X_train, Y_train):
        """
        get the prediction of network architecture X based on X_train and Y_train
        """
        X = np.mat(X)
        X_train = np.mat(X_train)
        Y_train = np.mat(Y_train)
        m_X = self.get_predict(X)
        m_X_train = self.get_predict(X_train)
        mat_train = self._get_cor_mat(X_train)
        mat_joint = self._get_cor_mat_joint(X, X_train)

        return m_X + mat_joint * np.linalg.inv(mat_train + self.hp_mat * np.eye(
            X_train.shape[0])) * (Y_train.T - m_X_train)

    def get_initial_mean(self, X, Y):
        """
        get initial mean of w
        """

        X = self._preprocess_X(X)
        X = np.mat(X)
        Y = np.mat(Y)
        self.w = np.linalg.inv(X.T * X + self.hp_mat * np.eye(X.shape[
                                                                  1])) * X.T * Y.T

        return self.w

    def get_initial_cov(self, X):
        """
        get initial coviarnce matrix of w
        """

        X = self._preprocess_X(X)
        X = np.mat(X)
        # self.cov_w = self.hp_cov * np.eye(X.shape[1])

        # 改动部分
        if self.icov == 1: # use inv(X.T*X) as initial covariance
            self.cov_w = self.hp_cov * np.linalg.inv(X.T * X)
        elif self.icov == 0: # use identity matrix as initial covariance
            self.cov_w = self.hp_cov * np.eye(X.shape[1])

        return self.cov_w

    def get_posterior_mean(self, X, Y):
        """
        get posterior mean of w
        """

        X = self._preprocess_X(X)
        X = np.mat(X)
        Y = np.mat(Y)
        cov_mat = self._get_cor_mat(X)
        if self.m_flag == 1:
            self.w = self.w + self.cov_w * X.T * np.linalg.inv(
                np.linalg.inv(cov_mat + self.hp_mat * np.eye(X.shape[0])) + X *
                self.cov_w * X.T + self.hp_mat * np.eye(X.shape[0])) * (
                             Y.T - X * self.w)
        else:
            self.w = np.linalg.inv(X.T * np.linalg.inv(
                cov_mat + self.hp_mat * np.eye(X.shape[0])) * X + np.linalg.inv(
                self.cov_w + self.hp_mat * np.eye(X.shape[
                                                      1])) + self.hp_mat * np.eye(X.shape[1])) * (
                             X.T * np.linalg.inv(cov_mat + self.hp_mat * np.eye(
                         X.shape[0])) * Y.T +
                             np.linalg.inv(self.cov_w + self.hp_mat * np.eye(
                                 X.shape[1])) * self.w)

        return self.w

    def get_posterior_cov(self, X, Y):
        """
        get posterior coviarnce matrix of w
        """

        X = self._preprocess_X(X)
        X = np.mat(X)
        Y = np.mat(Y)
        cov_mat = self._get_cor_mat(X)
        self.cov_mat = np.linalg.inv(
            np.linalg.inv(X.T * cov_mat * X + self.hp_mat * np.eye(X.shape[1]))
            + np.linalg.inv(self.cov_w + self.hp_mat * np.eye(X.shape[
                                                                  1])) + self.hp_mat * np.eye(X.shape[1]))

        return self.cov_mat
    
    # 增加部分
    def get_params(self, deep=True):
        return {
            "hp_mat": self.hp_mat,
            "hp_cov": self.hp_cov,
            "cov_w": self.cov_w,
            "w": self.w,
            "c_flag": self.c_flag,
            "m_flag": self.m_flag,
            "icov": self.icov,
            }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X, Y):
        self.get_initial_mean(X[0::2], Y[0::2])
        self.get_initial_cov(X)
        self.get_posterior_mean(X[1::2], Y[1::2])

    def predict(self, X):
        X = self._preprocess_X(X)
        X = np.mat(X)

        return X * self.w
