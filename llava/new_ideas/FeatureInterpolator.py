import numpy as np


class FeatureInterpolator:
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    def __init__(self, X, biased_idx=0):
        self.X = X
        self.feat_idx = biased_idx
        self.set_orig_covariances()
        self.set_orig_means()

    def set_orig_covariances(self):
        self.cov_all = np.cov(self.X, rowvar=False)
        self.cov11 = self.cov_all[self.feat_idx, self.feat_idx]
        self.cov12 = np.delete(self.cov_all[self.feat_idx, :], self.feat_idx)
        self.cov21 = self.cov12.T
        self.cov22 = np.delete(self.cov_all, self.feat_idx, axis=0)
        self.cov22 = np.delete(self.cov22, self.feat_idx, axis=1)

    def set_orig_means(self):
        self.mu1 = np.mean(self.X[:, self.feat_idx])
        static_feats = np.delete(self.X, self.feat_idx, axis=1)
        self.mu2 = np.mean(static_feats, axis=0)

    def interpolate(self, new_data):
        cov_prod = np.matmul(self.cov12, np.linalg.inv(self.cov22))
        a_all = np.delete(new_data, self.feat_idx, axis=1)
        a_dist_to_mu = a_all - np.vstack([self.mu2 for i in range(new_data.shape[0])])
        interpolated_feature = []
        for sample_idx in range(new_data.shape[0]):
            a_dist = a_dist_to_mu[sample_idx, :]
            interp_val = self.mu1 + np.dot(cov_prod, a_dist)
            interpolated_feature.append(interp_val)
        interpolated_data = np.copy(new_data)
        interpolated_data[:, self.feat_idx] = interpolated_feature
        return interpolated_data

    def zero_out(self, new_data):
        interpolated_data = np.copy(new_data)
        interpolated_data[:, self.feat_idx] = 0
        return interpolated_data
