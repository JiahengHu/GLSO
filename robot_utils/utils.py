from scipy.stats import norm

def uniform_2_gaussian(vect):
    '''
    Input a vector sampled from uniform distribution
    Transform to gaussian (assuming zero covariance)
    '''
    return norm.ppf(vect)


# this function takes in a torch obj
def gaus_2_uni(vect):
    return norm.cdf(vect.cpu().detach().numpy())

