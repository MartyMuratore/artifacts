import numpy as np
from scipy import stats

from eryn.moves import GroupStretchMove
class MeanGaussianGroupMove(GroupStretchMove):
    name = "mean gaussian group move"

    def __init__(self, **kwargs):
        
        GroupStretchMove.__init__(self, **kwargs)
        self.original_nfriends = self.nfriends

    def setup_friends(self, branches):
   
        ntemps, nwalkers, nleaves_max, ndim = branches["glitch"].shape
        # store cold-chain information
        friends = branches["glitch"].coords[0, branches["glitch"].inds[0]]

        time_occur = friends[:, 0].copy()  # need the copy

        # take unique to avoid errors at the start of sampling
        self.time_occur, uni_inds = np.unique(time_occur, return_index=True)
        self.friends = friends[uni_inds]

        if friends.size ==0:
           return

        # sort    
        inds_sort = np.argsort(self.time_occur)
        self.friends[:] = self.friends[inds_sort]
        self.time_occur[:] = self.time_occur[inds_sort]
        
        # get all current time_occur from all temperatures
        current_time_occur = branches["glitch"].coords[branches["glitch"].inds, 0]

        # calculate their distances to each stored friend
        dist = np.abs(current_time_occur[:, None] - self.time_occur[None, :])
        
        # get closest friends
        inds_closest = np.argsort(dist, axis=1)[:, : self.nfriends_now]
        
        # store in branch supplemental
        branches["glitch"].branch_supplemental[branches["glitch"].inds, :self.nfriends_now] = {
            "inds_closest": inds_closest
        }

        # make sure to "turn off" leaves that are deactivated by setting their 
        # index to -1. 
        branches["glitch"].branch_supplemental[~branches["glitch"].inds, :self.nfriends_now] = {
            "inds_closest": -np.ones(
                (ntemps, nwalkers, nleaves_max, self.nfriends_now), dtype=int
            )[~branches["glitch"].inds]
        }

        if self.nfriends_now < self.original_nfriends and np.any(branches["glitch"].branch_supplemental[:]["inds_closest"] >= self.nfriends_now):
            print('less available friends than you want but still >0')
            

    def fix_friends(self, branches):
        # when RJMCMC activates a new leaf, when it gets to this proposal, its inds_closest
        # will need to be updated
  
        # activated & does not have an assigned index
        fix = branches["glitch"].inds & (
            np.all(
                branches["glitch"].branch_supplemental[:]["inds_closest"] == -1,
                axis=-1,
            )
        )

        if not np.any(fix):
            return

        # same process as above, only for fix 
        current_time_occur = branches["glitch"].coords[fix, 0]

        dist = np.abs(current_time_occur[:, None] - self.time_occur[None, :])
        inds_closest = np.argsort(dist, axis=1)[:, : self.nfriends_now]

        branches["glitch"].branch_supplemental[fix, :self.nfriends_now] = {
            "inds_closest": inds_closest
        }

        # verify everything worked
        fix_check = branches["glitch"].inds & (
            np.all(
                branches["glitch"].branch_supplemental[:]["inds_closest"] == -1,
                axis=-1,
            )
        )
        #assert not np.any(fix_check) # i comment this out since i do not want to break
   

        if np.any(fix_check):
            return

        if self.nfriends_now < self.original_nfriends and np.any(branches["glitch"].branch_supplemental[:]["inds_closest"] >= self.nfriends_now):
            print('less available friends than you want but still >0')
           # breakpoint()
    
    @property
    def nfriends_now(self):
        if self.friends is None:
            return self.original_nfriends
        return self.original_nfriends if self.original_nfriends < self.friends.shape[0] else self.friends.shape[0]


    def find_friends(self, name, s, s_inds=None, branch_supps=None):

     
        # prepare buffer array
        friends = np.zeros_like(s)
  
        # determine the closest friends for s_inds == True
        inds_closest_here = branch_supps[name][s_inds]["inds_closest"]


        if self.friends.size ==0:
            return

        if self.nfriends_now < self.original_nfriends and np.any(branch_supps["glitch"][:]["inds_closest"] >= self.nfriends_now):
            breakpoint()

        # take one at random
        random_inds = inds_closest_here[
            np.arange(inds_closest_here.shape[0]),
            np.random.randint(
                self.nfriends, size=(inds_closest_here.shape[0],)
            ),
        ]

        # store in buffer array
        
        friends[s_inds] = self.friends[random_inds]


        return friends

    def get_proposal(self, s_all, random, gibbs_ndim=None, s_inds_all=None, branch_supps=None, **kwargs):

        ntemps, nwalkers, nleaves_max = s_all[list(s_all.keys())[0]].shape[:3]
        
        assert branch_supps is not None
        if self.friends is None or np.all(branch_supps["glitch"][:]["inds_closest"] == -1):  #  or self.nfriends_now < self.original_nfriends:
            factors = np.full((ntemps, nwalkers), -1e300)
            return s_all, factors
        else:
            # Use the parent class's proposal logic.
            return super(MeanGaussianGroupMove,self).get_proposal(s_all, random, gibbs_ndim=gibbs_ndim, s_inds_all=s_inds_all, branch_supps=branch_supps, **kwargs)






from eryn.moves import GroupMove

class SelectedCovarianceGroupMove(GroupMove):
    name = "selected covariance"

    def __init__(self, means=None, covs=None, **kwargs):
        
        GroupMove.__init__(self, **kwargs)

        self.covs = covs
        self.means = means
        if self.covs is not None or self.means is not None:
            assert self.covs is not None and self.means is not None
        
        self.setup_distributions()

    def setup_distributions(self):
        if self.means is None:
            return
        
        self.dists = [None for _ in range(len(self.means))]

        for i, (mean, cov) in enumerate(zip(self.means, self.covs)):
            self.dists[i] = stats.multivariate_normal(mean, cov, allow_singular=True)

    def setup_friends(self, branches):
        
        if self.means is None:
            return

        self.nfriends = self.means.shape[0]

        ntemps, nwalkers, nleaves_max, ndim = branches["glitch"].shape
        
        # get all current time_occur from all temperatures
       
        current_time_occur = branches["glitch"].coords[branches["glitch"].inds, 0]

        # calculate their distances to each stored friend
        dist = np.abs(current_time_occur[:, None] - self.means[None, :, 0])
        
        # get closest friends
        inds_closest = np.argmin(dist, axis=1)
        
        # store in branch supplemental
        branches["glitch"].branch_supplemental[branches["glitch"].inds] = {
            "inds_closest_cov": inds_closest
        }

        # make sure to "turn off" leaves that are deactivated by setting their 
        # index to -1. 
        branches["glitch"].branch_supplemental[~branches["glitch"].inds] = {
            "inds_closest_cov": -np.ones(
                (ntemps, nwalkers, nleaves_max), dtype=int
            )[~branches["glitch"].inds]
        }

    def fix_friends(self, branches):
        # when RJMCMC activates a new leaf, when it gets to this proposal, its inds_closest
        # will need to be updated
        if self.means is None:
            return
      
        # activated & does not have an assigned index
        fix = branches["glitch"].inds & (
            branches["glitch"].branch_supplemental[:]["inds_closest_cov"] == -1
        )

        if not np.any(fix):
            return

        # same process as above, only for fix 
        current_time_occur = branches["glitch"].coords[fix, 0]

        dist = np.abs(current_time_occur[:, None] - self.means[None, :, 0])
        inds_closest = np.argmin(dist, axis=1)

        branches["glitch"].branch_supplemental[fix] = {
            "inds_closest_cov": inds_closest
        }

        # verify everything worked
        fix_check = branches["glitch"].inds & (
            branches["glitch"].branch_supplemental[:]["inds_closest_cov"] == -1
        )
        assert not np.any(fix_check)

    def update_mean_cov(self, branches, means, covs):
        self.means = means
        self.covs = covs
        self.setup_distributions()
        self.setup_friends(branches)

    def get_proposal(self, s_all, random, gibbs_ndim=None, s_inds_all=None, branch_supps=None):
      
        ntemps, nwalkers, nleaves_max = s_all[list(s_all.keys())[0]].shape[:3]

        if self.means is None:
            return s_all, np.zeros((ntemps, nwalkers))

        assert list(s_all.keys()) == ["glitch"]
        
        s = s_all["glitch"]
        s_inds = s_inds_all["glitch"]

        # prepare buffer array
        q_tmp = np.zeros_like(s)
        factors_tmp = np.zeros((ntemps, nwalkers, nleaves_max))
        
        # determine the closest friends for s_inds == True
        inds_closest_all = branch_supps["glitch"][:]["inds_closest_cov"]

        for i in np.unique(inds_closest_all):
            assert i < len(self.dists)

            current = (inds_closest_all == i) & s_inds
            current_params = s[current]
            current_logpdf = self.dists[i].logpdf(current_params)

            new_params = self.dists[i].rvs(size=(current_params.shape[0],))
            new_logpdf = self.dists[i].logpdf(new_params)

            factors_tmp[current] = current_logpdf - new_logpdf
            q_tmp[current] = new_params

        q_out = {"glitch": q_tmp}
        factors = factors_tmp.sum(axis=-1)

        return q_out, factors
        


        # take one at random
        random_inds = inds_closest_here[np.arange(inds_closest_here.shape[0]),np.random.randint(self.nfriends, size=(inds_closest_here.shape[0],)),]

        # store in buffer array
        friends[s_inds] = self.friends[random_inds]
        return friends


###### ----------- ##########


class MeanGaussianGroupMove_MBH(GroupStretchMove):
    def __init__(self, **kwargs):
        # make sure kwargs get sent into group stretch parent class
        GroupStretchMove.__init__(self, **kwargs)

    def setup_friends(self, branches):

        ntemps, nwalkers, nleaves_max, ndim = branches["mbh"].shape

        # store cold-chain information
        friends = branches["mbh"].coords[0, branches["mbh"].inds[0]]
        means = friends[:, 10].copy()  # need the copy

        # take unique to avoid errors at the start of sampling
        self.means, uni_inds = np.unique(means, return_index=True)
        self.friends = friends[uni_inds]

        # sort
        inds_sort = np.argsort(self.means)
        self.friends[:] = self.friends[inds_sort]
        self.means[:] = self.means[inds_sort]


        # get all current means from all temperatures
        current_means = branches["mbh"].coords[branches["mbh"].inds, 10]

        # calculate their distances to each stored friend
        dist = np.abs(current_means[:, None] - self.means[None, :])

        # get closest friends
        inds_closest = np.argsort(dist, axis=1)[:, : self.nfriends]

        # store in branch supplimental
        branches["mbh"].branch_supplemental[branches["mbh"].inds] = {
            "inds_closest": inds_closest
        }

        # make sure to "turn off" leaves that are deactivated by setting their
        # index to -1.
        branches["mbh"].branch_supplemental[~branches["mbh"].inds] = {
            "inds_closest": -np.ones(
                (ntemps, nwalkers, nleaves_max, self.nfriends), dtype=int
            )[~branches["mbh"].inds]
        }

    def fix_friends(self, branches):
        # when RJMCMC activates a new leaf, when it gets to this proposal, its inds_closest
        # will need to be updated

        # activated & does not have an assigned index
        fix = branches["mbh"].inds & (
            np.all(
                branches["mbh"].branch_supplemental[:]["inds_closest"] == -1,
                axis=-1,
            )
        )

        if not np.any(fix):
            return

        # same process as above, only for fix
        current_means = branches["mbh"].coords[fix, 1]

        dist = np.abs(current_means[:, None] - self.means[None, :])
        inds_closest = np.argsort(dist, axis=1)[:, : self.nfriends]

        branches["mbh"].branch_supplemental[fix] = {
            "inds_closest": inds_closest
        }

        # verify everything worked
        fix_check = branches["mbh"].inds & (
            np.all(
                branches["mbh"].branch_supplemental[:]["inds_closest"] == -1,
                axis=-1,
            )
        )
        assert not np.any(fix_check)

    def find_friends(self, name, s, s_inds=None, branch_supps=None):

        # prepare buffer array
        friends = np.zeros_like(s)

        # determine the closest friends for s_inds == True
        inds_closest_here = branch_supps[name][s_inds]["inds_closest"]

        # take one at random
        random_inds = inds_closest_here[
            np.arange(inds_closest_here.shape[0]),
            np.random.randint(
                self.nfriends, size=(inds_closest_here.shape[0],)
            ),
        ]

        # store in buffer array
        friends[s_inds] = self.friends[random_inds]
        return friends