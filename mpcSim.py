

class SimConditions:
    def __init__(self, x0, xr, los_ang, r_tol, hatch_ofst, mean_mtn, time_stp):
        self.x0 = x0
        self.xr = xr
        self.los_ang = los_ang
        self.r_tol = r_tol
        self.hatch_ofst = hatch_ofst
        self.mean_mtn = mean_mtn
        self.time_stp = time_stp

class SimRun:
    def __init__(self, i_term, isSuccess, x_true_pcw, x_est, ctrlHist, ctrlrSeq, noiseHist):
        self.i_term = i_term
        self.isSuccess = isSuccess
        self.x_true_pcw = x_true_pcw
        self.x_est = x_est
        self.ctrlHist = ctrlHist
        self.ctrlrSeq = ctrlrSeq
        self.noiseHist = noiseHist

class Debris:
    def __init__(self, center, side_length):
        self.center = center
        self.side_length = side_length


class MPCParams:
    def __init__(self, Q_state, R_input, R_slack, V_ecr):
        self.Q = Q_state
        self.R_input = R_input
        self.R_slack = R_slack
        self.V_ecr = V_ecr

class FailSafeParams:
    def __init__(self, Q_fail, R_fail, C_int, K_dead):
        self.Q_fail = Q_fail
        self.R_fail = R_fail
        self.C_int = C_int
        self.K_dead = K_dead
