import numpy as np
from omni.audio2face.common import log_error, log_info, log_warn
from scipy.optimize import lsq_linear
from pythonosc import udp_client

class FacsSolver:

    def __init__(self, neutral_mat, delta_mat):
        self.client = udp_client.SimpleUDPClient('127.0.0.1', 5008)
        self.weightRegulCoeff = 3.5
        self.weightRegulCoeff_scale = 10.0
        self.prevRegulCoeff = 3.5
        self.prevRegulCoeff_scale = 100.0
        self.sparseRegulCoeff = 1.0
        self.sparseRegulCoeff_scale = 0.25
        self.symmetryRegulCoeff = 1.0
        self.symmetryRegulCoeff_scale = 10.0

        self.neutral_mat = neutral_mat
        self.delta_mat_orig = delta_mat
        self.delta_mat = delta_mat

        self.numPoses_orig = self.delta_mat_orig.shape[1]
        self.numPoses = self.numPoses_orig

        self.lb_orig = np.zeros(self.numPoses_orig)
        self.ub_orig = self.lb_orig + 1.0
        self.lb = self.lb_orig.copy()
        self.ub = self.ub_orig.copy()

        self.activeIdxMap = range(self.numPoses_orig)
        self.activePosesBool = np.array([True for pi in range(self.numPoses_orig)], dtype=bool)
        self.cancelPoseIndices = np.array([-1 for pi in range(self.numPoses_orig)], dtype=int)
        self.symmetryPoseIndices = np.array([-1 for pi in range(self.numPoses_orig)], dtype=int)
        self.cancelList = []
        self.symmetryList = []
        self.symShapeMat = np.zeros((self.numPoses_orig, self.numPoses_orig))
        self.prevWeights = np.zeros(self.numPoses_orig)
        # TODO L1 implementation
        l1RegulMat = np.ones((1, self.numPoses))
        self.l1RegulMat = np.dot(l1RegulMat.T, l1RegulMat)

        self.compute_A_mat()

    def compute_A_mat(self):
        self.A = (
            np.dot(self.delta_mat.T, self.delta_mat)
            + self.weightRegulCoeff * self.weightRegulCoeff_scale * np.eye(self.numPoses)
            + self.prevRegulCoeff * self.prevRegulCoeff_scale * np.eye(self.numPoses)
            + self.sparseRegulCoeff ** 2 * self.sparseRegulCoeff_scale * self.l1RegulMat
            + self.symmetryRegulCoeff * self.symmetryRegulCoeff_scale * self.symShapeMat
        )
        self.A = self.A.astype(np.float64)

    def set_activePoses(self, activePosesBool):
        self.activePosesBool = activePosesBool

        # 1 - simple approach
        # self.ub *= np.array(self.activePosesBool)

        # 2- less computation way
        self.delta_mat = self.delta_mat_orig[:, self.activePosesBool]
        self.numPoses = self.delta_mat.shape[1]
        self.lb = self.lb_orig[self.activePosesBool]
        self.ub = self.ub_orig[self.activePosesBool]
        self.prevWeights = np.zeros(self.numPoses)

        self.activeIdxMap = []
        cnt = 0
        for idx in range(self.numPoses_orig):
            if self.activePosesBool[idx]:
                self.activeIdxMap.append(cnt)
                cnt += 1
            else:
                self.activeIdxMap.append(-1)

        # update L1 regularization mat
        l1RegulMat = np.ones((1, self.numPoses))
        self.l1RegulMat = np.dot(l1RegulMat.T, l1RegulMat)

        # update cancel pair index
        self.set_cancelPoses(self.cancelPoseIndices)

        # update symmetry pair index
        self.set_symmetryPoses(self.symmetryPoseIndices)  # update self.A here

    def set_cancelPoses(self, cancelPoseIndices):
        self.cancelPoseIndices = cancelPoseIndices
        # filter out cancel shapes
        self.cancelList = []
        maxIdx = np.max(self.cancelPoseIndices)
        if maxIdx < 0:
            return

        for ci in range(maxIdx + 1):
            cancelIndices = np.where(self.cancelPoseIndices == ci)[0]
            if len(cancelIndices) > 2:
                log_warn("There is more than 2 poses for a cancel index %d" % ci)
                break
            elif len(cancelIndices) < 2:
                log_warn("There is less than 2 poses for a cancel index %d" % ci)
                break
            self.cancelList.append(cancelIndices)
        # print ('cancel shape list', self.cancelList)

        activeCancelList = []
        for pIdx1, pIdx2 in self.cancelList:
            if self.activePosesBool[pIdx1] and self.activePosesBool[pIdx2]:
                activeCancelList.append([self.activeIdxMap[pIdx1], self.activeIdxMap[pIdx2]])

        # print (activeCancelList)
        self.cancelList = activeCancelList

    def set_symmetryPoses(self, symmetryPoseIndices):
        self.symmetryPoseIndices = symmetryPoseIndices
        self.symmetryList = []

        maxIdx = np.max(self.symmetryPoseIndices)
        if maxIdx < 0:
            self.symShapeMat = np.zeros((self.numPoses, self.numPoses))
        else:
            for ci in range(maxIdx + 1):
                symmetryIndices = np.where(self.symmetryPoseIndices == ci)[0]
                if len(symmetryIndices) > 2:
                    log_warn("There is more than 2 poses for a cancel index %d" % ci)
                    break
                elif len(symmetryIndices) < 2:
                    log_warn("There is less than 2 poses for a cancel index %d" % ci)
                    break
                self.symmetryList.append(symmetryIndices)

            activeSymmetryList = []
            for pIdx1, pIdx2 in self.symmetryList:
                if self.activePosesBool[pIdx1] and self.activePosesBool[pIdx2]:
                    activeSymmetryList.append([self.activeIdxMap[pIdx1], self.activeIdxMap[pIdx2]])

            self.symmetryList = activeSymmetryList

            symShapeMat = np.zeros((len(self.symmetryList), self.numPoses))
            for si, [pose1Idx, pose2Idx] in enumerate(self.symmetryList):
                symShapeMat[si, pose1Idx] = 1.0
                symShapeMat[si, pose2Idx] = -1.0
            self.symShapeMat = np.dot(symShapeMat.T, symShapeMat)

        self.compute_A_mat()

    def set_l2_regularization(self, L2=3.5):
        self.weightRegulCoeff = L2
        self.compute_A_mat()

    def set_tempo_regularization(self, temporal=3.5):
        self.prevRegulCoeff = temporal
        self.compute_A_mat()

    def set_l1_regularization(self, L1=1.0):
        self.sparseRegulCoeff = L1
        self.compute_A_mat()

    def set_symmetry_regularization(self, value=1.0):
        self.symmetryRegulCoeff = value
        self.compute_A_mat()

    def computeFacsWeights(self, point_mat):
        target_delta_mat = point_mat - self.neutral_mat
        B = (
            np.dot(self.delta_mat.T, target_delta_mat).flatten()
            + self.prevRegulCoeff * self.prevRegulCoeff_scale * self.prevWeights
        )
        B = B.astype(np.float64)

        res = lsq_linear(self.A, B, bounds=(self.lb, self.ub), lsmr_tol="auto", verbose=0, method="bvls")

        # print ('first pass:', res.x)
        if len(self.cancelList) > 0:
            # check cancelling poses -
            ub = self.ub.copy()
            lb = self.lb.copy()

            for pose1Idx, pose2Idx in self.cancelList:
                if res.x[pose1Idx] >= res.x[pose2Idx]:
                    ub[pose2Idx] = 1e-10
                else:
                    ub[pose1Idx] = 1e-10

            res = lsq_linear(self.A, B, bounds=(lb, ub), lsmr_tol="auto", verbose=0, method="bvls")

        self.prevWeights = res.x
        # print ('second pass:', res.x)

        outWeight = np.zeros(self.numPoses_orig)
        outWeight[self.activePosesBool] = res.x

        outWeight = outWeight * (outWeight > 1.0e-9)
        # print (outWeight)

        mh_ctl_list = [
            ['CTRL_expressions_browDownR', "browLowerR", 1.0],  
            ['CTRL_expressions_browDownL', "browLowerL", 1.0], 

            ['CTRL_expressions_browLateralR', "browLowerR", 1.0],
            ['CTRL_expressions_browLateralL', "browLowerL", 1.0],

            ['CTRL_expressions_browRaiseinR', "innerBrowRaiserR", 0.5], 
            ['CTRL_expressions_browRaiseinL', "innerBrowRaiserL", 0.5],

            ['CTRL_expressions_browRaiseouterR', "innerBrowRaiserR", 0.5], 
            ['CTRL_expressions_browRaiseouterL', "innerBrowRaiserL", 0.5],

            ['CTRL_expressions_eyeLookUpR', "eyesLookUp", 1.0, "eyesLookDown", -1.0],
            ['CTRL_expressions_eyeLookDownR', "eyesLookUp", 1.0, "eyesLookDown", -1.0],
            ['CTRL_expressions_eyeLookUpL', "eyesLookUp", 1.0, "eyesLookDown", -1.0],
            ['CTRL_expressions_eyeLookDownL', "eyesLookUp", 1.0, "eyesLookDown", -1.0],
            

            ['CTRL_expressions_eyeLookLeftR', "eyesLookLeft", 1.0, "eyesLookRight", -1.0],
            ['CTRL_expressions_eyeLookRightR', "eyesLookLeft", 1.0, "eyesLookRight", -1.0],
            ['CTRL_expressions_eyeLookRightL', "eyesLookLeft", 1.0, "eyesLookRight", -1.0],
            ['CTRL_expressions_eyeLookRightL', "eyesLookLeft", 1.0, "eyesLookRight", -1.0],
            
            
            ['CTRL_expressions_eyeBlinkR', "eyesCloseR", 1.0, "eyesUpperLidRaiserR", -1.0], 
            ['CTRL_expressions_eyeBlinkL', "eyesCloseR", 1.0, "eyesUpperLidRaiserR", -1.0], 

            ['CTRL_expressions_eyeSquintinnerR', "squintR", 1.0], 
            ['CTRL_expressions_eyeSquintinnerL', "squintL", 1.0],

            ['CTRL_expressions_eyeCheekraiseR', "cheekRaiserR", 1.0], 
            ['CTRL_expressions_eyeCheekraiseL', "cheekRaiserL", 1.0],

            ['CTRL_expressions_mouthCheekSuckR', "cheekPuffR", 0.5], 
            ['CTRL_expressions_mouthCheekBlowR', "cheekPuffR", 0.5], 

            ['CTRL_expressions_mouthCheekSuckL', "cheekPuffL", 0.5],
            ['CTRL_expressions_mouthCheekBlowL', "cheekPuffL", 0.5],

            ['CTRL_expressions_noseNostrilDilateR', "noseWrinklerR", 1.0],
            ['CTRL_expressions_noseNostrilCompressR', "noseWrinklerR", 1.0],
            ['CTRL_expressions_noseWrinkleR', "noseWrinklerR", 1.0],
            ['CTRL_expressions_noseNostrilDepressR', "noseWrinklerR", 1.0],

            ['CTRL_expressions_noseNostrilDilateL', "noseWrinklerL", 1.0],
            ['CTRL_expressions_noseNostrilCompressL', "noseWrinklerL", 1.0],
            ['CTRL_expressions_noseWrinkleL', "noseWrinklerL", 1.0],
            ['CTRL_expressions_noseNostrilDepressL', "noseWrinklerL", 1.0],


            ['CTRL_expressions_jawOpen', "jawDrop", 1.0, "jawDropLipTowards", 0.6],

            ['CTRL_R_mouth_lipsTogetherU', "jawDropLipTowards", 1.0],
            ['CTRL_L_mouth_lipsTogetherU', "jawDropLipTowards", 1.0],
            ['CTRL_R_mouth_lipsTogetherD', "jawDropLipTowards", 1.0],
            ['CTRL_L_mouth_lipsTogetherD', "jawDropLipTowards", 1.0],

            ['CTRL_expressions_jawFwd', "jawThrust", -1.0],
            ['CTRL_expressions_jawBack', "jawThrust", -1.0],

            ['CTRL_expressions_jawRight', "jawSlideLeft", -1.0, "jawSlideRight", 1.0],
            ['CTRL_expressions_jawLeft', "jawSlideLeft", -1.0, "jawSlideRight", 1.0],

            ['CTRL_expressions_mouthLeft', "mouthSlideLeft", 0.5, "mouthSlideRight", -0.5],
            ['CTRL_expressions_mouthRight', "mouthSlideLeft", 0.5, "mouthSlideRight", -0.5],

            ['CTRL_expressions_mouthDimpleR', "dimplerR", 1.0], 
            ['CTRL_expressions_mouthDimpleL', "dimplerL", 1.0],

            ['CTRL_expressions_mouthCornerPullR', "lipCornerPullerR", 1.0], 
            ['CTRL_expressions_mouthCornerPullL', "lipCornerPullerL", 1.0], 

            ['CTRL_expressions_mouthCornerDepressR', "lipCornerDepressorR", 1.0], 
            ['CTRL_expressions_mouthCornerDepressL', "lipCornerDepressorL", 1.0],

            ['CTRL_expressions_mouthStretchR', "lipStretcherR", 1.0], 
            ['CTRL_expressions_mouthStretchL', "lipStretcherL", 1.0], 

            ['CTRL_expressions_mouthUpperLipRaiseR', "upperLipRaiserR", 1.0], 
            ['CTRL_expressions_mouthUpperLipRaiseL', "upperLipRaiserL", 1.0],

            ['CTRL_expressions_mouthLowerLipDepressR', "lowerLipDepressorR", 1.0], 
            ['CTRL_expressions_mouthLowerLipDepressL', "lowerLipDepressorR", 1.0],

            ['CTRL_expressions_jawChinRaiseDR', "chinRaiser", 1.0], 
            ['CTRL_expressions_jawChinRaiseDL', "chinRaiser", 1.0],

            ['CTRL_expressions_mouthLipsPressR', "lipPressor", 1.0], 
            ['CTRL_expressions_mouthLipsPressL', "lipPressor", 1.0],

            ['CTRL_expressions_mouthLipsTowardsUR', "pucker", 1.0], 
            ['CTRL_expressions_mouthLipsTowardsUL', "pucker", 1.0], 

            ['CTRL_expressions_mouthLipsTowardsDR', "pucker", 1.0], 
            ['CTRL_expressions_mouthLipsTowardsDL', "pucker", 1.0], 

            ['CTRL_expressions_mouthLipsPurseUR', "pucker", 1.0], 
            ['CTRL_expressions_mouthLipsPurseUL', "pucker", 1.0], 

            ['CTRL_expressions_mouthLipsPurseDR', "pucker", 1.0], 
            ['CTRL_expressions_mouthLipsPurseDL', "pucker", 1.0],

            ['CTRL_expressions_mouthFunnelUR', "funneler", 1.0], 
            ['CTRL_expressions_mouthFunnelUL', "funneler", 1.0], 

            ['CTRL_expressions_mouthFunnelDL', "funneler", 1.0], 
            ['CTRL_expressions_mouthFunnelDR', "funneler", 1.0],

            ['CTRL_expressions_mouthPressUR', "lipSuck", 1.0], 
            ['CTRL_expressions_mouthPressUL', "lipSuck", 1.0], 

            ['CTRL_expressions_mouthPressDR', "lipSuck", 1.0], 
            ['CTRL_expressions_mouthPressDL', "lipSuck", 1.0]
        ]

        facsNames = [
            "browLowerL",
            "browLowerR",
            "innerBrowRaiserL",
            "innerBrowRaiserR",
            "outerBrowRaiserL",
            "outerBrowRaiserR",
            "eyesLookLeft",
            "eyesLookRight",
            "eyesLookUp",
            "eyesLookDown",
            "eyesCloseL",
            "eyesCloseR",
            "eyesUpperLidRaiserL",
            "eyesUpperLidRaiserR",
            "squintL",
            "squintR",
            "cheekRaiserL",
            "cheekRaiserR",
            "cheekPuffL",
            "cheekPuffR",
            "noseWrinklerL",
            "noseWrinklerR",
            "jawDrop",
            "jawDropLipTowards",
            "jawThrust",
            "jawSlideLeft",
            "jawSlideRight",
            "mouthSlideLeft",
            "mouthSlideRight",
            "dimplerL",
            "dimplerR",
            "lipCornerPullerL",
            "lipCornerPullerR",
            "lipCornerDepressorL",
            "lipCornerDepressorR",
            "lipStretcherL",
            "lipStretcherR",
            "upperLipRaiserL",
            "upperLipRaiserR",
            "lowerLipDepressorL",
            "lowerLipDepressorR",
            "chinRaiser",
            "lipPressor",
            "pucker",
            "funneler",
            "lipSuck"
        ]


        
        for i in range(len(mh_ctl_list)):
            ctl_value = 0
            numInputs = (len(mh_ctl_list[i])-1) // 2
            for j in range(numInputs):
                weightMat = outWeight.tolist()
                poseIdx = facsNames.index(mh_ctl_list[i][j*2+1])
                ctl_value += weightMat[poseIdx] * mh_ctl_list[i][j*2+2]
            
            print(mh_ctl_list[i][0], ctl_value)
            self.client.send_message('/' + mh_ctl_list[i][0], ctl_value)
            
        return outWeight