# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from .detection import Detection
from . import iou_matching
from .track import Track
import rich

class Tracker:

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3, window_size=10):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1


        # DDM parameters
        self.match_history = []  # Store match history
        self.evidence_threshold = 3  # Example threshold
        self.window_size = window_size
        self.evidence = {}  # Initialize evidence dictionary
        self.DDM = DDM()


    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """

        # matches, unmatched_tracks, unmatched_detections = self._match(detections)
    
        # # Record current matches
        # self.match_history.append(matches)
        # if len(self.match_history) > self.window_size:
        #     self.match_history.pop(0)

        # # Use evidence accumulation to finalize matches
        # final_matches = self.accumulate_evidence(matches)

        # # Update track set with final matches
        # for track_idx, detection_idx in final_matches:

        #     self.tracks[track_idx].update(self.kf, detections[detection_idx])
        # for track_idx in unmatched_tracks:
        #     self.tracks[track_idx].mark_missed()
        # for detection_idx in unmatched_detections:
        #     self._initiate_track(detections[detection_idx])
        # self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # # Update distance metric
        # active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # features, targets = [], []
        # for track in self.tracks:
        #     if not track.is_confirmed():
        #         continue
        #     features += track.features
        #     targets += [track.track_id for _ in track.features]
        #     track.features = []
        # self.metric.partial_fit(
        #     np.asarray(features), np.asarray(targets), active_targets)
        ####################################################################################
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        
        #开始为DDM积累证据喵
        self.DDM.add_matches(matches)
        self.DDM.add_detections(detections=detections)
        self.DDM.add_tracks(self.tracks)
        self.DDM.remove_old()
        self.DDM.log_info()
        final_matches, new_unmatched_tracks, new_unmatched_detections = self.DDM.final_judge()
        matches = final_matches
        for new_unmatched_track in new_unmatched_tracks:
            unmatched_tracks.append(new_unmatched_track)
        for new_unmatched_detection in new_unmatched_detections:
            unmatched_detections.append(new_unmatched_detection)
        #self.DDM.final_judge()
        #rich.print(matches,unmatched_detections,unmatched_tracks)

         # Record current matches
        self.match_history.append(matches)
        if len(self.match_history) > self.window_size:
                self.match_history.pop(0)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            


            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

  

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, detection.cls_, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1


    

    def accumulate_evidence(self, current_matches):
        rich.print(current_matches)
        return current_matches
        evidence = {}

        # 初始化evidence字典，包含所有可能的匹配对，包括不在current_matches中的
        for past_matches in reversed(self.match_history):
            for track_idx, det_idx in past_matches:
                evidence[(track_idx, det_idx)] = evidence.get((track_idx, det_idx), 0)

        # 初始化current_matches中的匹配对
        for track_idx, det_idx in current_matches:
            evidence[(track_idx, det_idx)] = 1  # 初始分数为1

        # 积累证据
        for past_matches in reversed(self.match_history):
            for track_idx, det_idx in past_matches:
                evidence[(track_idx, det_idx)] += 10  # 如果在过去的匹配中出现过，则增加1分

        # 调整证据
        for key in evidence:
            evidence[key] -= 0.2  # 所有匹配对减去0.2分

        # 过滤匹配
        final_matches = [
            (track_idx, det_idx) for (track_idx, det_idx), score in evidence.items()
            if score >= self.evidence_threshold
        ]

        return final_matches


class DDM:
    def __init__(self, history_lenth = 10, threshold = 5, evidence_rate = 1):
        self.history_lenth = history_lenth
        self.threshold = threshold
        self.evidence_rate = evidence_rate
        self.history_detections =[]
        self.history_tracks =[]
        self.history_matches = []
        self.frame_id = 0

    def add_detections(self, detections):
        if detections is None:
            detections = []
        self.history_detections.append(detections)
      #  self.remove_old()

    def add_matches(self, matches):
        if matches is None:
            matches = []
        self.history_matches.append(matches)
     #   self.remove_old()

    def add_tracks(self, tracks):
        if tracks is None:
            tracks = []
        self.history_tracks.append(tracks)
     #   self.remove_old()

    def log_info(self):
        # rich.print("DDM当前长度",len(self.history_matches))
        # rich.print("DDM历史匹配：",self.history_matches[-1])
        # rich.print("DDM历史Detecions：",self.history_detections[-1])
        # rich.print("DDM历史Tracks:", self.history_tracks[-1])
        # rich.print("随便一组的tlbr")
        if self.frame_id>5:
          self.total_detection_match()
          rich.print("-------------------------------------------------------------")
          self.total_track_match()
          #self.final_judge()
        self.frame_id = self.frame_id+1
        
    


    def remove_old(self):
        if len(self.history_detections)!=len(self.history_matches) or len(self.history_detections)!=len(self.history_tracks):
            rich.print("DDM历史长度不匹配，请检查代码捏qaq")
            rich.print("Detection长度", len(self.history_detections))
            rich.print("Matches长度", len(self.history_matches))
            rich.print("Tracks长度", len(self.history_tracks))

        while True:
            if len(self.history_matches) > self.history_lenth:
                self.history_matches.pop(0)
                self.history_detections.pop(0)
                self.history_tracks.pop(0)
            else:
                break

    def total_detection_match(self):
        #current_detection = self.history_detections[-1]
        output = []
        for current_detection in self.history_detections[-1]:
            detection_result=[]
            for track in self.history_tracks:
                track_matches = []
                for detection in track:
                    match = self.iou_match(current_detection, detection)
                    track_matches.append(match)
                detection_result.append(track_matches)
            output.append(detection_result)
        #rich.print(output)
        return output
    
    def total_track_match(self):
        output = []
        for current_track in self.history_tracks[-1]:
            detection_result=[]
            for tracks in self.history_tracks:
                track_matches = []
                for track in tracks:
                    match = self.iou_match(current_track, track)
                    track_matches.append(match)
                detection_result.append(track_matches)
            output.append(detection_result)
       # rich.print(output)
        return output

    
    

    def final_judge(self):#最终裁决
        if self.frame_id<10:
            return self.history_matches[-1],[],[]
        current_detections = self.history_detections[-1]
        current_tracks = self.history_tracks[-1]
        current_matches = self.history_matches
        iou_detection_history_tracks_result = self.total_detection_match()
        iou_track_history_tracks_result = self.total_track_match()
       
        iou_detection_history_tracks_result = [sublist[::-1] for sublist in iou_detection_history_tracks_result]
        iou_track_history_tracks_result = [sublist[::-1] for sublist in iou_track_history_tracks_result]
        #iou_detection_history_tracks_result.reverse()
        #iou_track_history_tracks_result.reverse()


        evidence = {}

        #证据初始化捏
           
        # final_evidence = 0

        for now_detection_id, now_detection_id_detections in enumerate(iou_detection_history_tracks_result):
            for now_track_id, now_track_id_tracks in enumerate(iou_track_history_tracks_result):
                result_matrix = self.logical_and_lists(now_detection_id_detections, now_track_id_tracks)
                true_count = sum(sum(row) for row in result_matrix)
                #rich.print(true_count)
                evidence[(now_track_id,now_detection_id)] = self.evidence_rate*true_count
                
        # rich.print(current_matches[-1])
        # rich.print(evidence)
        # rich.print(iou_detection_history_tracks_result[-1])
        rich.print(len(iou_detection_history_tracks_result))
        new_unmatched_tracks = []
        new_unmatched_detections = []
        for (track,detection) in current_matches[-1]:
            if evidence[(track, detection)] <1:
                #pass
                new_unmatched_tracks.append(track)
                new_unmatched_detections.append(detection)
                #rich.print('niuniu', match) 


        
        fianl_matches = [match for match in current_matches[-1] if evidence[match] >= 1]
        return fianl_matches, new_unmatched_tracks, new_unmatched_detections






        # rich.print(iou_track_history_tracks_result)
        # for i in range(len(iou_detection_history_tracks_result)):
        #     for detections_ids,detections in enumerate(iou_detection_history_tracks_result[i]):
        #         for tracks_ids, tracks in enumerate(iou_track_history_tracks_result[i]):
        #             for match_id, match in enumerate(current_matches):
        #                 for detection_id, detection in enumerate(detections):
        #                     for track_id, track in enumerate(tracks):
        #                         if(track == True) and (detection == True) and (detections_ids == tracks_ids) :
        #                             final_evidence += self.evidence_rate + final_evidence

        
                        # if(detection == True) and (track == True):
                        #     final_evidence +=self.evidence_rate
                        #rich.print(detection, track, match, "牛子")

        #rich.print("当前得分：", final_evidence)








    def iou_match(self, detection, track ):#单组的detect与track的iou匹配哦
        ##rich.print(detection.to_tlbr())
        ##rich.print(track.to_tlbr())
        det_tlbr = detection.to_tlbr()
        track_tlbr = track.to_tlbr()

        # Calculate the intersection coordinates
        x_left = max(det_tlbr[0], track_tlbr[0])
        y_top = max(det_tlbr[1], track_tlbr[1])
        x_right = min(det_tlbr[2], track_tlbr[2])
        y_bottom = min(det_tlbr[3], track_tlbr[3])

        # Calculate the area of intersection
        if x_right < x_left or y_bottom < y_top:
            return False  # No overlap

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate the area of both rectangles
        det_area = (det_tlbr[2] - det_tlbr[0]) * (det_tlbr[3] - det_tlbr[1])
        track_area = (track_tlbr[2] - track_tlbr[0]) * (track_tlbr[3] - track_tlbr[1])

        # Calculate the union area
        union_area = det_area + track_area - intersection_area

        # Calculate the IoU
        iou = intersection_area / union_area

        if iou> 0.1:
            return True
        

        return False
    

    def logical_and_lists(self, list1, list2):
        if len(list1) != len(list2) or len(list1[0]) != len(list2[0]):
            return "Error: Input lists are not of the same size"
        
        result = [[list1[i][j] and list2[i][j] for j in range(len(list1[i]))] for i in range(len(list1))]
        
        return result




    



            
        


    
        