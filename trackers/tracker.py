import cv2
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
from utils import get_center_of_bbox, get_width_of_bbox, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.tracker.reset()

    def detect_frames(self, frames):
        batch_size=20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {
            "players" : [],
            "referee" : [],
            "ball" : []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Convert to supervision detection format
            detection_sv = sv.Detections.from_ultralytics(detection)
            
            for ind, class_id in enumerate(detection_sv.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_sv.class_id[ind] = cls_names_inv["player"]

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_sv)
            
            tracks["players"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox" : bbox}

                    
                if cls_id == cls_names_inv['referee']:
                    tracks["referee"][frame_num][track_id] = {"bbox" : bbox}

            for frame_detection in detection_sv:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox" : bbox}
                    
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks,f)

            print(detection_with_tracks)

    def draw_ellipse(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_width_of_bbox(bbox)

        cv2.ellipse(
            frame, 
            center = (x_center, y2),
            axes = (int(width), int(width * 0.35)),
            angle = 0,
            startAngle = -45,
            endAngle = 235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = int(x1_rect + 12)
            y1_text = int(y1_rect + 15)

            # Center text for longer numbers
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame, 
                f"{track_id}",
                (x1_text, y1_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.6,
                color = (0,0,0),
                thickness=2
            )
        
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _  = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED) # Triangle
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)        # Border

        return frame
    
    def draw_annotations(self, video_frames, tracks, ball_control):

        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referee"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                team_color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], team_color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player['bbox'], (0,0,255))
            
            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (255, 0, 0))

            output_video_frames.append(frame)

            # Team possessions:
            current_ball_control = ball_control[:frame_num+1]
            team1_possession = current_ball_control.count(1) / len(current_ball_control) * 100
            team2_possession = 100 - team1_possession

            cv2.putText(
                frame, 
                f"Team1 possession:{int(team1_possession)}%",
                (1300, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.8,
                color = (255,255,255),
                thickness=2
            )

            cv2.putText(
                frame, 
                f"Team2 possession:{int(team2_possession)}%",
                (1300, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.8,
                color = (255,255,255),
                thickness=2
            )

        return output_video_frames
    
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', {}) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions