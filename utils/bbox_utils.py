def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x2 + x1) / 2), int((y2 + y1) / 2)

def get_width_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return x2 - x1

def get_height_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return y2 - y1

def measure_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, int(y2)