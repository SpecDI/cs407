def parse_yolo_frame(frame_id, yolo_frame):
    """Convert a dictionary stored in the yolo format to a dictionary stored in mot16 format

    Parameters
    ----------
    frame_id     :  int
        The id of the current frame.
    yolo_frame   :  List
        List of dictionary entries for each individual identified in the corresponding frame.

    Returns
    -------
    List
        List of MOT16 based individuals    
    """

    frame_entries = []
    for entity in yolo_frame:
        current_line = str(frame_id) + ' -1 '
        current_line += str(entity['topleft']['x']) + ' '
        current_line += str(entity['topleft']['y']) + ' '

        bb_width = abs(entity['bottomright']['x'] - entity['topleft']['x'])
        current_line += str(bb_width) + ' '

        bb_height = abs(entity['topleft']['y'] - entity['bottomright']['y'])
        current_line += str(bb_height) + ' '
        
        current_line += str(entity['confidence']) + ' -1 -1 -1'

        frame_entries.append(current_line)

    return frame_entries


def main():
    yolo_frame = [{'label': 'person', 'confidence': 0.571275, 'topleft': {'x': 2035, 'y': 822}, 'bottomright': {'x': 2084, 'y': 957}}, {'label': 'person', 'confidence': 0.55631423, 'topleft': {'x': 2791, 'y': 872}, 'bottomright': {'x': 2843, 'y': 1004}}, {'label': 'person', 'confidence': 0.2706736, 'topleft': {'x': 2939, 'y': 877}, 'bottomright': {'x': 3025, 'y': 1020}}, {'label': 'person', 'confidence': 0.50331444, 'topleft': {'x': 2916, 'y': 821}, 'bottomright': {'x': 3077, 'y': 1056}}, {'label': 'person', 'confidence': 0.18280241, 'topleft': {'x': 3012, 'y': 900}, 'bottomright': {'x': 3075, 'y': 1000}}, {'label': 'person', 'confidence': 0.5808667, 'topleft': {'x': 2370, 'y': 978}, 'bottomright': {'x': 2420, 'y': 1127}}, {'label': 'person', 'confidence': 0.33659056, 'topleft': {'x': 1532, 'y': 1141}, 'bottomright': {'x': 1654, 'y': 1280}}, {'label': 'person', 'confidence': 0.3054465, 'topleft': {'x': 1615, 'y': 1127}, 'bottomright': {'x': 1680, 'y': 1277}}, {'label': 'person', 'confidence': 0.3813942, 'topleft': {'x': 1905, 'y': 1178}, 'bottomright': {'x': 1976, 'y': 1296}}, {'label': 'person', 'confidence': 0.5412082, 'topleft': {'x': 1926, 'y': 1202}, 'bottomright': {'x': 1990, 'y': 1331}}, {'label': 'bird', 'confidence': 0.17827891, 'topleft': {'x': 1519, 'y': 1130}, 'bottomright': {'x': 1684, 'y': 1295}}]
    mot_sequence = parse_yolo_frame(1, yolo_frame)
    for entry in mot_sequence:
        print(entry)


if __name__ == "__main__":
    main()