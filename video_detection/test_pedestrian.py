from pedestrian_detection import PedestrianDetector
from datatypes import Camera

cameras = [
    Camera(
        name="Pedestrians 1",
        source="video_detection/try.mp4",
        points=[
            [170, 230],
            [380, 225],
            [410, 330],
            [150, 350],
        ],
    ),
    Camera(
        name="Pedestrians 2",
        source="video_detection/calibrate_4.mp4",
        points=[
            [235, 200],
            [410, 195],
            [460, 270],
            [210, 275],
        ],
    ),
]


def main():
    detector = PedestrianDetector(
        model_path="../yolo26s.pt",
    )

    camera = cameras[1]

    records = detector.process_video(
        video_source=camera.source,
        polygon_pts=camera.points,
        window_name=f"Pedestrian Detector - {camera.name}",
    )
    print(records)


if __name__ == "__main__":
    main()
