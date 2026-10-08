from vehicle_detection import VehicleDetector
from datatypes import Camera

cameras = [
    Camera(
        name="Vehicles 1",
        source="video_detection/calibrate_1.mp4",
        points=[
            [230, 20],
            [410, 20],
            [550, 380],
            [150, 400],
        ],

    ),
    Camera(
        name="Vehicles 2",
        source="video_detection/calibrate_2.mp4",
        points=[
            [300, 50],
            [350, 50],
            [520, 450],
            [150, 450],
        ],
    ),
]


def main():
    detector = VehicleDetector(
        model_path="../yolo26s.pt",
        conf_thresh=0.15,
        miss_limit=15,
        frame_skip=2
    )

    camera = cameras[0]

    records = detector.process_video(
        video_source=camera.source,
        zone_pts=camera.points,
        window_name=f"Vehicle Detector - {camera.name}",
    )

    print(records)


if __name__ == "__main__":
    main()
