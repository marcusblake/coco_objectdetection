import argparse


def main():
    parser = argparse.ArgumentParser(description="script to help train, test, and build YOLO model on the COCO dataset")

    parser.add_argument("--network_arch", type=str, help="Filepath to JSON file defining neural network architecture")
    parser.add_argument("--model", type=str, help="Filepath to JSON containing parameters and information related to the machine learning model")
    parser.add_argument("--mode", help="", choices=["train_model", "run_tests", "start_video"], required=True)
    
    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    main()