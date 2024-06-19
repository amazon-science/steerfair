import sys

sys.path.insert(0, "/home/ubuntu")

from MM_Bench import MMBenchDataset

if __name__ == "__main__":
    dataset = MMBenchDataset("/home/ubuntu/MM_Bench/mmbench_dev_en_20231003.tsv")
    print(len(dataset))
    print(dataset[0].keys())
    for key in list(dataset[0].keys()):
        print(key)
        print(dataset[0][key])
        print("")
