






'''
    1) Clean out bounding box that are larger than 50% of the image
    2) Clean out bounding boxes that are smaller than 1% of the size of the image

'''
import ast

import pandas as pd
import hydra
import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    train_path = "../Datasets/global-wheat-detection/global-wheat-detection/train.csv"
    train_data = pd.read_csv(train_path)
    bbox = train_data[['bbox']]  ## string of float list

    areas = []
    indexes_to_filter = []
    for index, row in bbox.iterrows():
        bbox = ast.literal_eval(row['bbox'])
        if index < 5:
            print(bbox)
        areas.append(bbox[2]*bbox[3])

        if bbox[2]*bbox[3] >= cfg.max_area or bbox[2]*bbox[3] <= cfg.min_area:
            indexes_to_filter.append(index)

    print("Lenght before filtering", len(train_data.index))
    train_data = train_data.drop(axis='index', index=indexes_to_filter)
    train_data = train_data.reset_index(drop=True)
    print("Lenght after filtering", len(train_data.index))

    max_area = max(areas)
    min_area = min(areas)
    plt.hist(areas, bins=250)
    plt.title("Area BBox Distribution, Max Area" + str(max_area) + " Min Area : "+ str(min_area))
    plt.show()

    train_data.to_csv("../Datasets/global-wheat-detection/global-wheat-detection/train_cleaned.csv")



if __name__ == "__main__":

    main()