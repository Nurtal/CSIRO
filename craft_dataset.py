
import pandas as pd


def craft_dataset(embedding_file, label_file):
    """ """

    # load embedding
    df = pd.read_csv(embedding_file)

    # load labels
    df_label = pd.read_csv(label_file)

    data = []
    for index, row in df_label.iterrows():

        # extract data
        i = row['image_path']
        species = row['Species']
        pre = row['Pre_GSHH_NDVI']
        height = row['Height_Ave_cm']
        target_name = row['target_name']
        target = row['target']

        # create vector
        vector = {
            'ID':i.replace('train/', '').replace('.jpg', '')
        }

        print(vector)

    


if __name__ == "__main__":


    craft_dataset("data/test.csv", "data/biomass/train.csv")

    
