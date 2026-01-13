
import pandas as pd


def craft_dataset(embedding_file, label_file, output_folder):
    """ """

    # load embedding
    df = pd.read_csv(embedding_file)

    # load labels
    df_label = pd.read_csv(label_file)
    target_name_list = list(set(list(df_label['target_name'])))


    for target_name in target_name_list:

        df_sub_label = df_label[df_label['target_name'] == target_name]

        data = []
        for index, row in df_sub_label.iterrows():

            # extract data
            i = row['image_path']
            species = row['Species']
            pre = row['Pre_GSHH_NDVI']
            height = row['Height_Ave_cm']
            target_name = row['target_name']
            target = row['target']

            # create vector
            vector = {
                'ID':i.replace('train/', '').replace('.jpg', ''),
                'species':species,
                'height':height,
                'pre':pre,
                'target':target
            }
            data.append(vector)

        # craft dataframe
        df_sub = pd.DataFrame(data)
        df_joined = df.merge(df_sub, on="ID")
        df_joined.to_csv(f"{output_folder}/data_{target_name}.csv", index=False)



    


if __name__ == "__main__":


    craft_dataset("data/test.csv", "data/biomass/train.csv", "data/dataset")

    
