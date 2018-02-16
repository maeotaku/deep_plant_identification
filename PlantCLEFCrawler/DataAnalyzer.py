import os
import csv



base_path = "/Users/maeotaku/OneDrive/PhD_Research/Pasantias/INRiA/Code/PlantCLEFCrawler/"
dataset_path = base_path + "Config/ASUHerbarium/Output/"

def get_num_files(path):
    list_files = os.listdir(path)
    return len(list_files)

def gen_species_dict():
    species = {}
    for subdir, dirs, files in os.walk(dataset_path):
        name = os.path.basename(subdir)
        if name !="":
            species[name] = len(files) / 2
    return species

def write_csv(path, dictionary):
    writer = csv.writer(open(os.path.join(path, "analysis.csv"), 'w'))
    for key in dictionary:
        writer.writerow([key, dictionary[key]])

species = gen_species_dict()
write_csv(dataset_path, species)
