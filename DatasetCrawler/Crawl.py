from DataLayer.Base import *
from Crawler.DataSetDumper import *
from DataLayer.ASUHerbariaPlantDataSet import *
from DataLayer.CRNationalMuseumLeavesDataSet import *
from DataLayer.IDigBioDataSet import *
from DataLayer.PlantVillageDataSet import *

'''
#PlantCLEF
base_path = "/Users/maeotaku/OneDrive/PhD_Research/Code/CIRAD_INRIA/PlantCLEFCrawler/"
destination_path = base_path + "Config/ASUHerbarium/Output/"
#base_path = "/home/carranza/code/PlantCLEFCrawler/"
#destination_path = "/opt/convnet_models/ASUHerbaria/"
occurrence_file_path = base_path + "Config/ASUHerbarium/Input/occurrences.csv"
images_file_path = base_path + "Config/ASUHerbarium/Input/images.csv"
identification_file_path = base_path + "Config/ASUHerbarium/Input/identifications.csv"

if __name__ == "__main__":
    dataset = ASUHerbariaPlantDataSet(  destination_path,
                                        occurrence_file_path,
                                        images_file_path,
                                        identification_file_path)
    crawler = DataSetDumper(dataset)
    crawler.generate()
'''

'''
#Costa Rica
base_path = "/Users/maeotaku/OneDrive/PhD_Research/Datasets/ALl_CR_Leaves/"
destination_path = "/Users/maeotaku/OneDrive/PhD_Research/Code/CIRAD_INRIA/PlantCLEFCrawler/Config/CRNationalMuseumLeavesDataSet/Output/"
#base_path = "/home/carranza/code/PlantCLEFCrawler/"
#destination_path = "/opt/convnet_models/ASUHerbaria/"
dataset = CRNationalMuseumLeavesDataSet(destination_path, base_path)
crawler = DataSetDumper(dataset)
crawler.generate()
'''
'''
#Hebaria_Matches_CR
#base_path = "/Users/maeotaku/OneDrive/PhD_Research/Code/CIRAD_INRIA/PlantCLEFCrawler/"
base_path = "/home/jcarranza/code/PlantCLEFCrawler/"
destination_path = base_path + "Config/Herbaria_Matches_PlantCLEF/Output2/"
occurrence_file_path = base_path + "Config/Herbaria_Matches_PlantCLEF/Input/occurrence.csv"
images_file_path = base_path + "Config/Herbaria_Matches_PlantCLEF/Input/multimedia.csv"
dataset = IDigBioDataSet(destination_path, occurrence_file_path, images_file_path)
crawler = DataSetDumper(dataset)
crawler.generate()
'''

#PlantVillage
base_path = "/Users/maeotaku/OneDrive/PhD_Research/Code/CIRAD_INRIA/PlantCLEFCrawler/"
destination_path = base_path + "Config/PlantVillage/Output/"
images_file_path = base_path + "Config/PlantVillage/Input/"
dataset = PlantVillageDataSet(destination_path, images_file_path)
crawler = DataSetDumper(dataset)
crawler.generate()

'''
def wtf():
    import csv
    with open("/Users/maeotaku/OneDrive/PhD_Research/Datasets/ParisHerbarium/occurrence.csv", "r") as f:
        cont=0
        datareader=csv.reader(f)
        with open("/Users/maeotaku/OneDrive/PhD_Research/Datasets/ParisHerbarium/occurrence_small.csv", "w") as f2:
            spamwriter = csv.writer(f2, delimiter=';', quoting=csv.QUOTE_MINIMAL)
            for row in datareader:
                spamwriter.writerow(row)
                cont+=1
                if (cont==100):
                    f2.close()
                    f.close()
                    return

if __name__ == "__main__":
    wtf()
'''
