import pickle
import numpy as np


class LaminateStorage:
    def __init__(self):
        self.ply_ids = []
        self.coordinates = np.array([])
        self.angles = np.array([])

    def add_ply_id(self, *args):
        self.ply_ids.extend(args)

    def add_coordinate(self, *args):
        if not self.coordinates.size:  # Initialize coordinates if empty
            self.coordinates = np.array(args)
        else:
            self.coordinates = np.vstack((self.coordinates, args))

    def add_angle(self, *args):
        if not self.angles.size:  # Initialize coordinates if empty
            self.angles = np.array(args)
        else:
            self.angles = np.vstack((self.angles, args))

    def normalize_coordinates(self):
        self.coordinates -= self.coordinates[0]

    def normalize_angles(self):
        self.angles -= self.angles[0]

    def save_as_pickle(self, filename):
        data_to_save = {'ply_ids': self.ply_ids, 'coordinates': self.coordinates, 'angles': self.angles}
        with open(filename, 'wb') as file:
            pickle.dump(data_to_save, file)
        print(f"Data saved as pickle file: {filename}")

    @classmethod
    def load_from_pickle(cls, filename):
        with open(filename, 'rb') as file:
            loaded_data = pickle.load(file)
        instance = cls()
        instance.ply_ids = loaded_data['ply_ids']
        instance.coordinates = np.array(loaded_data['coordinates'])
        instance.angles = np.array(loaded_data['angles'])
        return instance


if __name__ == '__main__':
    # Example usage:
    data_storage = LaminateStorage()
    data_storage.add_ply_id(6, 2, 8)
    data_storage.add_coordinate((1348, 811), (1485, 850), (1525, 840))
    data_storage.add_angle(2.7, -35.4, -45.8)
    data_storage.normalize_coordinates()
    data_storage.normalize_angles()
    data_storage.save_as_pickle("laminates/1.pickle")

    # Loading data from pickle file
    loaded_data_storage = LaminateStorage.load_from_pickle("laminates/1.pickle")
    print("Loaded ply_ids:", loaded_data_storage.ply_ids)
    print("Loaded coordinates:", loaded_data_storage.coordinates[:])
    print("Angles", loaded_data_storage.angles)
