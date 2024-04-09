import numpy as np
import pickle
import os


class LaminateStorage:
    """
    Class to store laminate data including ply IDs, coordinates, and angles.

    Attributes:
        ply_ids (list): List of ply IDs.
        coordinates (numpy.ndarray): Array of coordinates.
        angles (numpy.ndarray): Array of angles.

    Methods:
        add_ply_id: Add one or more ply IDs to the storage.
        add_coordinate: Add one or more sets of coordinates to the storage.
        add_angle: Add one or more angles to the storage.
        normalize_coordinates: Normalize the coordinates relative to the first set of coordinates.
        normalize_angles: Normalize the angles relative to the first angle.
        save_as_pickle: Save the laminate data as a pickle file.
        load_from_pickle: Load laminate data from a pickle file.
    """

    def __init__(self):
        self.ply_ids = []
        self.coordinates = np.array([])
        self.angles = np.array([])

    def add_ply_id(self, *args):
        """
        Add one or more ply IDs to the storage.

        Args:
            *args: One or more ply IDs.
        """
        self.ply_ids.extend(args)

    def add_coordinate(self, *args):
        """
        Add one or more sets of coordinates to the storage.

        Args:
            *args: One or more sets of coordinates.
        """
        if not self.coordinates.size:  # Initialize coordinates if empty
            self.coordinates = np.array(args)
        else:
            self.coordinates = np.vstack((self.coordinates, args))

    def add_angle(self, *args):
        """
        Add one or more angles to the storage.

        Args:
            *args: One or more angles.
        """
        if not self.angles.size:  # Initialize angles if empty
            self.angles = np.array(args)
        else:
            self.angles = np.vstack((self.angles, args))

    def normalize_coordinates(self):
        """Normalize the coordinates relative to the first set of coordinates."""
        self.coordinates -= self.coordinates[0]

    def normalize_angles(self):
        """Normalize the angles relative to the first angle."""
        self.angles -= self.angles[0]

    def save_as_pickle(self, filename):
        """
        Save the laminate data as a pickle file.

        Args:
            filename (str): Name of the pickle file to save.
        """
        data_to_save = {'ply_ids': self.ply_ids, 'coordinates': self.coordinates, 'angles': self.angles}
        with open(filename, 'wb') as file:
            pickle.dump(data_to_save, file)
        print(f"Data saved as pickle file: {filename}")

    @classmethod
    def load_from_pickle(cls, filename):
        """
        Load laminate data from a pickle file.

        Args:
            filename (str): Name of the pickle file to load.

        Returns:
            LaminateStorage: An instance of LaminateStorage with data loaded from the pickle file.
        """
        with open(filename, 'rb') as file:
            loaded_data = pickle.load(file)
        instance = cls()
        instance.ply_ids = loaded_data['ply_ids']
        instance.coordinates = np.array(loaded_data['coordinates'])
        instance.angles = np.array(loaded_data['angles'])
        return instance


if __name__ == '__main__':
    # Directory where the laminate data will be stored
    directory = 'laminates'
    # Filename for the pickle file
    filename = 2
    # Create an instance of LaminateStorage to store the data
    data_storage = LaminateStorage()
    # Add ply IDs, coordinates, and angles to the data storage
    data_storage.add_ply_id(6, 3, 8, 9)
    data_storage.add_coordinate((1164, 760), (824, 548), (507, 548), (1164, 760))
    data_storage.add_angle(-1.59, -1, -46.35, 0)
    # Normalize the coordinates and angles
    data_storage.normalize_coordinates()
    data_storage.normalize_angles()
    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    # Save the data storage as a pickle file
    data_storage.save_as_pickle(f"{directory}/{filename}.pickle")
    # Loading data from the saved pickle file
    loaded_data_storage = LaminateStorage.load_from_pickle(f"{directory}/{filename}.pickle")
    print("Loaded ply_ids:", loaded_data_storage.ply_ids)
    print("Loaded coordinates:", loaded_data_storage.coordinates[:])
    print("Angles", loaded_data_storage.angles)
