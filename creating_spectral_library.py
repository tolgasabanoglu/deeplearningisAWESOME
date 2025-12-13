def create_mixing_recipe(satellite):
    # Define paths based on the selected satellite
    if satellite == 'enmap':
        woody_path = '/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/Pure_spectral/woody_enmap.csv'
        herbaceous_path = '/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/Pure_spectral/herbaceous_enmap.csv'
        soil_path = '/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/Pure_spectral/soil_enmap.csv'
    elif satellite == 'sentinel2':
        woody_path = '/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/Pure_spectral/woody_sentinel2.csv'
        herbaceous_path = '/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/Pure_spectral/herbaceous_sentinel2.csv'
        soil_path = '/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/Pure_spectral/soil_sentinel2.csv'
    else:
        raise ValueError("Invalid satellite type. Choose 'enmap' or 'sentinel2'.")

    # Create endmember dictionary
    pure_spectral_dict = {
        'woody': np.genfromtxt(woody_path, delimiter=','),
        'herbaceous': np.genfromtxt(herbaceous_path, delimiter=','),
        'soil': np.genfromtxt(soil_path, delimiter=',')
    }

    # List of endmember names
    land_cover_keys = list(pure_spectral_dict.keys())

    # The higher the number, the likely the land cover will be drafted
    land_cover_weights = [2, 2, 2]

    # The number of mixed spectral data to generate
    number_of_sample = 30000 

    mixed_spectral_data = []
    label_data = []

    # For each mixed spectral sample
    for _ in tqdm(range(number_of_sample)):
        # Mixed spectral sample starts with 0
        x = 0
        
        # Fraction label sample start with [0, 0, 0, 0, 0]
        y = np.array([0, 0, 0], np.float32)
        
        # How many number of mixture
        mixture_number = random.choice([1, 2, 3])
        
        # Which land cover type will be drafed
        selected_land_cover = random.choices(land_cover_keys, k=mixture_number, weights=land_cover_weights)
        
        # Create random fraction
        # If 2 land cover mixtures, it will generate something like [0.2, 0.8]
        # If 3 land cover mixtures, it will generate something like [0.2, 0.3, 0.5]
        random_fraction = np.random.dirichlet(np.ones(len(selected_land_cover)),size=1)[0]
        
        # For each land cover in the mixture
        for i in range(len(selected_land_cover)):
            # Which index of that land cover
            index = random.randrange(pure_spectral_dict[selected_land_cover[i]].shape[0])
            
            # What is this land cover index
            land_cover_index = land_cover_keys.index(selected_land_cover[i])
            
            # Linearly mix of the mixture
            x += pure_spectral_dict[selected_land_cover[i]][index] * random_fraction[i]
            y[land_cover_index] += random_fraction[i]
        
        # Add the mixed spectral sample and its label to the dataset
        mixed_spectral_data.append(x)
        label_data.append(y)
        
    # Convert to array and save
    mixed_spectral_data = np.asarray(mixed_spectral_data, dtype=np.float32)
    label_data = np.asarray(label_data, dtype=np.float32)

    # Save files based on the selected satellite
    if satellite == 'enmap':
        np.save('/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/Mixed_spectral/mixed_spectral_enmap.npy', arr=mixed_spectral_data)
        np.save('/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/Mixed_spectral/label_enmap.npy', arr=label_data)
    elif satellite == 'sentinel2':
        np.save('/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/Mixed_spectral/mixed_spectral_sentinel2.npy', arr=mixed_spectral_data)
        np.save('/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/Mixed_spectral/label_sentinel2.npy', arr=label_data)

if __name__ == '__main__':
    create_mixing_recipe(satellite='enmap')  # For Enmap
    create_mixing_recipe(satellite='sentinel2')  # For Sentinel-2
