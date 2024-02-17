def get_pure_spectral(image_path, point_name, image_type):
    # Read geopackage data, also works with shapefile (.shp)
    gdf = gpd.read_file('/Users/tolgasabanoglu/Desktop/thesis_data/data/Pure_points/' + point_name + '.gpkg')

    # Get coordinates of each point
    # Make sure coordinates of points and satellite image are matched
    coord_list = [(x, y) for x, y in zip(gdf['geometry'].x, gdf['geometry'].y)]

    # Sampling spectral information to points
    with rasterio.open(image_path) as src:
        data = [x for x in src.sample(coord_list)]

    # Convert into 2D array (Rows are points, Columns are spectral bands)
    data = np.array(data)

    # Save data to CSV
    saving_path = '/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/Pure_spectral/' + point_name + '_' + image_type + '.csv'
    np.savetxt(saving_path, data, delimiter=',', fmt='%i')
    print(f'Saved {point_name} spectral for {image_type} at {saving_path}')

if __name__ == '__main__':
    sentinel2_path = '/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/sentinel_stm_dry.tif'
    spectral_image_path = '/Users/tolgasabanoglu/Desktop/geoinpython/termpaper/20230502_SPECTRAL_IMAGE.TIF'

    name_list = ['woody', 'herbaceous', 'soil']

    for point_name in name_list:
        get_pure_spectral(image_path=sentinel2_path, point_name=point_name, image_type='sentinel2')
        get_pure_spectral(image_path=spectral_image_path, point_name=point_name, image_type='enmap')
