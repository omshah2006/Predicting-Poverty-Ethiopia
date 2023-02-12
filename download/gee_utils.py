import ee
import pandas as pd


def create_fc(df):
    features = []
    for i in range(len(df)):
        properties = df.iloc[i].to_dict()
        # lon, lat
        point = ee.Geometry.Point(
            [
                df.iloc[i]["lon"],
                df.iloc[i]["lat"],
            ]
        )

        feature = ee.Feature(point, properties)
        features.append(feature)

    return ee.FeatureCollection(features)


def export_tiles(collection, export, prefix, fname, selectors, dropselectors, bucket):
    if dropselectors is not None:
        if selectors is None:
            selectors = collection.first().propertyNames()

        selectors = selectors.removeAll(dropselectors)

    if export == "gcs":
        task = ee.batch.Export.table.toCloudStorage(
            collection=collection,
            description=fname,
            bucket=bucket,
            fileNamePrefix=f"{prefix}/{fname}",
            fileFormat="TFRecord",
            selectors=selectors,
        )

    elif export == "drive":
        task = ee.batch.Export.table.toDrive(
            collection=collection,
            description=fname,
            folder=prefix,
            fileNamePrefix=fname,
            fileFormat="TFRecord",
            selectors=selectors,
        )

    else:
        raise ValueError(f'export "{export}" is not one of ["gcs", "drive"]')

    task.start()

    return task


def make_sample_tile(point, patches_array, scale):
    arrays_samples = patches_array.sample(
        region=point.geometry(),
        scale=scale,
        projection="EPSG:3857",
        factor=None,
        numPixels=None,
        dropNulls=False,
        tileScale=12,
    )
    return arrays_samples.first().copyProperties(point)


def make_tiles(
    img,
    scale,
    ksize,
    points,
    export,
    prefix,
    fname,
    selectors=None,
    dropselectors=None,
    bucket=None,
):
    kern = ee.Kernel.square(radius=ksize, units="pixels")
    patches_array = img.neighborhoodToArray(kern)

    samples = points.map(lambda pt: make_sample_tile(pt, patches_array, scale))

    return export_tiles(
        collection=samples,
        export=export,
        prefix=prefix,
        fname=fname,
        selectors=selectors,
        dropselectors=dropselectors,
        bucket=bucket,
    )


def mask_l8_sr(img):
    # Bit 0 - Fill
    # Bit 1 - Dilated Cloud
    # Bit 2 - Cirrus
    # Bit 3 - Cloud
    # Bit 4 - Cloud Shadow
    qa_mask = img.select("QA_PIXEL").bitwiseAnd(int("11111", 2)).eq(0)
    sat_mask = img.select("QA_RADSAT").eq(0)

    opt_bands = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    therm_bands = img.select("ST_B.*").multiply(0.00341802).add(149.0)

    return (
        img.addBands(opt_bands, None, True)
        .addBands(therm_bands, None, True)
        .updateMask(qa_mask)
        .updateMask(sat_mask)
    )


def add_nl_band(img):
    nighttime_lights_col = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
        .filter(ee.Filter.date("2018-01-01", "2018-12-31"))
        .median()
    )
    img = img.addBands(nighttime_lights_col.select("avg_rad").rename("VIIRS"))

    return img


def add_deltatemp_band(img):
    l8_temp_col = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filter(ee.Filter.date("2017-01-01", "2018-12-31"))
        .median()
    )

    l5_temp_col = (
        ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
        .filter(ee.Filter.date("1985-01-01", "1986-12-31"))
        .median()
    )

    img = img.addBands(l8_temp_col.select("ST_B10").subtract(l5_temp_col.select("ST_B6")).rename("DELTA_TEMP"))

    return img


def add_latlon_band(img):
    latlon_img = ee.Image.pixelLonLat().select(
        ["longitude", "latitude"], ["LON", "LAT"]
    )
    img = img.addBands(latlon_img)

    return img


def create_ms_img(geometry):
    l8_imgcol = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l8_img = l8_imgcol.filterDate("2017-01-01", "2019-12-31").filterBounds(geometry)
    composite = l8_img.map(mask_l8_sr).median()

    landsat_bands = [
        "SR_B1",
        "SR_B2",
        "SR_B3",
        "SR_B4",
        "SR_B5",
        "SR_B6",
        "SR_B7",
        "ST_B10",
        "QA_PIXEL",
    ]
    renamed_bands = [
        "AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "NIR",
        "SW_IR1",
        "SW_IR2",
        "TEMP",
        "QA_PIXEL",
    ]

    composite = composite.select(landsat_bands)
    composite = composite.rename(renamed_bands)

    return composite
