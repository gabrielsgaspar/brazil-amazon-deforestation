#!/usr/bin/env python3
import argparse, io, json, os, rasterio, re, yaml, zipfile
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
from typing import Optional, Union
from unidecode import unidecode
from tqdm import tqdm


# Main function to download deforestation data from GFC
def main():
    """
    Main function to download deforestation data from GFC.
    """
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Downloads deforestation data from GFC")
    ap.add_argument("--config", default="config/scrape.yaml", help="Path to YAML with scraping schema")
    ap.add_argument("--data_path", default="data", help="Path to data folders")
    ap.add_argument("--shapefiles_path", default="data/shapefiles", help="Path to other shapefiles")
    ap.add_argument("--output_path", default="data/clean", help="Path to where to save data locally")
    args = ap.parse_args()

    # Load scrape schema
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Get relevant variables from config
    start_year      = int(config["params"]["start_year"])
    end_year        = int(config["params"]["end_year"])
    header          = {"User-Agent": config["params"]["header"]}
    skip_size       = config["params"]["skip_size"]
    amazon_granules = config["granules"]["amazon"]
    projected_crs   = config["params"]["projected_crs"]

    # Get url paths
    gfc_loss_year_url  = config["urls"]["gfc_loss_year"]
    gfc_tree_cover_url = config["urls"]["gfc_tree_cover"]
    gfc_datamask_url   = config["urls"]["gfc_datamask"]

    # Read clean data to get ids of areas to keep
    df_clean = pd.read_parquet(f"{args.output_path}/gfc_areas.parquet")
    df_clean = df_clean[["area_id"]]
    
    # Housekeeping, main variables to order
    main_vars = ["area_id", "year", "pixels", "defor", "no_data", "water"]
    
    # Initiate lists to hold shapes and data
    data_lst = []
    
    # Loop over Amazon granules
    for granule in tqdm(amazon_granules, desc="Downloading GFC data"):
        
        # Read GFC deforestation
        with rasterio.open(gfc_loss_year_url.format(GRANULE=granule)) as src:
            defor_data   = src.read(1)
            defor_transf = src.transform
            defor_crs    = src.crs
            rows, cols   = defor_data.shape

        # Read GFC datamask
        with rasterio.open(gfc_datamask_url.format(GRANULE=granule)) as src:
            data_mask = src.read(1)

        # Initiate dictionary to hold data
        data_dict = {"area_id"    : [],
                     "year"       : [],
                     "pixels"     : [],
                     "defor"      : [],
                     "no_data"    : [],
                     "water"      : [],
                     "geometry"   : []}

        # Set iteration for rows and columns to populate dictionary
        for i in np.arange(0, rows, skip_size):
            for j in np.arange(0, cols, skip_size):

                # Get filtered deforestation, area id and polygon as centroid
                df_year = defor_data.T[i:i+skip_size, j:j+skip_size]
                area_id = f"{granule}_" + "{:05d}".format(i)+"{:05d}".format(i+skip_size)+"{:05d}".format(j)+"{:05d}".format(j+skip_size)
                polygon = Polygon([defor_transf * (i, j), defor_transf * (i+skip_size, j), defor_transf * (i+skip_size, j+skip_size), defor_transf * (i, j+skip_size)])

                # Get matrices for masks
                df_mask = data_mask.T[i:i+skip_size, j:j+skip_size]

                # Iterate over years
                for year in np.arange(start_year, end_year+1, 1):

                    # Filter deforestation for year and multiply by tree cover
                    df_year_ = np.where(df_year!=year-2000, 0, df_year)
                    df_year_ = np.where(df_year_==year-2000, 1, df_year_)

                    # Append data do dictionary
                    data_dict["area_id"].append(area_id)
                    data_dict["year"].append(year)
                    data_dict["pixels"].append(df_mask.shape[0] * df_mask.shape[1])
                    data_dict["defor"].append(df_year_.sum())
                    data_dict["no_data"].append(np.where(df_mask==0, 1, 0).sum())
                    data_dict["water"].append(np.where(df_mask==2, 1, 0).sum())
                    data_dict["geometry"].append(polygon)
        
        # Append geopandas to lists
        data_lst.append(gpd.GeoDataFrame(pd.DataFrame(data_dict), crs=projected_crs))

    # Concatenate to get data and keep relevant ids
    df_defor = gpd.GeoDataFrame(pd.concat(data_lst, ignore_index=True).reset_index(drop=True), crs=projected_crs)
    df_defor = df_clean.merge(df_defor, on=["area_id"], how="left").query("year==year")
    df_defor = df_defor[main_vars]
    
    # Make relevant variables ints
    for v in ["year", "pixels", "defor", "no_data", "water"]:
        df_defor[v] = df_defor[v].astype(int)

    # Save data
    df_defor.to_parquet(f"{args.output_path}/gfc_defor.parquet", index=False, engine="pyarrow")
    
    #print(df_geo.shape)
    #print(df_geo.head())

# Run script directly
if __name__ == "__main__":
    main()