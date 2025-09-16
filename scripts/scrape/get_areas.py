#!/usr/bin/env python3
import argparse, io, json, os, rasterio, re, yaml, zipfile
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, MultiPolygon
from typing import Optional, Union
from unidecode import unidecode
from tqdm import tqdm


# Helper function to calculate distances
def _kd_query(src_xy: np.ndarray, dst_xy: np.ndarray):
    """
    Return distances and indices for each src point's nearest neighbour in dst.
    """
    # Build tree and calculate distances
    tree = cKDTree(dst_xy)
    dists, j = tree.query(src_xy, k=1)
    return dists.astype(float), j.astype(int)


# Calculate distances to given variable indicator in geopandas
def get_distance_loc(gdf: gpd.GeoDataFrame, marker_col: str, id_col: str,
                     dist_name: str, distance_crs: Optional[Union[str, int]]=None,
                     return_nearest_id: bool = False) -> gpd.GeoDataFrame:
    """
    Fast centroid-to-centroid signed distance to the nearest opposite-type square.
    - Negative distance for marker not-null.
    - Positive distance for marker null.
    - Distances in kilometers.
    """

    # Check for CRS and CRS argument
    if gdf.crs is None:
        raise ValueError("gdf.crs is None. Set a CRS or pass distance_crs.")
    # Choose a metric CRS
    if distance_crs is None:
        if getattr(gdf.crs, "is_geographic", False):
            try:
                distance_crs = gdf.estimate_utm_crs()
            except Exception:
                distance_crs = 3857  # fallback (meters, approx.)
        else:
            distance_crs = gdf.crs

    # Project once and compute centroids
    g_metric = gdf.to_crs(distance_crs)
    cents    = g_metric.geometry.centroid
    
    # Stack column of centroid coordinates
    xy = np.column_stack([cents.x.to_numpy(), cents.y.to_numpy()])

    # Filter geopandas dataframe for those iniside and outside marker variable
    is_in   = gdf[marker_col].notna().to_numpy()
    is_out  = ~is_in
    idx_in  = np.where(is_in)[0]
    idx_out = np.where(is_out)[0]

    # Prepare outputs
    dist_km     = np.full(len(gdf), np.nan, dtype="float64")
    nearest_idx = np.full(len(gdf), -1, dtype=int)

    # If any compute
    if idx_in.size and idx_out.size:
        XA = xy[idx_in]
        XB = xy[idx_out]

        # For A get nearest B and vice-versa
        dA, jB = _kd_query(XA, XB)
        dB, jA = _kd_query(XB, XA)

        # Assign signed distances (km) and signage
        dist_km[idx_in]  = -(dA / 1000.0)
        dist_km[idx_out] = +(dB / 1000.0)

        # Get index of distances
        nearest_idx[idx_in]  = idx_out[jB]
        nearest_idx[idx_out] = idx_in[jA]

    # Build output on the original CRS/geometry
    out = gdf[[id_col]].copy()
    out[dist_name] = dist_km
    if return_nearest_id and (nearest_idx >= 0).any():
        out["nearest_id"] = gdf.iloc[nearest_idx.clip(min=0)][id_col].to_numpy()
        out.loc[nearest_idx < 0, "nearest_id"] = pd.NA
    return out


# Create dictionary with neighboring municipalities
def create_neighbors_dict(df: pd.DataFrame, id_col: str, save_path: str) -> dict:
    """
    Create a dictionary with neighboring municipalities.
    """
    # Perform a spatial join to find intersecting polygons
    df_borders = gpd.sjoin(df, df, how="inner", predicate="intersects")
    df_borders = df_borders.query(f"{id_col}_left!={id_col}_right").rename(columns={f"{id_col}_left": id_col, f"{id_col}_right": "neighbor_ids"})[[id_col, "neighbor_ids"]]
    df_borders = df_borders.groupby([id_col], as_index=False)["neighbor_ids"].agg(list).reset_index(drop=True)

    # Create dictionary with data and save
    borders_dict = {df_borders.loc[i, id_col]: df_borders.loc[i, "neighbor_ids"] for i in df_borders.index}
    with open(f"{save_path}", "w") as json_file:
        json.dump(borders_dict, json_file, indent=4)


# Main function to download data from GFC and join with others
def main():
    """
    Main function to download data from GFC and organize with other sources.
    """
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Downloads data from GFC, creates square areas and combines with other sources")
    ap.add_argument("--config", default="config/scrape.yaml", help="Path to YAML with scraping schema")
    ap.add_argument("--data_path", default="data", help="Path to data folders")
    ap.add_argument("--shapefiles_path", default="data/shapefiles", help="Path to other shapefiles")
    ap.add_argument("--output_path", default="data/shapefiles/gfc", help="Path to where to save data locally")
    args = ap.parse_args()

    # Load scrape schema
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Get relevant variables from config
    start_year      = config["params"]["start_year"]
    end_year        = config["params"]["end_year"]
    header          = {"User-Agent": config["params"]["header"]}
    skip_size       = config["params"]["skip_size"]
    amazon_granules = config["granules"]["amazon"]
    projected_crs   = config["params"]["projected_crs"]

    # Get url paths
    gfc_loss_year_url  = config["urls"]["gfc_loss_year"]
    gfc_tree_cover_url = config["urls"]["gfc_tree_cover"]
    gfc_datamask_url   = config["urls"]["gfc_datamask"]

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Read data for Amazon municipalities
    df_amazon  = pd.read_excel(f"{args.data_path}/ibge/Municipios_da_Amazonia_Legal_2022.xlsx", usecols=["CD_MUN", "SIGLA_UF"], converters={"CD_MUN": str}).rename(columns={"CD_MUN": "municipality_id", "SIGLA_UF": "state"})
    amazon_ids = df_amazon.municipality_id.unique().tolist()

    # Read Amazon municipality shapefile
    df_mun          = gpd.read_file(f"{args.shapefiles_path}/municipalities_amazon/municipalities_legal_amazon.shp").rename(columns={"nome": "municipality_name", "geocodigo": "municipality_id"})
    df_mun          = pd.merge(df_mun[["municipality_id", "municipality_name", "geometry"]], df_amazon, on="municipality_id", how="left")
    df_mun["state"] = df_mun.apply(lambda x: x.state if x.municipality_name not in ["Passagem Franca"] else "MA", axis=1)
    df_mun          = df_mun[["state", "municipality_id", "municipality_name", "geometry"]]
    df_mun["mun_geometry"] = df_mun["geometry"]
 
    # Read and organize indigenous territory data
    df_tis = gpd.read_file(f"{args.shapefiles_path}/terras_indigenas/tis_poligonais_portariasPolygon.shp").rename(columns={"terrai_cod": "ti_id", "terrai_nom": "ti_name", "etnia_nome": "ethnicity_name", "municipio_": "municipality_name_ti", "fase_ti": "judicial_status", "modalidade": "occupation_type", "uf_sigla": "state"})
    
    # Get year for date variables
    for v in ["data_em_es", "data_delim", "data_decla", "data_homol", "data_regul"]:
        df_tis[v] = df_tis[v].dt.year

    # Determine data when becomes indigenous land, filter variables and create auxiliary geometry
    df_tis["year_ti"] = df_tis.apply(lambda x: min(x.data_homol, x.data_regul) if x.data_homol==x.data_homol or x.data_regul==x.data_regul else np.nan, axis=1)
    df_tis = df_tis[["ti_id", "ti_name", "ethnicity_name", "judicial_status", "municipality_name_ti", "year_ti", "geometry"]]
    df_tis["ti_geometry"] = df_tis["geometry"]
    tis_vars = ["ti_id", "ti_name", "ethnicity_name", "judicial_status", "municipality_name_ti", "year_ti"]

    # Conservation unit data
    df_ucs   = gpd.read_file("./data/shapefiles/prodes/conservation_units_legal_amazon.shp")
    df_ucs["year_uc"]     = df_ucs["ano_cria"]
    df_ucs["uc_geometry"] = df_ucs["geometry"]
    df_ucs   = df_ucs[["year_uc", "uc_geometry", "geometry"]]
    ucs_vars = ["year_uc"]

    # Read aldeia and sedes data
    df_alds  = gpd.read_file("./data/shapefiles/aldeias/aldeias_pontosPoint.shp")
    df_alds["aldeia"] = 1
    df_sedes = gpd.read_file("./data/shapefiles/municipalities_amazon/Sedes_Mun_Amazonia_Legal_2022.shp")
    df_sedes["sede"]  = 1

    # Create and save neighbors dictionary
    borders_dict = create_neighbors_dict(df_mun, "municipality_id", f"{args.data_path}/clean/municipality_neighbors.json")

    # Housekeeping, main variables to order
    main_vars = ["area_id", "cluster1_small_id", "cluster2_small_id", "cluster1_large_id", "cluster2_large_id", "latitude", "longitude", "state", "municipality_id", "municipality_name", "cover2000"]
    
    # Initiate lists to hold shapes and data
    data_lst           = []
    cluster1_small_lst = []
    cluster2_small_lst = []
    cluster1_large_lst = []
    cluster2_large_lst = []

    # Loop over Amazon granules
    for granule in tqdm(amazon_granules, desc="Downloading GFC data"):
        
        # Read GFC tree cover data
        with rasterio.open(gfc_tree_cover_url.format(GRANULE=granule)) as src:
            tree_data   = src.read(1)
            tree_transf = src.transform
            tree_crs    = src.crs
            rows, cols  = tree_data.shape

        # Initiate dictionary to hold data
        data_dict = {"area_id"    : [],
                     "pixels"     : [],
                     "tree_cover" : [],
                     "geometry"   : []}

        # Set iteration for rows and columns to populate dictionary
        for i in np.arange(0, rows, skip_size):
            for j in np.arange(0, cols, skip_size):

                # Get filtered deforestation, area id and polygon as centroid
                df_year = tree_data.T[i:i+skip_size, j:j+skip_size]
                area_id = f"{granule}_" + "{:05d}".format(i)+"{:05d}".format(i+skip_size)+"{:05d}".format(j)+"{:05d}".format(j+skip_size)
                polygon = Polygon([tree_transf * (i, j), tree_transf * (i+skip_size, j), tree_transf * (i+skip_size, j+skip_size), tree_transf * (i, j+skip_size)])

                # Get matrices for masks
                df_mask = tree_data.T[i:i+skip_size, j:j+skip_size]

                # Append data do dictionary
                data_dict["area_id"].append(area_id)
                data_dict["pixels"].append(df_mask.shape[0] * df_mask.shape[1])
                data_dict["tree_cover"].append(df_year.sum())
                data_dict["geometry"].append(polygon)
        
        # Initiate dictionary to hold cluster
        for ll, cluster, size in zip([cluster1_small_lst, cluster2_small_lst, cluster1_large_lst, cluster2_large_lst], [1, 2, 1, 2], ["small", "small", "large", "large"]):
            
            # Initiate dictionary to hold cluster
            cluster_dict = {f"cluster{cluster}_{size}_id": [],
                            "pixels"                     : [],
                            "geometry"                   : []}

            # Define skip factor and first row
            skip_factor = 10 if size == "small" else 15
            skip_factor = skip_factor * skip_size
            init_row    = 0 if cluster==1 else int(skip_factor / 2)
            
            # Set iteration for rows and columns to populate dictionary
            for i in np.arange(init_row, rows, skip_factor):
                for j in np.arange(init_row, cols, skip_factor):
                    
                    # Get filtered deforestation, area id and polygon as centroid
                    cluster_id = f"cluster{cluster}_{size}_{granule}_" + "{:05d}".format(i)+"{:05d}".format(i+skip_factor)+"{:05d}".format(j)+"{:05d}".format(j+skip_factor)
                    polygon    = Polygon([tree_transf * (i, j), tree_transf * (i+skip_factor, j), tree_transf * (i+skip_factor, j+skip_factor), tree_transf * (i, j+skip_factor)])

                    # Get matrices for masks
                    df_mask = tree_data.T[i:i+skip_factor, j:j+skip_factor]

                    # Append data do dictionary
                    cluster_dict[f"cluster{cluster}_{size}_id"].append(cluster_id)
                    cluster_dict["pixels"].append(df_mask.shape[0] * df_mask.shape[1])
                    cluster_dict["geometry"].append(polygon)

            # Append to lists
            ll.append(gpd.GeoDataFrame(pd.DataFrame(cluster_dict), crs=projected_crs))
           
        # Append geopandas to lists
        data_lst.append(gpd.GeoDataFrame(pd.DataFrame(data_dict), crs=projected_crs))
    
    # Join to get data
    df_geo = gpd.GeoDataFrame(pd.concat(data_lst, ignore_index=True).reset_index(drop=True), crs=projected_crs)
    df_cluster1_small = gpd.GeoDataFrame(pd.concat(cluster1_small_lst, ignore_index=True).reset_index(drop=True), crs=projected_crs)
    df_cluster2_small = gpd.GeoDataFrame(pd.concat(cluster2_small_lst, ignore_index=True).reset_index(drop=True), crs=projected_crs)
    df_cluster1_large = gpd.GeoDataFrame(pd.concat(cluster1_large_lst, ignore_index=True).reset_index(drop=True), crs=projected_crs)
    df_cluster2_large = gpd.GeoDataFrame(pd.concat(cluster2_large_lst, ignore_index=True).reset_index(drop=True), crs=projected_crs)

    # Create tree cover variable and latitude longitude
    df_geo["cover2000"] = (df_geo["tree_cover"] / df_geo["pixels"]) / 100
    df_geo["latitude"]  = df_geo.to_crs("EPSG:3857").geometry.centroid.y
    df_geo["longitude"] = df_geo.to_crs("EPSG:3857").geometry.centroid.x
    df_geo              = df_geo[["area_id", "latitude", "longitude", "cover2000", "geometry"]]

    # Go over geopandas dataframe to join with main
    for df_ in [df_cluster1_small, df_cluster2_small, df_cluster1_large, df_cluster2_large]:

        # Get cluster id column and create auxiliary geometry
        cluster_col = [i for i in df_.columns if "cluster" in i][0]
        df_["cluster_geometry"] = df_["geometry"]

        # Join with main dataframe and get overlap
        df_ = gpd.sjoin(df_geo[["area_id", "geometry"]].drop_duplicates(), df_.to_crs(crs=projected_crs), how="left", predicate="intersects")
        df_["overlap_area"] = df_.apply(lambda x: x["geometry"].intersection(x["cluster_geometry"]).area / x["geometry"].area if x["cluster_geometry"] is not None else 0, axis=1)
        df_ = df_.sort_values(by=["area_id", "overlap_area"], ascending=False).drop_duplicates(subset="area_id", keep="first")
        df_ = df_[["area_id", cluster_col]]

        # Join back with main dataframe
        df_geo = df_geo.merge(df_, on=["area_id"], how="left")
        df_geo = df_geo[sorted(df_geo.columns)]

    # Join with municipality data - filter for cells in municipalities
    df_geo = gpd.sjoin(df_geo, df_mun.to_crs(crs=projected_crs), how="left", predicate="intersects").query("municipality_id==municipality_id")
    df_geo["overlap_area"] = df_geo.apply(lambda x: x["geometry"].intersection(x["mun_geometry"]).area / x["geometry"].area if x["mun_geometry"] is not None else 0, axis=1)
    df_geo = df_geo.sort_values(by=["area_id", "overlap_area"], ascending=False).drop_duplicates(subset="area_id", keep="first")
    df_geo = df_geo[main_vars + ["geometry"]]
    df_geo = df_geo.query("municipality_id==municipality_id")

    # Join with TI data
    df_geo = gpd.sjoin(df_geo, df_tis.to_crs(crs=projected_crs), how="left", predicate="intersects")
    df_geo["overlap_area"] = df_geo.apply(lambda x: x["geometry"].intersection(x["ti_geometry"]).area / x["geometry"].area if x["ti_geometry"] is not None else 0, axis=1)
    df_geo = df_geo.sort_values(by=["area_id", "overlap_area"], ascending=False).drop_duplicates(subset="area_id", keep="first")
    df_geo = df_geo[main_vars + tis_vars + ["geometry"]]

    # Join with UC data
    df_geo = gpd.sjoin(df_geo, df_ucs.to_crs(crs=projected_crs), how="left", predicate="intersects")
    df_geo["overlap_area"] = df_geo.apply(lambda x: x["geometry"].intersection(x["uc_geometry"]).area / x["geometry"].area if x["uc_geometry"] is not None else 0, axis=1)
    df_geo = df_geo.sort_values(by=["area_id", "overlap_area"], ascending=False).drop_duplicates(subset="area_id", keep="first")
    df_geo = df_geo[main_vars + tis_vars + ucs_vars + ["geometry"]]
    
    # Get distances to regions
    df_tis_dist = get_distance_loc(df_geo, "year_ti", "area_id", "dist_ti", distance_crs="EPSG:3857")
    df_ucs_dist = get_distance_loc(df_geo, "year_uc", "area_id", "dist_uc", distance_crs="EPSG:3857")

    # Organize dataframes
    df_areas = df_geo[main_vars + tis_vars + ucs_vars].copy()
    df_geo   = df_geo[["area_id", "geometry"]]

    # Get aldeias and sede data per area id
    df_alds = gpd.sjoin(df_geo.drop_duplicates(), df_alds.to_crs(crs=projected_crs), predicate="contains", how="left")
    df_ald_dist = get_distance_loc(df_alds.assign(aldeia=df_alds["aldeia"].replace(0, np.nan)), "aldeia", "area_id", "dist_aldeia", distance_crs="EPSG:3857")
    df_alds = df_alds[["area_id", "aldeia"]].groupby("area_id", as_index=False).sum()
    df_sedes = gpd.sjoin(df_geo[["area_id", "geometry"]].drop_duplicates(), df_sedes.to_crs(crs=projected_crs), predicate="contains", how="left")
    df_sedes = df_sedes[["area_id", "sede"]].groupby("area_id", as_index=False).sum()

    # Merge back to main data
    df_areas = df_areas.merge(df_alds, on=["area_id"], how="left")
    df_areas = df_areas.merge(df_ald_dist, on=["area_id"], how="left")
    df_areas = df_areas.merge(df_sedes, on=["area_id"], how="left")
    df_areas = df_areas.merge(df_tis_dist, on=["area_id"], how="left")
    df_areas = df_areas.merge(df_ucs_dist, on=["area_id"], how="left")

    # Save data
    df_geo[["area_id", "geometry"]].to_file(f"{args.output_path}/gfc.shp")
    df_areas.to_parquet(f"{args.data_path}/clean/gfc_areas.parquet", index=False, engine="pyarrow")
    
    #print(df_geo.shape)
    #print(df_geo.head())

# Run script directly
if __name__ == "__main__":
    main()