import os
import geopandas as gpd
import pandas as pd
import numpy as np
import contextily as cx
import rtree
import logging
from visualisation import CartoFeatureVis
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import joblib
from math import ceil
import pyproj
from shapely.geometry import shape
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union
# from torchvision.io import read_image
import matplotlib.pyplot as plt

from scipy.spatial import distance


from typing import Optional

# Residential class mapping
res_code_to_class = {
    1: "single-house",
    2: "multi-unit-occupancy",
    3: "single-house",
    4: "cluster",
    5: "low-rise-apartments",
    6: "high-rise-apartments",
    7: "other-urban",
    8: "vegetation",
    9: "bare-earth",
    11: "rural"
}


def map_res_code_to_class(df):
    df["res_class"] = df["res_code"].map(res_code_to_class)
    return df

def dask_within_operation(ddf, aoi):
    """
    Perform a within operation on a Dask GeoDataFrame based on an area of interest (AOI) polygon.
    
    Parameters:
        - ddf (dask_geopandas.GeoDataFrame): Dask GeoDataFrame containing targeted polygons.
        - aoi (geopandas.GeoDataFrame: AOI polygon
        
    Returns:
        - GeoDataFrame: GeoDataFrame containing geometries within the AOI polygon.
    """    
    
    # Transform the CRS of the Dask GeoDataFrame
    ddf = ddf.to_crs(aoi.crs)
    
    # Use the within function to find geometries within the AOI polygon
    within_condition = ddf.within(aoi.unary_union)
    
    # Subset the original Dask GeoDataFrame based on the within condition
    ddf_aoi = ddf[within_condition]
    
    # Trigger computation
    ddf_aoi = ddf_aoi.compute()
    
    return ddf_aoi

def dask_intersect_operation(ddf, aoi):
    """
    Perform an intersection operation on a Dask GeoDataFrame based on an area of interest (AOI) polygon.
    
    Parameters:
        - ddf (dask_geopandas.GeoDataFrame): Dask GeoDataFrame containing targeted polygons.
        - aoi (geopandas.GeoDataFrame): AOI polygon.
        
    Returns:
        - GeoDataFrame: GeoDataFrame containing geometries that intersect with the AOI polygon.
    """
    
    # Transform the CRS of the Dask GeoDataFrame to match the AOI's CRS
    ddf = ddf.to_crs(aoi.crs)
    
    # Use the intersects function to find geometries that intersect with the AOI polygon
    intersect_condition = ddf.intersects(aoi.unary_union)
    
    # Subset the original Dask GeoDataFrame based on the intersect condition
    ddf_aoi = ddf[intersect_condition]
    
    # Trigger computation
    ddf_aoi = ddf_aoi.compute()
    
    return ddf_aoi

    
def generate_lot_type_dict(row):
    """
    Generate dictionaries to store lot types for each row.
    """
    # Define lot types and initialize lot type dictionary
    lot_types = ["OTHER", "STPLN", "CROWN", "ADMIN", "SSPLN", "LEASE", "RESVE", "FHOLD", "SVEXT"]
    lot_type_dict = {lot_type: 0 for lot_type in lot_types}
    
    # Increment the count for the lot type of the cadastre
    lot_type_dict[row["lot_type"]] += 1
    
    # Return both dictionaries
    return {"lot_type_dict": lot_type_dict}


def update_pin_list_and_dicts_for_overlapping_polygons(gdf, index, row, pin_col_name:str, pin_list_col_name: str, overlapping_rows):
    """
    Update list of pins and lot type dictionaries for overlapping polygons.
    
    Parameters:
        - gdf: Input GeoDataFrame containing cadastre polygons.
        - index: Index of the current row being processed.
        - row: Current row being processed.
        - pin_col_name: Name of the column that is used to store pin (Cadastre datasets may have different pin column name)
        - pin_list_col_name: Name of the column for storing the list of pins of overlapping polygons
        - overlapping_rows: DataFrame containing overlapping polygons.
    """
    
    # Extract pins of overlapping polygons to a list
    pin_list = overlapping_rows[pin_col_name].tolist()

    # Assign the list of pins to the pin list column for the current row
    gdf.at[index, pin_list_col_name] = str(pin_list)
    
    # Initialize dictionaries to count lot types in sibling polygons
    lot_type_dict_updates = row["lot_type_dict"].copy()

    # Iterate over overlapping rows to count lot types
    for _, overlap_row in overlapping_rows.iterrows():
        
        # Increment the count for the lot type of the cadastre
        lot_type_dict_updates[overlap_row["lot_type"]] += 1

    # Update the lot_type_dict in the DataFrame
    gdf.at[index, "lot_type_dict"] = lot_type_dict_updates


def flatten_cadastre_2023(cad_gdf):
    """
    Create additional columns to preserve pins and lot type attributes of overlapping 
    cadastres that will be removed. First, preserve attributes of 'sibling' polygons (same geometry) 
    in one of the sibling polygon, before removing the rest from the GeoDataFrame. Next, identify 
    the 'parent' polygons, which superimpose smaller 'child' polygons, and preserve attributes of 
    the child polygons in the parent polygons, before removing them. Need to remove sibling polygons 
    first to avoid any two polygons of same size overlapping child polygons. 
    
    Note: Use this for Perth cadastre 2023 dataset
    
    Parameters:
        - cad_gdf: Input GeoDataFrame containing cadastre polygons.

    Returns:
        - Flattened GeoDataFrame with retained attributes.
    """
    
    # Make a copy of the input GeoDataFrame
    gdf = cad_gdf.copy()
    
    # Rename the land_type column to lot_type
    gdf = gdf.rename(columns={"land_type": "lot_type"})

    # Create dictionary columns for storing lot type data
    gdf[["lot_type_dict"]] = gdf.apply(generate_lot_type_dict, axis=1, result_type="expand")
    
    # -----------------------------------------------------
    # Flatten and preserve attributes of 'sibling polygons'
    # -----------------------------------------------------
    
    # Iterate over rows
    for index, row in gdf.iterrows():
        
        # Get all polygons with the same geometry as the current row
        same_geom_rows = gdf[gdf.geometry == row.geometry]
        
        # Exclude the current row being processed
        same_geom_rows = same_geom_rows[same_geom_rows.index != index]
        
        # Check if the row has any sibling polygons
        if not same_geom_rows.empty:
            
            # Preserve sibling polygons' pins, lot types, and render nors
            update_pin_list_and_dicts_for_overlapping_polygons(gdf, index, row, "polygon_nu", "sibling_pin", same_geom_rows)
            
    # Remove rows with duplicated geometries, keeping only the first occurrence
    gdf = gdf[~gdf.duplicated(["geometry"], keep="first")]

        
    # ---------------------------------------------------
    # Flatten and preserve attributes of 'child polygons'
    # inside 'parent polygons'
    # ---------------------------------------------------

    # Iterate over rows
    for index, row in gdf.iterrows():
    
        # Get all polygons that are overlapping the current row
        overlapping_rows = gdf[gdf.geometry.within(row.geometry)]

        # Exclude polygons that are merely touching the border of the parent polygon
        overlapping_rows = overlapping_rows[~overlapping_rows.geometry.touches(row.geometry)]

        # Exclude the current row itself
        overlapping_rows = overlapping_rows[overlapping_rows.index != index]

        # Check if the row has any overlapping polygons
        if not overlapping_rows.empty:

            # Get the area of the polygon being processed
            pol_area = row["calculated"]

            # Get the area of all overlapping polygons
            overlapping_rows_area = overlapping_rows["calculated"].tolist()

            # Check if the area of the polygon is larger than all overlapping polygons
            if pol_area > max(overlapping_rows_area):

                # Preserve child polygons' pins, lot types, and render nors
                update_pin_list_and_dicts_for_overlapping_polygons(gdf, index, row, "polygon_nu", "child_pin", overlapping_rows)

                # Remove child polygon rows
                gdf.drop(index= overlapping_rows.index, inplace=True)           
      
    # Reset the index of the GeoDataFrame so index of "df" matches with "gdf"
    gdf.reset_index(drop=True, inplace=True)
    
    # Create dataframe containing individual lot types as columns
    df = pd.json_normalize(gdf["lot_type_dict"])

    # Convert numeric values to strings
    df = df.astype(str)
    
    # Replace all occurrences of "0" with an empty string
    df.replace("0", "", inplace=True)

    # Combine with original GeoDataFrame
    gdf = pd.concat([gdf, df], axis=1)
    
    # Drop the original "lot_type_dict" column
    gdf.drop(columns=["lot_type_dict"], inplace=True)
    
    return gdf

def create_spatial_index(gdf):
    idx = rtree.index.Index()
    for i, geometry in enumerate(gdf.geometry):
        idx.insert(i, geometry.bounds)
    return idx


def flatten_cadastre(cad_gdf, proj_crs, 
                     brisbane=False, 
                     adelaide=False,
                     merge_by_filled_polygon=False, 
                     merge_by_convex_hull=False):
    """
    Flatten "parent-child" and "sibling" cadastral polygons. "Parent" polygons contain
    smaller "child" polygons. "Sibling" polygons are perfectly overlapping each other.
    Brisbane cadastres have to be merged by lotplan. Merge building footprints to the parcel polygon.

    Note: Use on any cadastre datasets for any cities.

    Parameters:
        - cad_gdf: Input GeoDataFrame containing cadastre polygons.
        - proj_crs: Projected CRS for calculating area.
        - brisbane: Flag for merging cadastre polygons by lotplan (for Brisbane).
        - adelaide: Flag for dissolving cadastre polygons by plan and volume (for Adelaide).
        - merge_by_filled_polygon: Flag for merging building polygons with parent lot by filled polygon
        - merge_by_convex_hull: Flag for merging building polygons with parent lot by convex hull

    Returns:
        - Flattened GeoDataFrame with retained attributes.
        - GeoDataFrame with convex hull geometries (if merge_building_footprints is True).
    """
    
    # Make a copy of the input GeoDataFrame
    gdf = cad_gdf.copy()
    
    # Calculate area of each parcel
    gdf = gdf.to_crs(proj_crs)
    gdf["area"] = gdf.geometry.area
    
    
    # -----------------------------------------------------
    # Flatten 'sibling polygons'
    # -----------------------------------------------------
    
    # Remove polygons with duplicated geometry
    gdf = gdf[~gdf.duplicated(["geometry"], keep="first")]

    # Create a set to store indices of polygons to drop
    indices_to_drop = set()

    # Iterate over rows
    for index, row in gdf.iterrows():
        # Skip rows that are already marked for removal
        if index in indices_to_drop:
            continue

        # Find polygons that intersect with the current row and have the same area
        intersecting_rows = gdf[
            (gdf.geometry.intersects(row.geometry)) & 
            (gdf["area"] == row["area"])
        ]

        # Exclude the current row itself from intersecting_rows
        intersecting_rows = intersecting_rows[intersecting_rows.index != index]

        # Mark intersecting polygons for removal
        indices_to_drop.update(intersecting_rows.index)

    # Drop all marked indices from gdf
    gdf = gdf.drop(index=indices_to_drop).reset_index(drop=True)
        
    # ---------------------------------------------------
    # Remove 'child polygons' inside 'parent polygons'
    # ---------------------------------------------------

    # Iterate over rows
    for index, row in gdf.iterrows():
    
        # Get all polygons that are within the current row
        overlapping_rows = gdf[gdf.geometry.within(row.geometry.buffer(0.1))]

        # Exclude the current row itself
        overlapping_rows = overlapping_rows[overlapping_rows.index != index]

        # Check if the row has any overlapping polygons
        if not overlapping_rows.empty:

            # Get the area of the polygon being processed
            pol_area = row["area"]

            # Get the area of all overlapping polygons
            overlapping_rows_area = overlapping_rows["area"].tolist()

            # Check if the area of the polygon is larger than all overlapping polygons
            if pol_area > max(overlapping_rows_area):

                # Remove child polygon rows
                gdf.drop(index= overlapping_rows.index, inplace=True)           
      
    # Reset the index of the GeoDataFrame so index of "df" matches with "gdf"
    gdf.reset_index(drop=True, inplace=True)
           
        
    # ---------------------------------------------------
    # Merge polygons by lotplan (for Brisbane)
    # ---------------------------------------------------
    
    # Create spatial index
    spatial_index = create_spatial_index(gdf)
    
    if brisbane:
        # Create a set to store indices of polygons to drop
        indices_to_drop = set()

        # Iterate over rows
        for idx, row in gdf.iterrows():
            # Skip if this polygon has already been marked for removal
            if idx in indices_to_drop:
                continue

            # Find potential matches using the spatial index
            intersecting_indices = list(spatial_index.intersection(row.geometry.bounds))
            potential_matches = gdf.iloc[intersecting_indices]
            potential_matches = potential_matches[
                (potential_matches["lotplan"] == row["lotplan"]) & 
                (potential_matches.index != idx) & 
                (potential_matches.geometry.intersects(row.geometry))
            ]

            if not potential_matches.empty:
                # Create a merged polygon from the current polygon and all matching polygons
                polygons_to_merge = [gdf.loc[idx, "geometry"]] + potential_matches.geometry.tolist()

                try:
                    # Attempt to merge the polygons
                    merged_geometry = unary_union(polygons_to_merge)

                    # Update the current polygon's geometry with the merged geometry
                    gdf.at[idx, "geometry"] = merged_geometry

                    # Mark the matching polygons for removal
                    indices_to_drop.update(potential_matches.index)

                except Exception as e:
                    # If merge fails, continue with the next polygon
                    print(f"merge failed for {row.PFI}- {e}")
                    continue

        # Remove all merged polygons
        if indices_to_drop:
            gdf = gdf.drop(index=indices_to_drop)
            
        # Recalculate area
        gdf = gdf.drop(columns=["area"])
        gdf["area"] = gdf.geometry.area
        
        # Rereate the spatial index
        spatial_index = create_spatial_index(gdf)

        
    # ---------------------------------------------------
    # Merge by PLAN (for Adelaide)
    # ---------------------------------------------------
    
    if adelaide:
        if "plan_t" in gdf.columns: 
            # Filter the rows where plan_t is "C" or "S"
            gdf_dissolve = gdf[(gdf["plan_t"] == "C") | (gdf["plan_t"] == "S")]

            # Dissolve by plan
            gdf_dissolve = gdf_dissolve.dissolve(by = "plan")

            # Dissolve by volume
            gdf_dissolve = gdf_dissolve.dissolve(by = "volume")

            # Keep the rest of the parcels (not "C" or "S")
            gdf_remaining = gdf[~(gdf["plan_t"] == "C") | (gdf["plan_t"] == "S")]

            # Concatenate the dissolved and remaining data
            gdf = pd.concat([gdf_remaining, gdf_dissolve]).reset_index(drop=True)
        
        elif "PLAN_T" in gdf.columns:
            # Filter the rows where plan_t is "C" or "S"
            gdf_dissolve = gdf[(gdf["PLAN_T"] == "C") | (gdf["PLAN_T"] == "S")]

            # Dissolve by plan
            gdf_dissolve = gdf_dissolve.dissolve(by = "PLAN")

            # Dissolve by volume
            gdf_dissolve = gdf_dissolve.dissolve(by = "VOLUME")

            # Keep the rest of the parcels (not "C" or "S")
            gdf_remaining = gdf[~(gdf["PLAN_T"] == "C") | (gdf["PLAN_T"] == "S")]

            # Concatenate the dissolved and remaining data
            gdf = pd.concat([gdf_remaining, gdf_dissolve]).reset_index(drop=True)
            
        
        spatial_index = create_spatial_index(gdf)
 

    # ---------------------------------------------------
    # Merge building footprints (within block) to the 
    # larger polygon using convex hull and filled polygon
    # ---------------------------------------------------
    
    if merge_by_filled_polygon:
        
        # ----------------------
        # Filled polygon method
        # ----------------------
        
        # Create a set to store indices of polygons to drop
        indices_to_drop = set()

        # Iterate over rows
        for idx, row in gdf.iterrows():
            # Skip if this polygon has already been marked for removal
            if idx in indices_to_drop:
                continue

            # Handle Polygon and MultiPolygon separately
            if isinstance(row.geometry, Polygon):
                # For a single Polygon, create a "filled" version without holes
                filled_polygon = Polygon(row.geometry.exterior)
            elif isinstance(row.geometry, MultiPolygon):
                # For MultiPolygon, create a "filled" MultiPolygon without holes
                filled_polygon = MultiPolygon([Polygon(poly.exterior) for poly in row.geometry.geoms])
            
            # Get the polygons that are within the filled
            intersecting_indices = list(spatial_index.intersection(filled_polygon.bounds))
            potential_matches = gdf.iloc[intersecting_indices]
            potential_matches = potential_matches[
                potential_matches.buffer(-1).geometry.within(filled_polygon) & 
                (potential_matches.index != idx)
            ]

            if not potential_matches.empty:
                # Create a merged polygon from the current polygon and all matching polygons
                polygons_to_merge = [row.geometry] + potential_matches.geometry.tolist()

                try:
                    # Attempt to merge the polygons
                    merged_geometry = unary_union(polygons_to_merge)

                    # Update the current polygon's geometry with the merged geometry
                    gdf.at[idx, "geometry"] = merged_geometry

                    # Mark the matching polygons for removal
                    indices_to_drop.update(potential_matches.index)

                except Exception as e:
                    # If merge fails, continue with the next polygon
                    print(f"merge failed for {row.PFI}- {e}")
                    continue

        # Remove all merged polygons
        if indices_to_drop:
            gdf = gdf.drop(index=indices_to_drop).reset_index(drop=True)
            
            # Rereate the spatial index
            spatial_index = create_spatial_index(gdf)
        
        
    if merge_by_convex_hull:
        # ----------------------
        # Convex hull method
        # ----------------------
        
        # Apply convex hull to each geometry in the GeoDataFrame
        gdf["convex_hull"] = gdf.geometry.apply(lambda geom: geom.convex_hull)

        # Drop original geometry column and convex_hull geometry to actual geometry
        gdf_ch = gdf.drop(columns=["geometry"]).rename(columns={"convex_hull": "geometry"})

        # Calculate convex hull:area ratio
        gdf_ch["ch_area"] = gdf_ch.geometry.area
        gdf_ch["ch_to_area_ratio"] = gdf_ch["ch_area"] / gdf_ch["area"]

        # Create a set to store indices of polygons to drop
        indices_to_drop = set()

        # Iterate over rows
        for idx, row in gdf_ch.iterrows():
            # Skip if this polygon has already been marked for removal
            if idx in indices_to_drop:
                continue

            # Filter out parcels larger than 10,000 sq.m.
            if row["area"] < 5000 and row["ch_to_area_ratio"] < 3:

                # Get the polygons that are within the filled
                intersecting_indices = list(spatial_index.intersection(row.geometry.bounds))
                potential_matches = gdf.iloc[intersecting_indices]
                potential_matches = potential_matches[
                    row.geometry.contains_properly(potential_matches.copy().geometry.buffer(-1)) & 
                    (potential_matches.index != idx)
                ]

                if not potential_matches.empty:
                    # Create a merged polygon from the current polygon and all matching polygons
                    polygons_to_merge = [gdf.loc[idx, "geometry"]] + potential_matches.geometry.tolist()

                    try:
                        # Attempt to merge the polygons
                        merged_geometry = unary_union(polygons_to_merge)

                        # Update the current polygon's geometry with the merged geometry
                        gdf.at[idx, "geometry"] = merged_geometry

                        # Mark the matching polygons for removal
                        indices_to_drop.update(potential_matches.index)
                    except Exception as e:
                        # If merge fails, continue with the next polygon
                        print(f"merge failed for {row.PFI}- {e}")
                        continue

        # Remove all merged polygons
        if indices_to_drop:
            gdf = gdf.drop(index=indices_to_drop).reset_index(drop=True)
        
    return gdf


def sample_meshblocks(meshblocks, num_non_residential, sample_per_quintile, num_highest_dwelling_density, seed_value):
    """
    Sample a number of non-residential meshblocks from each land use type and a number of residential
    meshblocks from each dwelling density quantile.

    Args:
        meshblocks (GeoDataFrame): A GeoDataFrame containing meshblocks data.
        num_non_residential (int): Number of non-residential meshblocks to sample per land use type.
        sample_per_quintile (dict): Dictionary specifying the number of samples for each quantile.
        num_highest_dwelling_density (int): Number of meshblocks with highest dwelling density to be sampled(excluding meshblocks sampled by quintiles).
        seed_value (int): Seed value for reproducibility in random sampling.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the sampled meshblocks.
    """
    # Separate residential and non-residential meshblocks
    mb_non_residential = meshblocks[meshblocks["MB_CAT21"] != "Residential"].copy()
    mb_residential = meshblocks[meshblocks["MB_CAT21"] == "Residential"].copy()

    # Sample non-residential meshblocks
    mb_non_residential_sampled = mb_non_residential.groupby("MB_CAT21").apply(lambda x: x.sample(min(len(x), num_non_residential), random_state=seed_value))
    mb_non_residential_sampled.reset_index(drop=True, inplace=True)

    # Calculate quintiles based on "dwelling_density" for residential meshblocks
    quintiles = pd.qcut(mb_residential["dwelling_density"], q=5, labels=False)

    # Add quintiles as a new column to the residential GeoDataFrame
    mb_residential.loc[:, "quintile"] = quintiles

    # Sample residential meshblocks from each quintile
    def sample_quintile(group):
        quintile = group.name
        num_samples = sample_per_quintile.get(quintile, 10)  # Default to 10 samples if not specified for the quantile
        return group.sample(min(len(group), num_samples), random_state=seed_value)

    mb_residential_sampled = mb_residential.groupby("quintile").apply(sample_quintile)
    mb_residential_sampled.reset_index(drop=True, inplace=True)

    # Exclude already sampled residential meshblocks
    remaining_meshblocks = mb_residential[~mb_residential["MB_CODE21"].isin(mb_residential_sampled["MB_CODE21"])]

    # Select the top X residential meshblocks with the highest dwelling density from the remaining meshblocks
    mb_residential_highest_density = remaining_meshblocks.sort_values(by="dwelling_density", ascending=False).head(num_highest_dwelling_density)
    
    # Add a flag for top 50 Mesh Blocks sampling
    mb_residential_highest_density["top_density_flag"] = 1

    # Concatenate sampled residential and non-residential meshblocks
    mb_sampled = pd.concat([mb_residential_sampled, mb_non_residential_sampled, mb_residential_highest_density], ignore_index=True)

    return mb_sampled


def combine_nmai_features_mb(meshblock_codes, folder_path):
    """
    Load Nearmap AI features for specified meshblock codes from local parquet files and
    combine them into a single GeoDataFrame.

    Args:
        meshblock_codes (list): List of meshblock codes.
        folder_path (str): Path to the folder containing land cover map parquet files.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the concatenated land cover maps.
    """
    # Initialize an empty list to store GeoDataFrames
    dfs = []

    # Iterate through the list of meshblock codes
    for meshblock_code in meshblock_codes:
        # Construct the file path for the land cover map parquet file
        file_path = os.path.join(folder_path, f"{meshblock_code}_aifeatures.parquet")

        # Check if the file exists
        if os.path.exists(file_path):
            # Load the land cover map GeoDataFrame from the parquet file
            land_cover_map = gpd.read_parquet(file_path)

            # Append the land cover map GeoDataFrame to the list
            dfs.append(land_cover_map)
        else:
            print(f"File not found for meshblock code: {meshblock_code}")

    # Concatenate the list of GeoDataFrames into a single GeoDataFrame
    concatenated_gdf = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs="EPSG:4326")

    return concatenated_gdf


def spatial_join_largest_intersection(proj_crs, out_crs, left_df, right_df, id_col_left, intersection_area_col, id_col_right):
    """
    Spatial join based on largest intersection

    Args:
        proj_crs (str): CRS to be used for projecting geometries
        out_crs (str): CRS of the resulting GeoDataFrame
        left_df (gpd): Variable holding GeoPandas dataframe as left table in spatial join
        right_df (gpd): Variable holding GeoPandas dataframe as right table in spatial join
        id_col_left (str): Column containing unique feature IDs in left table
        intersection_area_col (str): Name of column for storing area of intersection between left and right tables
        id_col_right (str): Column containing unique feature IDs in left table - to append to left table based on largest intersection

    Returns:
        gpd: GeoPandas dataframe of left table with id_col_right appended based on largest intersection
    """    

    # Project both GeoDataFrames to the same coordinate reference system
    left_df = left_df.to_crs(proj_crs)
    right_df = right_df.to_crs(proj_crs)
    
    # Perform spatial overlay to find the intersection between the two GeoDataFrames
    overlay = gpd.overlay(left_df, right_df, how="intersection")
    
    # Calculate the area of intersection for each geometry and add it as a new column
    overlay[intersection_area_col] = overlay.geometry.area
    
    # Keep only the largest intersection for each feature in the left table
    out_df = overlay.sort_values([id_col_left, intersection_area_col], ascending=False).drop_duplicates(subset=[id_col_left], keep="first")
    
    # Select necessary columns for the final output
    out_df = out_df[[id_col_left, intersection_area_col, id_col_right]]
    
    # Merge the selected columns back into the left table
    left_df = left_df.merge(out_df, how="left", on=id_col_left).drop(columns=[intersection_area_col])
    
    # Reproject the resulting GeoDataFrame to the specified output coordinate reference system
    left_df = left_df.to_crs(out_crs)

    return left_df
        

def count_items_in_subdirectories(folder_path, logger):
    """
    Function to count the number of items (files and directories) in each sub-directory
    within a given folder path.
    
    Args:
    - folder_path (str): Path to the folder to search through.
    - logger (logging.Logger): Logger instance to log the information.- logger: 
    
    Returns:
    - None
    """
    # Iterate through each item (file or directory) in the given folder path
    for root, dirs, files in os.walk(folder_path):
        # Extract the name of the last directory in the current sub-directory path
        subdirectory_name = os.path.basename(root)
        # Count the number of items (files and directories)
        num_items = len(dirs) + len(files)
        # Print the name of the current sub-directory and the number of items in it
        logger.info(f"{subdirectory_name}: {num_items}")
        

def add_cad_attributes(row, nearmap_folder_path, proj_crs):
    mb_codes = row["overlapping_MB_CODE21"]
    
    # Load nmai_feature of the meshblock(s) overlapping the cadastre being processed
    nmais = []
    for mb_code in mb_codes:
        nmai = gpd.read_parquet(os.path.join(nearmap_folder_path, f"{mb_code}_aifeatures.parquet"))
        nmais.append(nmai)
    
    combined_nmai = pd.concat(nmais, ignore_index=True)
    
    # Extract 3d attributes from the attribute column to their own columns
    nmai_feature = CartoFeatureVis.extract_3d_attributes(combined_nmai)
    
    # Transform crs of nmai_feature to match with the cadastre
    nmai_feature = nmai_feature.to_crs(proj_crs)
    
    # Calculate unclipped area for each polygon
    nmai_feature["unclipped_area"] = nmai_feature.geometry.area
    
    # Clip nmai features to the extent of the cadastre
    nmai_feature_clip = gpd.clip(nmai_feature, row.geometry)
    
    # Calculate the area for each polygon
    nmai_feature_clip["area"] = nmai_feature_clip.geometry.area
    
    # Calculate coverage percentage of each building relative to the parcel area
    nmai_feature_clip["coverage_percent"] = (nmai_feature_clip["area"] / row["parcel_area"]) * 100
    
#     # Drop rows where description is "Building" AND (area/unclipped_area_sqm < 0.3 AND coverage_percent < 30)
#     mask = (
#         (nmai_feature_clip["description"] == "Building") & 
#         (nmai_feature_clip["area"] / nmai_feature_clip["unclipped_area_sqm"] < 0.3) &
#         (nmai_feature_clip["coverage_percent"] < 30)
#     )
#     nmai_feature_clip = nmai_feature_clip[~mask]
        
    attributes = {}
    
    # Get percentage areas of different feature classes
    for feature_class, feature_name in [
        ("Medium and High Vegetation (>2m)", "percent_med_high_veg"),
        ("Low Vegetation (0.5m-2m)", "percent_low_veg"),
        ("Very Low Vegetation (<0.5m)", "percent_very_low_veg"),
        ("Lawn Grass", "percent_lawn"),
        ("Natural (soft)", "percent_pervious"),
        ("Building", "percent_building"),
        ("Construction Site", "percent_construction_site")
    ]:
        area_sum = nmai_feature_clip[nmai_feature_clip["description"] == feature_class]["area"].sum()
        percent_area = round((area_sum / row["parcel_area"]) * 100, 2)
        attributes[feature_name] = percent_area
     
    # Get percentage impervious area by summing Concrete Slab and Asphalt
    concrete_area = nmai_feature_clip[nmai_feature_clip["description"] == "Concrete Slab"]["area"].sum()
    asphalt_area = nmai_feature_clip[nmai_feature_clip["description"] == "Asphalt"]["area"].sum()
    attributes["percent_impervious"] = round(((concrete_area + asphalt_area) / row["parcel_area"]) * 100, 2)
       
    # Get 3D building attributes    
    building = nmai_feature_clip[nmai_feature_clip["description"] == "Building"]
    
    if not building.empty:
        # Convert heightM to numeric values
        building["heightM"] = pd.to_numeric(building["heightM"], errors="coerce")
        
        # Replace "3+" stories with "3"
        building["num_stories"] = building["num_stories"].replace("3+", "3")

        # Convert num_stories to numeric values
        building["num_stories"] = pd.to_numeric(building["num_stories"], errors="coerce")
        
        # Get maximum height of buildings
        attributes["max_height"] = building["heightM"].max()
        
        # Get maximum number of stories
        attributes["max_stories"] = building["num_stories"].max()
        
        # Get number of buildings
        attributes["num_building"] = len(building)
        
        # Get ratio of clipped:unclipped building area
        attributes["building_clipped_unclipped_ratio"] = round(building["area"].sum()/building["unclipped_area"].sum(), 2)
        
        # Get maximum height of buildings
        max_height = building["heightM"].max()
        attributes["max_height"] = max_height
        
        # Get maximum number of stories
        max_stories = building["num_stories"].max()
        attributes["max_stories"] = max_stories
        
        # Create a flag if any building is missing 3D attributes
        missing_3d = ((building["heightM"].isna()) & (building["num_stories"].isna())).any()
        attributes["building_missing_3d"] = int(missing_3d)
       
    else:
        # If there are no buildings, set attributes to 0
        attributes["max_height"] = 0
        attributes["max_stories"] = 0
        attributes["num_building"] = 0
        attributes["building_clipped_unclipped_ratio"] = 0
        attributes["building_missing_3d"] = 0
    
    return pd.Series(attributes)

def detect_density_changes():
    """Detect changes in residential density.
    
    1. change in cadastre polygon shape and / or lot types
    2. large change in dominant building footprint
    """
    pass


def jsd(
    df_t1: gpd.GeoDataFrame,
    df_t2: gpd.GeoDataFrame,
    classes: list,
    fig_size: tuple = (7, 7),
    vmin: float = 0,
    vmax: float = 0.5,
    clipped_summary: bool = True, 
    basemap=cx.providers.CartoDB.Positron,
    save_df_path: Optional[str] = None,
    save_fig_path: Optional[str] = None,
):
    """Compute Jensen Shannon Distance between probability distributions. 

    This can be used to provide a composite (single value) measure of change in land cover
    between two time points considering multiple land cover classes. For example, a range of
    Nearmap AI rollup classes can be selected from two time points and the change between these
    two time points can be summarised by a single value. 

    A value of 0 indicates no change.

    A value of 1 indicates complete change. 

    Args:
        df_t1 (gpd.GeoDataFrame): GeoDataFrame of Nearmap AI rollup at time point 1.
        df_t2 (gpd.GeoDataFrame): GeoDataFrame of Nearmap AI rollup at time point 2.
        classes (list): list of Nearmap AI classes from rollup column heading prefixes (either "_clipped_area_sqm" or "_unclipped_area_sqm" will be appended in the function call).
        fig_size (tuple, optional): Figure size for JSD maps. Defaults to (7, 7).
        vmin (float, optional): Min JSD value to render. Defaults to 0.
        vmax (float, optional): Max JSD value to render. Defaults to 0.5.
        clipped_summary (bool, optional): Whether to use clipped or unclipped Nearmap AI rollup stats. Defaults to True.
        basemap (_type_, optional): Contextily basemap. Defaults to cx.providers.CartoDB.Positron.
        save_df_path (Optional[str], optional): Path to save parquet file of JSD values. Defaults to None.
        save_fig_path (Optional[str], optional): Path to save map of JSD values. Defaults to None.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame of JSD values. 
    """
    cols = []
    for ai_class in classes:
        if clipped_summary:
            col = ai_class + "_clipped_area_sqm"
        else:
            col = ai_class + "_unclipped_area_sqm"
        cols.append(col)

    # df 1 - earlier time point
    row_dfs = []
    for i, r in df_t1.iterrows():
        aoi_id = r["aoi_id"]
        cols_values = np.array(r[cols].tolist())
        denominator = sum(cols_values)
        cols_values_norm = cols_values / denominator
        df_r = pd.DataFrame({"aoi_id": [aoi_id], "ai_classes_norm": [cols_values_norm]})
        row_dfs.append(df_r)
    
    df_1_norm = pd.concat(row_dfs, axis=0, ignore_index=True)

    # df 2 - later time point
    row_dfs = []
    for i, r in df_t2.iterrows():
        aoi_id = r["aoi_id"]
        cols_values = np.array(r[cols].tolist())
        denominator = sum(cols_values)
        cols_values_norm = cols_values / denominator
        df_r = pd.DataFrame({"aoi_id": [aoi_id], "ai_classes_norm": [cols_values_norm]})
        row_dfs.append(df_r)
    
    df_2_norm = pd.concat(row_dfs, axis=0, ignore_index=True)

    df_merge = df_1_norm.merge(df_2_norm, how="inner", on="aoi_id")

    dfs_jsd = []
    
    # process jsd
    for i, r in df_merge.iterrows():
        aoi_id = r["aoi_id"]
        t1_arr = r["ai_classes_norm_x"]
        t2_arr = r["ai_classes_norm_y"]
        jsd = distance.jensenshannon(t1_arr, t2_arr)
        df_jsd_r = pd.DataFrame({"aoi_id": [aoi_id], "jsd": [jsd]})
        dfs_jsd.append(df_jsd_r)

    df_jsd = pd.concat(dfs_jsd, axis=0, ignore_index=True)
    df_1_geom = df_t1.loc[:, ["aoi_id", "geometry"]]
    
    gdf_jsd = df_1_geom.merge(df_jsd, how="inner", on="aoi_id")

    if save_df_path:
        gdf_jsd.to_parquet(save_df_path)
    
    fig, ax = plt.subplots(1, 1, figsize=fig_size, layout="constrained")
    gdf_jsd.plot(column="jsd", ax=ax, vmin=vmin, vmax=vmax, cmap="plasma", legend=True, legend_kwds={"shrink": 0.5})
    cx.add_basemap(ax, crs=gdf_jsd.crs, source=basemap)
    
    if save_fig_path:
        plt.savefig(save_fig_path, bbox_inches="tight", dpi=300)

    plt.show()

    return gdf_jsd


def create_grids(aoi, description, grid_size, targeted_projected_crs):
    """
    Create a geodataframe of grid parcels in EPSG:4326 coordinates representing the area of interest.

    Args:
        aoi (GeoDataFrame): Area of interest
        description (str): Name of area of interest
        grid_size (int): Side length of each square grid in meters
        targeted_projected_crs (str): Projected CRS for transformation

    Returns:
        A geodataframe with parcel_id, description, and geometry columns
    """
    
    # Transform AOI to the targeted projected CRS
    aoi_proj = aoi.to_crs(targeted_projected_crs)

    # Get the bounding box of the AOI
    aoi_bounds = aoi_proj.total_bounds  # Returns (minx, miny, maxx, maxy)
    x_min, y_min, x_max, y_max = aoi_bounds

    # Calculate the number of rows and columns for the grid
    num_cols = ceil((x_max - x_min) / grid_size)
    num_rows = ceil((y_max - y_min) / grid_size)

    # Create grid cells within the AOI's bounding box
    grid = []
    for row in range(num_rows):
        for col in range(num_cols):
            x_left = x_min + col * grid_size
            y_bottom = y_min + row * grid_size
            x_right = x_left + grid_size
            y_top = y_bottom + grid_size
            cell = Polygon([(x_left, y_bottom), (x_right, y_bottom), 
                            (x_right, y_top), (x_left, y_top)])
            # Add cell if it intersects with the AOI
            if cell.within(aoi_proj.unary_union):
                grid.append(cell)

    # Create a GeoDataFrame from the grid and add description/parcel_id
    grid_gdf = gpd.GeoDataFrame({"geometry": grid}, crs=targeted_projected_crs)
    grid_gdf["grid_id"] = range(1, len(grid_gdf) + 1)

    # Add description based on the provided input
    grid_gdf["aoi"] = description
    
    # Calculate the area for each polygon
    grid_gdf["area"] = grid_gdf.geometry.area

    return grid_gdf
